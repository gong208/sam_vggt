import os, random, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from segment_anything.modeling import MaskDecoder, TwoWayTransformer
import shutil
import tempfile
import random
from torchvision import transforms
from sam_vggt_model import SamVGGT, build_sam_vggt
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, LargeScaleJitter, Resize, OnlineDataset
from utils.loss_mask import loss_masks
from utils.misc import sample_points_for_instances


# ---- Helper function to create non-distributed dataloader ----
class SceneGroupedBatchSampler:
    """
    Custom batch sampler that ensures groups of num_frames consecutive samples 
    come from the same scene/dataset. Each batch contains multiple such groups.
    """
    def __init__(self, datasets, batch_size, num_frames=4, shuffle=True):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        
        # Create indices grouped by dataset
        self.dataset_indices = []
        current_offset = 0
        for dataset in datasets:
            dataset_len = len(dataset)
            # Create indices for this dataset
            indices = list(range(current_offset, current_offset + dataset_len))
            self.dataset_indices.append(indices)
            current_offset += dataset_len
        
        # Create groups: each group contains num_frames consecutive samples from the same dataset
        self.groups = []
        for dataset_idx_list in self.dataset_indices:
            if len(dataset_idx_list) < num_frames:
                continue  # Skip datasets with fewer than num_frames images
            
            # Group consecutive indices into groups of num_frames
            for i in range(0, len(dataset_idx_list) - num_frames + 1, num_frames):
                group = dataset_idx_list[i:i + num_frames]
                if len(group) == num_frames:
                    self.groups.append(group)
            
            # If there are remaining samples, create a group with the last num_frames
            if len(dataset_idx_list) >= num_frames:
                remaining = len(dataset_idx_list) % num_frames
                if remaining > 0:
                    # Use the last num_frames samples
                    group = dataset_idx_list[-num_frames:]
                    self.groups.append(group)
        
        # Create batches by combining groups (each batch should be multiple of num_frames)
        num_groups_per_batch = batch_size // num_frames
        self.batches = []
        for i in range(0, len(self.groups), num_groups_per_batch):
            batch = []
            for j in range(i, min(i + num_groups_per_batch, len(self.groups))):
                batch.extend(self.groups[j])
            if len(batch) == batch_size:
                self.batches.append(batch)
        
        self.num_batches = len(self.batches)
    
    def __iter__(self):
        if self.shuffle:
            # Shuffle batches, but keep samples within each group in order
            batches = self.batches.copy()
            random.shuffle(batches)
            for batch in batches:
                yield batch
        else:
            for batch in self.batches:
                yield batch
    
    def __len__(self):
        return self.num_batches


def create_dataloader_non_distributed(name_im_gt_list, my_transforms=[], batch_size=1, num_frames=4):
    """
    Create a dataloader without distributed sampling (for single-GPU training).
    Ensures that groups of num_frames consecutive samples come from the same scene.
    """
    
    if len(name_im_gt_list) == 0:
        return None, None
    
    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8
    
    gos_datasets = []
    for i in range(len(name_im_gt_list)):
        gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms))
        gos_datasets.append(gos_dataset)
    
    gos_dataset = ConcatDataset(gos_datasets)
    
    # Create custom batch sampler that groups by scene
    batch_sampler = SceneGroupedBatchSampler(
        gos_datasets, 
        batch_size=batch_size, 
        num_frames=num_frames, 
        shuffle=True
    )
    
    dataloader = DataLoader(
        gos_dataset, 
        batch_sampler=batch_sampler,
        num_workers=num_workers_
    )
    
    return dataloader, gos_dataset



# ---- New training loop for SAM-VGGT with 4 continuous images ----
def train_sam_vggt(
    model,
    train_dataloader,
    optimizer,
    device: str = "cuda",
    num_frames: int = 8,
    epochs: int = 1,
    amp: bool = True,
):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for epoch in range(epochs):
        running_loss = 0.0
        running_mask_loss = 0.0
        running_dice_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(train_dataloader):
            # Inputs from dataloader
            # data["image"]:  [B_total, 3, 1024, 1024]
            # data["label"]:  [B_total, 1, 1024, 1024]
            images = data["image"].to(device)
            labels = data["label"].to(device)

            B_total = images.shape[0]
            assert B_total % num_frames == 0
            B = B_total // num_frames
            N = num_frames

            # Reshape into batches of groups
            images_bn = images.view(B, N, 3, 1024, 1024)           # [B,N,3,1024,1024]
            labels_bn = labels.view(B, N, 1, 1024, 1024)           # [B,N,1,1024,1024]

            # Build SAM preprocessing input
            sam_batch_list = []
            for b in range(B):
                sam_batch_list.append([images_bn[b, i].cpu() for i in range(N)])
            sam_pre, _ = model.preprocess_sam_images_batched(sam_batch_list)   # [B,N,3,1024,1024]
            sam_pre = sam_pre.to(device)

            # Build VGGT preprocessing input
            vggt_batch_list = []
            for b in range(B):
                frames = []
                for i in range(N):
                    img_np = images_bn[b, i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    frames.append(Image.fromarray(img_np))
                vggt_batch_list.append(frames)

            vggt_pre = model.preprocess_vggt_images_batched(vggt_batch_list).to(device)  # [B,N,3,Hv,Wv]

            # Compute instance masks per batch group
            point_coords_list = []
            point_labels_list = []
            point_frame_indices_list = []

            for b in range(B):
                labels_b = labels_bn[b]        # [N,1,1024,1024]
                labels_b_2d = labels_b.squeeze(1)   # [N,1024,1024]

                # Collect non-zero instance ids from all frames
                unique_ids = torch.unique(labels_b_2d)
                unique_ids = unique_ids[unique_ids != 0]

                target_instance_id = None
                mask_proportion = 0
                attempts = 0

                while mask_proportion < 0.0025 and attempts < 100:
                    rid = torch.randint(0, len(unique_ids), (1,)).item()
                    val = unique_ids[rid].item()
                    target_instance_id = torch.tensor(val, dtype=labels_b_2d.dtype, device=device)

                    binary_masks_b = (labels_b_2d == target_instance_id).float()
                    frame_pixels = labels_b_2d.shape[-1] * labels_b_2d.shape[-2]
                    proportions = [(binary_masks_b[i].sum().item() / frame_pixels) for i in range(N)]
                    mask_proportion = max(proportions)
                    attempts += 1

                binary_masks_b = binary_masks_b.unsqueeze(1)    # [N,1,H,W]

                # Concatenate masks along width: [N,1,H,W] → [1,H,W*N]
                mask_cat = torch.cat([binary_masks_b[i] for i in range(N)], dim=2)   # wrong dim, correct below

                # Fix: concatenate along width dim=3
                mask_cat = torch.cat([binary_masks_b[i] for i in range(N)], dim=3)   # [N,1,H,W*N]
                mask_cat = mask_cat[0].unsqueeze(0)                                 # [1,1,H,W*N]

                # Sample points from concatenated mask
                sampled = sample_points_for_instances(
                    mask_cat,              # [1,1,H,W*N]
                    target_instance_id.view(1),
                    k=10
                )[0]                       # [10,2]

                coords_b = []
                labels_b_out = []
                frame_idx_b = []

                frame_width = 1024

                for p in sampled:
                    x, y = p[0].item(), p[1].item()
                    fidx = int(x // frame_width)
                    x_local = x % frame_width
                    if 0 <= fidx < N:
                        coords_b.append(torch.tensor([x_local, y], device=device))
                        labels_b_out.append(torch.tensor(1, device=device))
                        frame_idx_b.append(fidx)

                point_coords_list.append(torch.stack(coords_b, dim=0))
                point_labels_list.append(torch.tensor(labels_b_out, device=device))
                point_frame_indices_list.append(torch.tensor(frame_idx_b, dtype=torch.long, device=device))

            # Forward pass
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model.forward(
                    sam_pre=sam_pre,
                    vggt_pre=vggt_pre,
                    point_coords_list=point_coords_list,
                    point_labels_list=point_labels_list,
                    point_frame_indices_list=point_frame_indices_list,
                    multimask_output=False,
                    visualize=False,
                )

            low_res_masks = outputs["low_res_logits"]    # [B,C,256,256*N]
            iou_preds = outputs["iou_predictions"]

            # Build target masks for each batch group
            target_masks = []
            for b in range(B):
                # binary mask cat = [N,1,H,W] → concatenate width → [H,W*N]
                mask_cat_b = torch.cat(
                    [labels_bn[b, i] for i in range(N)], dim=3
                )  # [1,1024,1024*N]
                target_masks.append(mask_cat_b)

            target_masks = torch.stack(target_masks, dim=0).to(device)  # [B,1,1024,1024*N]

            # Use only first mask channel if multimask_output=False
            pred_masks = low_res_masks[:, 0:1]  # [B,1,256,256*N]

            # Loss
            loss_mask, loss_dice = loss_masks(
                pred_masks.float(),
                target_masks.float(),
                num_masks=1.0,
            )
            loss = loss_mask + loss_dice

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_mask_loss += loss_mask.item()
            running_dice_loss += loss_dice.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                      f"Loss={running_loss/num_batches:.4f}, "
                      f"Mask={running_mask_loss/num_batches:.4f}, "
                      f"Dice={running_dice_loss/num_batches:.4f}")

        print(f"Completed epoch {epoch+1}: "
              f"Loss={running_loss/num_batches:.4f}, "
              f"Mask={running_mask_loss/num_batches:.4f}, "
              f"Dice={running_dice_loss/num_batches:.4f}")


def main():
    # 1. Define your datasets (same format as SAM training)
    train_datasets = [
        {
            "name": "Dataset1",
            "im_dir": "./ai_001_001_00/color",
            "gt_dir": "./ai_001_001_00/instance",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "Dataset2",
            "im_dir": "./ai_017_003_00/color",
            "gt_dir": "./ai_017_003_00/instance",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "Dataset3",
            "im_dir": "./ai_018_009_01/color",
            "gt_dir": "./ai_018_009_01/instance",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        # Add more datasets as needed
    ]
    
    # 2. Create dataloader (non-distributed version for single-GPU training)
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloader, _ = create_dataloader_non_distributed(
        train_im_gt_list,
        my_transforms=[
            RandomHFlip(),
            Resize(size=[1024,1024]),
            # LargeScaleJitter()  # Outputs 1024x1024 images
        ],
        batch_size=8,
        num_frames=8
    )
    
    # 3. Initialize your model
    model = build_sam_vggt()  # Your model initialization
    # model = None
    # 4. Set up optimizer
    # 1. Freeze encoders
    for p in model.sam.image_encoder.parameters():
        p.requires_grad = False

    for p in model.vggt.parameters():
        p.requires_grad = False

    for p in model.sam.prompt_encoder.parameters():
        p.requires_grad = False

    # 2. Train your fusion modules
    for p in model.embedding_fusion_mlp.parameters():
        p.requires_grad = True

    for p in model.prompt_fusion_mlp.parameters():
        p.requires_grad = True

    # 3. Freeze entire decoder first
    for p in model.sam.mask_decoder.parameters():
        p.requires_grad = False

    # 4. UNFREEZE ONLY the small "heads" inside the decoder
    #    (A) Hypernetwork MLPs (predict per-mask dynamic filters)
    for mlp in model.sam.mask_decoder.output_hypernetworks_mlps:
        for p in mlp.parameters():
            p.requires_grad = True

    #    (B) IoU prediction head
    for p in model.sam.mask_decoder.iou_prediction_head.parameters():
        p.requires_grad = True

    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)
    # optimizer = None
    # 5. Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 6. Call the training function
    train_sam_vggt(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        num_frames=8,  # Number of continuous images per batch
        epochs=1,  # Number of epochs to train
        amp=True  # Use mixed precision
    )
    
    print("Training completed!")

if __name__ == "__main__":
    # Check if running in distributed mode
    # if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
    #     main_distributed()
    # else:
    main()  # Single GPU mode

