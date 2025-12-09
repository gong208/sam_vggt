import os, random, math
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
from utils.dataloader import create_dataloader, RandomHFlip, Resize
from utils.loss_mask import loss_masks
from utils.misc import sample_points_for_instances
import time
import wandb
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


def train_sam_vggt(
    model,
    train_dataloader,
    optimizer,
    device="cuda",
    epochs=1,
    amp=True,
    num_frames=8,
):

    wandb.init(
        project="sam-vggt-multiview",
        config={
            "epochs": epochs,
            "batch_size": train_dataloader.batch_size,
            "num_frames": num_frames,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "amp": amp,
            "consecutive": True,
            "scene": 'ai_001_001_00'
        },
        name=f"experiment_epoch{epochs}_bs{train_dataloader.batch_size}",
    )
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for epoch in range(epochs):
        running_loss = 0
        running_mask_loss = 0
        running_dice_loss = 0
        num_batches = 0

        for batch_idx, data in enumerate(train_dataloader):

            # ------------------------------------------------------------------
            #  Extract batched inputs from dataloader
            # ------------------------------------------------------------------
            images = data["images"].to(device)       # [B, N, 3, 1024, 1024]
            labels = data["labels"].to(device)       # [B, N, 1, 1024, 1024]
            # print(f"labels shape: {labels.shape}")
            valid_ids_list = data["valid_ids"]       # list of B tensors, e.g. [tensor([3,10,...]), ...]

            B, N, _, H, W = images.shape
            assert N == num_frames

            # ------------------------------------------------------------------
            #  Random instance selection PER GROUP using offline valid-IDs
            # ------------------------------------------------------------------
            chosen_ids = []
            for b in range(B):
                valid_ids = valid_ids_list[b]
                rid = torch.randint(0, len(valid_ids), (1,))
                chosen_ids.append(valid_ids[rid])

            # stack shape = [B]
            chosen_ids = torch.stack(chosen_ids).to(device).squeeze(1)  # => [B]

            # ------------------------------------------------------------------
            #  Build binary mask for each (B, N) and then concatenate width
            # ------------------------------------------------------------------
            # labels: [B,N,1,H,W] → squeeze for comparison
            labels_2d = labels.squeeze(2)            # [B,N,H,W]

            # chosen_ids: [B] → match dims for broadcasting
            chosen_ids_exp = chosen_ids[:, None, None, None]   # [B,1,1,1]
            # print("chosen_ids.shape =", chosen_ids.shape)
            # print("chosen_ids_exp.shape =", chosen_ids_exp.shape)

            # print(f"labels_2d shape: {labels_2d.shape}")
            binary_masks = (labels_2d == chosen_ids_exp).float()   # [B,N,H,W]
            # print(f"binary_masks shape: {binary_masks.shape}")

            # Reshape to concatenate across frames
            # mask_cat = binary_masks.reshape(B, H, W * N)  # [B,H,W*N]

            # labels_cat = labels_2d.reshape(B, H, W * N)   # [B, H, W*N]
            labels_cat = torch.cat([labels_2d[:, i] for i in range(N)], dim=2)     # [B, H, W*N]
            mask_cat   = torch.cat([binary_masks[:, i] for i in range(N)], dim=2) # [B, H, W*N]

            sampled_points = sample_points_for_instances(
                labels_cat,        # correct
                chosen_ids,        # [B]
                k=10
            )

            # sampled_points[b] = tensor [K, 2]

            # ------------------------------------------------------------------
            #  Convert global concatenated coords → per-frame coords
            # ------------------------------------------------------------------
            frame_width = W

            point_coords_list = []
            point_labels_list = []
            point_frame_indices_list = []

            for b in range(B):
                pts = sampled_points[b]       # [K,2]
                x_all, y_all = pts[:, 0], pts[:, 1]

                frame_idx = (x_all // frame_width).long()  # which of N frames
                x_local = x_all % frame_width

                point_coords = torch.stack([x_local, y_all], dim=1).to(device)
                point_labels = torch.ones(len(pts), dtype=torch.int64, device=device)

                point_coords_list.append(point_coords)
                point_labels_list.append(point_labels)
                point_frame_indices_list.append(frame_idx.to(device))

            # ------------------------------------------------------------------
            #  SAM + VGGT Preprocessing (B,N,3,H,W) → pre-tensors
            # ------------------------------------------------------------------
            # SAM preprocess is already batched:
            # sam_pre, _ = model.preprocess_sam_images_batched(images)   # [B,N,3,1024,1024]

            sam_pre = images.to(device)

            # ------------------------------------------------------------------
            #  Forward pass
            # ------------------------------------------------------------------
            forward_start = time.time()
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model.forward(
                    sam_pre=sam_pre,
                    point_coords_list=point_coords_list,
                    point_labels_list=point_labels_list,
                    point_frame_indices_list=point_frame_indices_list,
                    multimask_output=False,
                    visualize=False,
                )
            forward_end = time.time()
            # print(f"Forward pass completed in {forward_end - forward_start} seconds")
            low_res_masks = outputs["low_res_logits"]    # [B,1,256,256*N]

            # ------------------------------------------------------------------
            #  Build target masks for loss
            # ------------------------------------------------------------------
            target_masks = mask_cat.unsqueeze(1).to(device)   # [B,1,H,W*N]
            # print(f"low_res_masks shape: {low_res_masks.shape}")
            pred_masks = low_res_masks[:, 0:1]

            # ------------------------------------------------------------------
            #  Loss
            # ------------------------------------------------------------------
            loss_mask, loss_dice = loss_masks(
                pred_masks.float(),
                target_masks.float(),
                num_masks=1.0,
            )
            loss = loss_mask + loss_dice
            wandb.log({
                "loss/total": loss.item(),
                "loss/mask": loss_mask.item(),
                "loss/dice": loss_dice.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "step": batch_idx + epoch * len(train_dataloader)
            })

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # ------------------------------------------------------------------
            #  Logging
            # ------------------------------------------------------------------
            running_loss += loss.item()
            running_mask_loss += loss_mask.item()
            running_dice_loss += loss_dice.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                    f"Loss={running_loss/num_batches:.4f}, "
                    f"Mask={running_mask_loss/num_batches:.4f}, "
                    f"Dice={running_dice_loss/num_batches:.4f}"
                )

        print(
            f"Completed epoch {epoch+1}: "
            f"Loss={running_loss/num_batches:.4f}, "
            f"Mask={running_mask_loss/num_batches:.4f}, "
            f"Dice={running_dice_loss/num_batches:.4f}"
        )
        wandb.log({
            "epoch_loss/total": running_loss / num_batches,
            "epoch_loss/mask": running_mask_loss / num_batches,
            "epoch_loss/dice": running_dice_loss / num_batches,
            "epoch": epoch + 1
        })
    wandb.finish()


def main():
    # 1. Define your datasets (same format as SAM training)

    train_loader = create_dataloader(
        root_dir="./hypersim",
        batch_size=1,
        num_frames=8,
        my_transforms=[
            Resize(size=[1024,1024]),
        ]
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
        p.requires_grad = True

    # 4. UNFREEZE ONLY the small "heads" inside the decoder
    #    (A) Hypernetwork MLPs (predict per-mask dynamic filters)
    # for mlp in model.sam.mask_decoder.output_hypernetworks_mlps:
    #     for p in mlp.parameters():
    #         p.requires_grad = True

    # #    (B) IoU prediction head
    # for p in model.sam.mask_decoder.iou_prediction_head.parameters():
    #     p.requires_grad = True

    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)
    # optimizer = None
    # 5. Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 6. Call the training function
    time_start = time.time()
    train_sam_vggt(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        num_frames=8,  # Number of continuous images per batch
        epochs=100,  # Number of epochs to train
        amp=True  # Use mixed precision
    )
    time_end = time.time()
    print(f"Training completed in {time_end - time_start} seconds")
    print("Training completed!")

if __name__ == "__main__":
    main()  # Single GPU mode

