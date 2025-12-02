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
    num_frames: int = 4,
    epochs: int = 1,
    amp: bool = True
):
    """
    Training loop for SAM-VGGT that:
    - Loads 4 continuous images from dataset per batch
    - Generates point prompts randomly from any of the 4 images
    - Adjusts point coordinates for concatenated layout
    - Trains for specified number of epochs
    
    Args:
        model: SamVGGT model
        train_dataloader: DataLoader
        optimizer: Optimizer
        device: Device string or torch.device
        num_frames: Number of continuous images per batch (default: 4)
        epochs: Number of epochs to train (default: 1)
        amp: Use mixed precision training
    """
    
    model.train()
    
    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # Training loop over epochs
    for epoch in range(epochs):
        
        running_loss = 0.0
        running_mask_loss = 0.0
        running_dice_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_dataloader):
            # data['image']: [B, C, 1024, 1024] - single image per sample
            # data['label']: [B, 1, 1024, 1024] - single mask per sample
            
            inputs = data['image']  # [B, C, 1024, 1024]
            labels = data['label']  # [B, 1, 1024, 1024]
            # Get image paths if available in dataset
            im_paths = data.get('ori_im_path', None)
            if im_paths is None:
                # Fallback: need to get paths from dataset indices
                im_paths = [None] * inputs.shape[0]
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            batch_size = inputs.shape[0]
            print("batch_size: ", batch_size)
            
            for group_start in range(0, batch_size, num_frames):
                group_end = min(group_start + num_frames, batch_size)
                if group_end - group_start < num_frames:
                    break  # Skip incomplete groups
                
                group_inputs = inputs[group_start:group_end]  # [num_frames, C, 1024, 1024]
                group_labels = labels[group_start:group_end]  # [num_frames, 1, 1024, 1024]
                # Convert images to numpy for saving/loading
                imgs_np = group_inputs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [4, 1024, 1024, C]
                
                all_point_coords = []
                all_point_labels = []
                point_frame_indices = []  # Track which frame each point belongs to
                
                # First, determine target instance ID from the first frame we sample points from
                # We'll use this same instance ID for all frames (tracking a single instance)
                unique_ids = torch.unique(group_labels.squeeze(1))
                unique_ids = unique_ids[unique_ids != 0]
                mask_proportion = 0
                max_attempts = 100  # Prevent infinite loop
                attempt = 0
                while (mask_proportion < 0.0025) and (attempt < max_attempts):
                    # Get a random index and extract the scalar value
                    rand_idx = torch.randint(0, len(unique_ids), (1,)).item()
                    target_instance_id_val = unique_ids[rand_idx].item()  # Extract scalar value
                    # Convert to same dtype as group_labels for proper comparison
                    target_instance_id = torch.tensor(target_instance_id_val, dtype=group_labels.dtype, device=group_labels.device)
                    binary_masks = (group_labels.squeeze(1) == target_instance_id).float()  # [4, 1024, 1024]
                    
                    # Calculate proportion for each frame: (number of pixels with instance) / (total pixels)
                    frame_size = group_labels.shape[-2] * group_labels.shape[-1]
                    proportions_per_frame = [binary_masks[i].sum().item() / frame_size for i in range(num_frames)]
                    mask_proportion = max(proportions_per_frame)
                    
                    attempt += 1
                    
                binary_masks = binary_masks.unsqueeze(1)  # [4, 1, 1024, 1024] 

                # Concatenate along width dimension: group_labels is [4, 1, 1024, 1024]
                # First remove channel dimension: [4, 1, 1024, 1024] -> [4, 1024, 1024]
                group_labels_2d = group_labels.squeeze(1)  # [4, 1024, 1024]
                # Concatenate along width (dim=1): [4, 1024, 1024] -> [1024, 4096]
                concatenated_mask = torch.cat([group_labels_2d[i] for i in range(num_frames)], dim=1)  # [1024, 4096]
                # Add batch dimension for sample_points_for_instances: [1024, 4096] -> [1, 1024, 4096]
                concatenated_mask_batch = concatenated_mask.unsqueeze(0)  # [1, 1024, 4096]

                # Sample 10 points from the concatenated mask
                # sample_points_for_instances expects instance_ids to be [B], so we need to reshape target_instance_id
                target_instance_id_batch = target_instance_id.unsqueeze(0) if target_instance_id.dim() == 0 else target_instance_id.reshape(1)
                sampled_points = sample_points_for_instances(concatenated_mask_batch, target_instance_id_batch, k=10)  # [1, 10, 2] with (x, y) coordinates
                sampled_points = sampled_points.squeeze(0)  # [10, 2]

                # Map points back to individual frames based on x-coordinate
                frame_width = group_inputs.shape[-1]
                for point_idx in range(sampled_points.shape[0]):
                    x_global, y = sampled_points[point_idx, 0].item(), sampled_points[point_idx, 1].item()
                    # Determine which frame this point belongs to
                    frame_idx = int(x_global // frame_width)
                    # Get local x coordinate within the frame
                    x_local = x_global % frame_width
                    
                    # Only add points that are within valid frame range (0-3)
                    if 0 <= frame_idx < num_frames:
                        all_point_coords.append(torch.tensor([[x_local, y]], device=sampled_points.device))
                        all_point_labels.append(torch.ones(1, device=sampled_points.device, dtype=torch.long))
                        point_frame_indices.append(frame_idx)
                
                # Save images to temporary files for model to load
                # Model expects image paths, so we create temp files
                temp_dir = tempfile.mkdtemp()
                image_paths = []
                
                for i in range(num_frames):
                    # Save image to temp file
                    temp_path = os.path.join(temp_dir, f"frame_{i}.png")
                    img_pil = Image.fromarray(imgs_np[i])
                    img_pil.save(temp_path)
                    image_paths.append(temp_path)
                # Concatenate all points
                point_coords_concat = torch.cat(all_point_coords, dim=0)  # [N_points, 2]
                point_labels_concat = torch.cat(all_point_labels, dim=0)  # [N_points]
                point_frame_indices = torch.tensor(point_frame_indices, dtype=torch.long)  # [N_points]
                print("point coords shape: ", point_coords_concat.shape)
                point_coords_list = [point_coords_concat.cpu()]  # [N_points, 2] in concatenated coords
                point_labels_list = [point_labels_concat.cpu()]  # [N_points]
                point_frame_indices_list = [point_frame_indices.cpu()]  # [N_points] - frame index for each point
                # Feed to model
                
            with torch.cuda.amp.autocast(enabled=amp):
                # Model forward - coordinates are already in concatenated space
                outputs = model.forward(
                    batch_image_paths=[image_paths],
                    point_coords_list=point_coords_list,
                    point_labels_list=point_labels_list,
                    point_frame_indices_list=point_frame_indices_list,  # Per-point frame indices
                    multimask_output=False,
                    return_embeddings=False,
                )
            
            outputs = outputs[0]
            decoder_masks = outputs["low_res_logits"]  # [1, C, H, W*N] or similar
            
            _, C_mask, _, _ = decoder_masks.shape

            # Concatenate labels along width dimension
            labels_concat = torch.cat([binary_masks[i] for i in range(num_frames)], dim=3) # shape: [1,1,1024,1024*num_frames]

            if C_mask > 1:
                # Use the first mask channel for loss (or select best)
                decoder_masks_for_loss = decoder_masks[:, 0:1, :, :]  # [1, 1, H_mask, W_mask]
            else:
                decoder_masks_for_loss = decoder_masks  # [1, 1, H_mask, W_mask]
            print("decoder_masks_for_loss shape: ", decoder_masks_for_loss.shape)
            print("labels_concat shape: ", labels_concat.shape)
            loss_mask, loss_dice = loss_masks(
                decoder_masks_for_loss.float(), 
                labels_concat.float(), 
                num_masks=1.0
            )
            
            # Total loss
            loss = loss_mask + loss_dice
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Clean up temp files
            shutil.rmtree(temp_dir)
            
            # Accumulate losses for logging
            running_loss += loss.item()
            running_mask_loss += loss_mask.item()
            running_dice_loss += loss_dice.item()
            num_batches += 1
            
            # Log progress periodically
            if (batch_idx + 1) % 50 == 0:
                avg_loss = running_loss / num_batches
                avg_mask = running_mask_loss / num_batches
                avg_dice = running_dice_loss / num_batches
                print(f"[Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}] "
                        f"Loss: {avg_loss:.4f} (Mask: {avg_mask:.4f}, Dice: {avg_dice:.4f})")
        
        # Log epoch summary
        if num_batches > 0:
            avg_loss = running_loss / num_batches
            avg_mask = running_mask_loss / num_batches
            avg_dice = running_dice_loss / num_batches
            print(f"==> Epoch {epoch+1}/{epochs} completed. "
                  f"Avg Loss: {avg_loss:.4f} (Mask: {avg_mask:.4f}, Dice: {avg_dice:.4f})")


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

