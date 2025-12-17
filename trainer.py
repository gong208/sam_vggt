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
        name=f"experiment_epoch{epochs}_lr{optimizer.param_groups[0]['lr']}",
    )
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_loss = float("inf")

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
            print(f"B: {B}, N: {N}, H: {H}, W: {W}")
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
            # print(f"target_masks shape: {target_masks.shape}")
            pred_masks = low_res_masks[:, 0:1]
            num_masks = float(pred_masks.shape[0])
            # ------------------------------------------------------------------
            #  Loss
            # ------------------------------------------------------------------
            loss_mask, loss_dice = loss_masks(
                pred_masks.float(), target_masks.float(),
                num_masks=1.0, debug=True,
                ce_mode="dense",
                use_pos_weight=False,
                dense_ce_on="src",
            )
            loss = loss_mask + 5.0 * loss_dice
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
        os.makedirs("checkpoints", exist_ok=True)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),  # if using AMP
            "loss": running_loss / num_batches,
            "num_frames": num_frames,
        }


        if epoch == 0 or running_loss < best_loss:
            best_loss = running_loss
            torch.save(checkpoint, "checkpoints/sam_vggt_best.pth")

    wandb.finish()


def main():
    # 1. Define your datasets (same format as SAM training)

    train_loader = create_dataloader(
        root_dir="./hypersim",
        batch_size=1,
        num_frames=4,
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
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-2, weight_decay=0.0)
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
        num_frames=1,  # Number of continuous images per batch
        epochs=100,  # Number of epochs to train
        amp=False  # Use mixed precision
    )
    time_end = time.time()
    print(f"Training completed in {time_end - time_start} seconds")
    print("Training completed!")

if __name__ == "__main__":
    main()  # Single GPU mode

