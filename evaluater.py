import os, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sam_vggt_model import build_sam_vggt
from utils.dataloader import create_dataloader, Resize
from utils.misc import sample_points_for_instances


def load_full_checkpoint(model, optimizer, ckpt_path, device="cuda", scaler=None, strict=True):
    """
    Loads a full checkpoint dict containing model/optimizer/scaler states.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    num_frames = ckpt.get("num_frames", None)
    print(f"[Checkpoint] Loaded {ckpt_path} (epoch={start_epoch}, num_frames={num_frames})")
    return ckpt


def make_points_from_labels(labels, valid_ids_list, k=10):
    """
    labels: [B,N,1,H,W] int instance IDs
    valid_ids_list: list length B, each tensor [K_i] of valid IDs (union over frames)
    returns:
        chosen_ids: [B]
        mask_cat: [B,H,W*N] (binary GT for chosen instance)
        point_coords_list, point_labels_list, point_frame_indices_list
    """
    device = labels.device
    B, N, _, H, W = labels.shape
    labels_2d = labels.squeeze(2)  # [B,N,H,W]

    # Choose random instance id per group
    chosen_ids = []
    for b in range(B):
        valid_ids = valid_ids_list[b].to(device)
        rid = torch.randint(0, len(valid_ids), (1,), device=device)
        chosen_ids.append(valid_ids[rid])
    chosen_ids = torch.stack(chosen_ids).squeeze(1)  # [B]

    # Build GT binary masks per frame
    chosen_ids_exp = chosen_ids[:, None, None, None]      # [B,1,1,1]
    binary_masks = (labels_2d == chosen_ids_exp).float()   # [B,N,H,W]

    # Concatenate along width (correct!)
    labels_cat = torch.cat([labels_2d[:, i] for i in range(N)], dim=2)        # [B,H,W*N]
    mask_cat   = torch.cat([binary_masks[:, i] for i in range(N)], dim=2)     # [B,H,W*N]

    # Sample global points on concatenated LABEL map
    sampled_points = sample_points_for_instances(labels_cat, chosen_ids, k=k) # [B,k,2]

    # Convert global -> per-frame coords (for model.forward)
    frame_width = W
    point_coords_list = []
    point_labels_list = []
    point_frame_indices_list = []
    for b in range(B):
        pts = sampled_points[b]      # [k,2]
        x_all, y_all = pts[:, 0], pts[:, 1]
        frame_idx = (x_all // frame_width).long()
        x_local = x_all % frame_width

        point_coords = torch.stack([x_local, y_all], dim=1).to(device)  # [k,2]
        point_labels = torch.ones(len(pts), dtype=torch.int64, device=device)

        point_coords_list.append(point_coords)
        point_labels_list.append(point_labels)
        point_frame_indices_list.append(frame_idx.to(device))

    return chosen_ids, mask_cat, point_coords_list, point_labels_list, point_frame_indices_list


def visualize_prediction(low_res_masks, H=1024, W=1024, N=8, title="pred"):
    """
    low_res_masks: [B,1,256,256*N] logits
    We upsample to [B,1,H,W*N], threshold >0.
    """
    pred_up = F.interpolate(
        low_res_masks.detach(),
        size=(H, W * N),
        mode="bilinear",
        align_corners=False
    )
    pred_bin = (pred_up > 0).float().cpu()  # [B,1,H,W*N]
    return pred_bin


def save_panorama(
    mask_hwN,
    save_path,
    N=8,
    W=1024,
    title="",
):
    """
    mask_hwN: [H, W*N] or [1, H, W*N]
    save_path: output PNG path
    """
    if mask_hwN.ndim == 3:
        mask_hwN = mask_hwN[0]  # [H, W*N]

    mask_np = mask_hwN.cpu().numpy()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(18, 4))
    plt.imshow(mask_np, cmap="gray")

    for i in range(1, N):
        plt.axvline(i * W, color="red", linestyle="--", linewidth=1)

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_and_visualize(
    root_dir="./hypersim",
    ckpt_path="checkpoints/sam_vggt_best.pth",
    device="cuda",
    num_frames=8,
    batch_size=1,
    num_batches_to_show=3,
):
    # Dataloader (NO augmentation)
    loader = create_dataloader(
        root_dir=root_dir,
        batch_size=batch_size,
        num_frames=num_frames,
        my_transforms=[Resize(size=[1024, 1024])]
    )

    # Build model
    model = build_sam_vggt().to(device)
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
    # Optimizer is optional for evaluation, but needed to load full checkpoint cleanly
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-3, weight_decay=0.0)

    # Load full checkpoint
    ckpt = load_full_checkpoint(model, optimizer, ckpt_path, device=device, scaler=None, strict=True)

    model.eval()

    shown = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            images = data["images"].to(device)   # [B,N,3,1024,1024]
            labels = data["labels"].to(device)   # [B,N,1,1024,1024]
            valid_ids_list = data["valid_ids"]   # list length B

            B, N, _, H, W = images.shape
            assert N == num_frames

            # Build prompts + GT mask_cat (same as training)
            chosen_ids, mask_cat, point_coords_list, point_labels_list, point_frame_indices_list = \
                make_points_from_labels(labels, valid_ids_list, k=10)

            # Forward
            outputs = model.forward(
                sam_pre=images,
                point_coords_list=point_coords_list,
                point_labels_list=point_labels_list,
                point_frame_indices_list=point_frame_indices_list,
                multimask_output=False,
                visualize=False,
            )
            low_res_masks = outputs["low_res_logits"]  # [B,1,256,256*N]

            # Upsample + threshold prediction
            pred_bin = visualize_prediction(low_res_masks, H=H, W=W, N=N)  # [B,1,H,W*N]

            # Visualize first element of batch
            b = 0
            save_panorama(mask_cat[b], save_path="model_output/gt_mask.png", N=N, W=W, title=f"GT binary mask_cat (chosen_id={int(chosen_ids[b])})")
            save_panorama(pred_bin[b, 0], save_path=f"model_output/pred_mask.png", N=N, W=W, title="Pred mask (upsampled low_res_logits > 0)")

            shown += 1
            if shown >= num_batches_to_show:
                break



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_and_visualize(
        root_dir="./hypersim",
        ckpt_path="checkpoints/sam_vggt_best.pth",
        device=device,
        num_frames=8,
        batch_size=1,
        num_batches_to_show=3,
    )
