import os, json, random, time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sam_vggt_model import build_sam_vggt
from utils.dataloader import create_dataloader, Resize
from utils.misc import sample_points_for_instances
from utils.loss_mask import loss_masks


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

import os, numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def visualize_logits_panorama(low_res_logits, H=1024, W=1024, N=4, 
                             upsample=True, q=99, title="logits", save_path=None):
    """
    low_res_logits: [B,1,256,256*N] logits
    Returns: [H, W*N] numpy array for the first batch element.
    """
    x = low_res_logits.detach()
    if upsample:
        x = F.interpolate(x, size=(H, W*N), mode="bilinear", align_corners=False)  # [B,1,H,W*N]
    x = x[0,0].cpu().numpy()  # [H, W*N]

    # robust symmetric range centered at 0
    vmax = np.percentile(np.abs(x), q)
    vmin = -vmax

    plt.figure(figsize=(18,4))
    plt.imshow(x, cmap="coolwarm", vmin=vmin, vmax=vmax)
    for i in range(1, N):
        plt.axvline(i * W, color="black", linestyle="--", linewidth=1)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.title(f"{title} (robust range ±p{q}(|logit|))")
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
    return x


def save_panorama(
    mask_hwN,
    save_path,
    N=4,
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


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_to_rgb(feat_nchw: torch.Tensor, fit_pca=None, eps=1e-6):
    """
    feat_nchw: [N, C, H, W] float tensor
    Returns:
        rgb: [N, H, W, 3] in [0,1]
        pca: fitted PCA object (so you can reuse across views or across models)
    """
    N, C, H, W = feat_nchw.shape
    X = feat_nchw.permute(0, 2, 3, 1).reshape(-1, C).detach().cpu().numpy()  # [(N*H*W), C]

    if fit_pca is None:
        pca = PCA(n_components=3, random_state=0)
        Z = pca.fit_transform(X)
    else:
        pca = fit_pca
        Z = pca.transform(X)

    Z = Z.reshape(N, H, W, 3)

    # robust normalize per-channel to [0,1]
    rgb = np.zeros_like(Z)
    for k in range(3):
        ch = Z[..., k]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        rgb[..., k] = (ch - lo) / (hi - lo + eps)
    rgb = np.clip(rgb, 0, 1)
    return rgb, pca

def show_rgb_grid(rgb, title, save_path, ncols=4):
    N = rgb.shape[0]
    ncols = min(ncols, N)
    nrows = int(np.ceil(N / ncols))
    plt.figure(figsize=(3.2*ncols, 3.2*nrows))
    for i in range(N):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(rgb[i])
        plt.axis("off")
        plt.title(f"view {i}")
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import matplotlib.pyplot as plt

def cross_view_attention_maps(vggt_nchw: torch.Tensor, ref_view=0, temp=0.07, eps=1e-6):
    """
    vggt_nchw: [N, C, H, W]
    Return:
      attn: [N, H, W] where attn[j,h,w] = softmax_j(sim(ref,j))
    """
    N, C, H, W = vggt_nchw.shape
    x = vggt_nchw  # [N,C,H,W]

    # L2 normalize across channel
    x = x / (x.norm(dim=1, keepdim=True) + eps)

    ref = x[ref_view:ref_view+1]  # [1,C,H,W]
    # cosine sim per view: sum_c ref*cand
    sim = (ref * x).sum(dim=1)  # [N,H,W]
    attn = torch.softmax(sim / temp, dim=0)  # softmax over views
    return attn.detach().cpu().numpy(), sim.detach().cpu().numpy()

def visualize_probs_panorama(low_res_logits, H=1024, W=1024, N=4, title="probs", save_path=None):
    x = low_res_logits.detach()
    x = F.interpolate(x, size=(H, W*N), mode="bilinear", align_corners=False)
    p = x.sigmoid()[0,0].cpu().numpy()

    plt.figure(figsize=(18,4))
    plt.imshow(p, cmap="viridis", vmin=0.0, vmax=1.0)
    for i in range(1, N):
        plt.axvline(i * W, color="red", linestyle="--", linewidth=1)
    plt.colorbar(fraction=0.03, pad=0.02)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
    return p

def logit_histogram(low_res_logits, target_masks, H=1024, W=1024, N=4, bins=80, save_path=None):
    # upsample logits to target size
    x = F.interpolate(low_res_logits.detach(), size=(H, W*N), mode="bilinear", align_corners=False)
    x = x[0,0].cpu().numpy()  # [H, W*N]

    y = target_masks.detach()
    # Handle different tensor dimensions: [B,1,H,W*N], [B,H,W*N], or [H,W*N]
    if y.dim() == 4:  # [B, 1, H, W*N]
        y = y[0, 0]
    elif y.dim() == 3:  # [B, H, W*N]
        y = y[0]
    # else: y is already 2D [H, W*N]
    y = y.cpu().numpy().astype(np.float32)  # [H, W*N] in {0,1}

    # Ensure x and y have the same shape
    assert x.shape == y.shape, f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"

    fg = x[y > 0.5]
    bg = x[y <= 0.5]

    plt.figure(figsize=(6,4))
    # use same range for fair comparison
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    plt.hist(bg, bins=bins, range=(lo, hi), alpha=0.6, label="background")
    plt.hist(fg, bins=bins, range=(lo, hi), alpha=0.6, label="foreground")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.legend()
    plt.title("Logit distribution (fg vs bg)")
    plt.xlabel("logit")
    plt.ylabel("count")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    # helpful stats
    stats = {
        "fg_mean": float(np.mean(fg)) if fg.size else None,
        "bg_mean": float(np.mean(bg)) if bg.size else None,
        "fg_p95": float(np.percentile(fg, 95)) if fg.size else None,
        "bg_p95": float(np.percentile(bg, 95)) if bg.size else None,
        "overall_mean": float(np.mean(x)),
        "overall_p95": float(np.percentile(x, 95)),
        "overall_p99": float(np.percentile(x, 99)),
    }
    return stats


def show_attn(attn, ref_view=0, save_path="model_output/vggt_attn.png"):
    N = attn.shape[0]
    plt.figure(figsize=(3.2*N, 3.2))
    for j in range(N):
        plt.subplot(1, N, j+1)
        plt.imshow(attn[j], cmap="viridis")
        plt.axis("off")
        plt.title(f"w(ref={ref_view}→{j})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_and_visualize(
    root_dir="./hypersim",
    ckpt_path="checkpoints/sam_vggt_best.pth",
    device="cuda",
    num_frames=4,
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
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-1, weight_decay=0.0)

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
            _ = visualize_logits_panorama(low_res_masks, H=H, W=W, N=N,
                                        title="Pred logits", save_path="model_output/pred_logits.png")
            _ = visualize_probs_panorama(low_res_masks, H=H, W=W, N=N,
                             title="Pred sigmoid probabilities",
                             save_path="model_output/pred_probs.png")
            stats = logit_histogram(low_res_masks, mask_cat, H=H, W=W, N=N,
                        save_path="model_output/logit_hist.png")
            print(stats)

            # Upsample + threshold prediction
            pred_bin = visualize_prediction(low_res_masks, H=H, W=W, N=N)  # [B,1,H,W*N]
            print("low_res min/max:", low_res_masks.min().item(), low_res_masks.max().item())
            print("pred_bin mean:", pred_bin.mean().item())  # should be ~1.0 if all positive

            # Visualize first element of batch
            b = 0
            save_panorama(mask_cat[b], save_path="model_output/gt_mask.png", N=N, W=W, title=f"GT binary mask_cat (chosen_id={int(chosen_ids[b])})")
            save_panorama(pred_bin[b, 0], save_path=f"model_output/pred_mask.png", N=N, W=W, title="Pred mask (upsampled low_res_logits > 0)")
            vggt_feats = outputs["vggt_feats"]
            sam_feats = outputs["sam_feats"]
            B, N = vggt_feats.shape[:2]
            assert B == 1

            vggt_nchw = vggt_feats[0]  # [N,2048,64,64]
            sam_nchw  = sam_feats[0]   # [N, 256,64,64]

            # Fit PCA on VGGT across all views & pixels
            vggt_rgb, vggt_pca = pca_to_rgb(vggt_nchw, fit_pca=None)
            show_rgb_grid(vggt_rgb, "VGGT embedding PCA (3 components as RGB)", save_path="model_output/vggt_pca.png")

            # Fit PCA on SAM separately
            sam_rgb, sam_pca = pca_to_rgb(sam_nchw, fit_pca=None)
            show_rgb_grid(sam_rgb, "SAM embedding PCA (3 components as RGB)", save_path="model_output/sam_pca.png")

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
                pred_masks.float(),
                target_masks.float(),
                ce_mode="dense",
                use_pos_weight=True,
                dense_ce_on="src",
                num_masks=1.0,
                debug=True,
            )
            print(f"Loss mask: {loss_mask.item()}, Loss dice: {loss_dice.item()}")
            shown += 1
            if shown >= num_batches_to_show:
                break



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_and_visualize(
        root_dir="./hypersim",
        ckpt_path="checkpoints/sam_vggt_best.pth",
        device=device,
        num_frames=4,
        batch_size=1,
        num_batches_to_show=1,
    )
