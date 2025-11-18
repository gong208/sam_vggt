# vggt_encode_and_visualize.py  (PCA-RGB)
import os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square

# ---------- helpers
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_color_heatmap(norm_01: torch.Tensor, out_path: str, cmap="magma", gamma=1.0): 
    """ norm_01: [H,W] in [0,1] -> colored PNG """ 
    h = norm_01.detach().float().cpu().clamp(0, 1) 
    if gamma != 1.0: 
        h = h.pow(gamma) 
    rgb = cm.get_cmap(cmap)(h.numpy())[...,:3] # drop alpha 
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(out_path)


def save_rgb_image(rgb_01: np.ndarray, out_path: str):
    """ rgb_01: [H,W,3] in [0,1] """
    Image.fromarray((np.clip(rgb_01,0,1) * 255).astype(np.uint8)).save(out_path)

def global_minmax_norm_bnhw(m: torch.Tensor, across_batch: bool = False) -> torch.Tensor:
    """
    Normalize a stack of maps with one global min/max per scene.

    Args:
        m: [B, N, H, W] tensor (e.g., L2 maps per frame)
        across_batch: if True, use a single min/max across the entire batch
                      (all scenes). Default False = one min/max per scene (per B).

    Returns:
        m01: [B, N, H, W] normalized to [0,1] with global (scene-level) min/max.
    """
    eps = 1e-6
    if across_batch:
        mn = m.amin()                   # scalar
        mx = m.amax()                   # scalar
    else:
        mn = m.amin(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
        mx = m.amax(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
    return (m - mn) / (mx - mn + eps)


def get_patch_size(model) -> int:
    cand_attrs = [
        "backbone.patch_size",
        "backbone.vit.patch_size",
        "backbone.vit.patch_embed.patch_size",
        "backbone.patch_embed.patch_size",
    ]
    for ca in cand_attrs:
        try:
            obj = model
            for part in ca.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, (tuple, list)):
                return int(obj[0])
            return int(obj)
        except Exception:
            pass
    return 14  # DINOv2 ViT-14 default

def pca_rgb_per_frame(patch_btpc: torch.Tensor, gh: int, gw: int, H: int, W: int) -> np.ndarray:
    """
    patch_btpc: [B, N, P, C] float tensor of patch tokens (special tokens already removed)
    returns: [B, N, H, W, 3] in [0,1]
    """
    B, N, P, C = patch_btpc.shape
    assert P == gh * gw, f"P={P} must equal gh*gw={gh*gw}"
    out = np.zeros((B, N, H, W, 3), dtype=np.float32)

    # Work in float32 on CPU for stable SVD/PCA
    X_all = patch_btpc.detach().to(torch.float32).cpu()

    for b in range(B):
        for i in range(N):
            X = X_all[b, i]  # [P, C]

            # Try fast low-rank PCA; fallback to SVD if unavailable
            try:
                # centers internally if center=True (PyTorch >=1.9)
                U, S, V = torch.pca_lowrank(X, q=3, center=True)
                Y = X @ V[:, :3]  # [P,3]
            except Exception:
                Xc = X - X.mean(dim=0, keepdim=True)
                U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                V = Vh.transpose(-2, -1)[:, :3]  # [C,3]
                Y = Xc @ V                       # [P,3]

            # Min-max per channel to [0,1]
            ymin = Y.min(dim=0, keepdim=True).values
            ymax = Y.max(dim=0, keepdim=True).values
            Y = (Y - ymin) / (ymax - ymin + 1e-6)  # [P,3] in [0,1]

            # Reshape to grid and upsample back to (H,W)
            rgb = Y.reshape(gh, gw, 3).permute(2, 0, 1).unsqueeze(0)  # [1,3,gh,gw]
            rgb_up = F.interpolate(rgb, size=(H, W), mode="bilinear", align_corners=False)[0]  # [3,H,W]
            out[b, i] = rgb_up.permute(1, 2, 0).numpy()  # [H,W,3]

    return out

# ---------- config
device = "cuda" if torch.cuda.is_available() else "cpu"
scene_images = [
    "vggt/examples/kitchen/images/00.png",
    "vggt/examples/kitchen/images/01.png",
    "vggt/examples/kitchen/images/02.png",
    "vggt/examples/kitchen/images/03.png",
]
out_dir = "vggt_encoder_viz_kitchen"
ensure_dir(out_dir)

# ---------- load model + local weights
ckpt_path = "vggt/checkpoints/model.pt"

def load_local_vggt(ckpt: str, device="cuda"):
    model = VGGT()
    raw = torch.load(ckpt, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        sd = raw["model"]
    else:
        sd = raw
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[VGGT] loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:   print("  (first few missing):", missing[:8])
    if unexpected: print("  (first few unexpected):", unexpected[:8])
    return model.to(device).eval()

model = load_local_vggt(ckpt_path, device=device)

# ---------- images & patch grid
imgs = load_and_preprocess_images_square(scene_images, 896)[0].to(device)   # [N,3,H,W]
N, _, H, W = imgs.shape
print(f"imgs.shape: {imgs.shape}")
imgs_batched = imgs.unsqueeze(0)                             # [B=1,N,3,H,W]
patch = get_patch_size(model) or 14
gh, gw = H // patch, W // patch
assert gh > 0 and gw > 0, f"Bad grid: H,W={H,W}, patch={patch}"
print(f"gh: {gh}")
print(f"gw: {gw}")

# ---------- run AA encoder
with torch.no_grad():
    agg_tokens_list, ps_idx = model.aggregator(imgs_batched)

# tokens is concat of FA||GA at the last AA depth
tokens = agg_tokens_list[-1]                       # [B, N, Tpf, 2C]
B, Nf, Tpf, C2x = tokens.shape
print(f"tokens.shape: {tokens.shape}")
print(f"B: {B}")
print(f"Nf: {Nf}")
print(f"Tpf: {Tpf}")
print(f"C2x: {C2x}")
assert Nf == N
C = C2x // 2                                       # base channel dim

# Use the patch_start_idx returned by aggregator (you called it ps_idx earlier)
# patches are from [patch_start_idx : patch_start_idx + P]
P = gh * gw
patch_start = ps_idx                                # int, typically 1 + num_register_tokens (1+4=5)
patch_tok = tokens[:, :, patch_start:patch_start+P, :]   # [B, N, P, 2C]

# split FA / GA along channels
fa_last = patch_tok[..., :C].contiguous()          # [B, N, P, C]
ga_last = patch_tok[...,  C:].contiguous()         # [B, N, P, C]

# reshape to grids: [B,N,C,gh,gw]
fa_patch = fa_last.permute(0,1,3,2).reshape(B, N, C, gh, gw)
ga_patch = ga_last.permute(0,1,3,2).reshape(B, N, C, gh, gw)
print(f"fa_patch.shape: {fa_patch.shape}")
print(f"ga_patch.shape: {ga_patch.shape}")
# ----- visualize (example: L2 + per-frame norm for FA; scene-global norm for GA)
m_fa = torch.linalg.vector_norm(fa_patch, dim=2)             # [B,N,gh,gw]
mn = m_fa.flatten(2).min(dim=2, keepdim=True).values.view(B,N,1,1)
mx = m_fa.flatten(2).max(dim=2, keepdim=True).values.view(B,N,1,1)
m01_frame = (m_fa - mn) / (mx - mn + 1e-6)                   # per-frame

m_ga = torch.linalg.vector_norm(ga_patch, dim=2)             # [B,N,gh,gw]
m01_global = global_minmax_norm_bnhw(m_ga, across_batch=False)  # scene-global

m_up_frame = F.interpolate(m01_frame.reshape(B*N,1,gh,gw), size=(H,W),
                           mode="bilinear", align_corners=False).reshape(B,N,H,W)
m_up_global = F.interpolate(m01_global.reshape(B*N,1,gh,gw), size=(H,W),
                            mode="bilinear", align_corners=False).reshape(B,N,H,W)

for i in range(N):
    save_color_heatmap(m_up_frame[0, i], os.path.join(out_dir, f"frame{i+1:02d}_FA_norm.png"))
    save_color_heatmap(m_up_global[0, i], os.path.join(out_dir, f"frame{i+1:02d}_GA_norm.png"))
