import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from contextlib import nullcontext

from segment_anything_hq import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# ---------- helpers
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_color_heatmap(norm_01: torch.Tensor, out_path: str, cmap="magma", gamma=1.0):
    """
    norm_01: [H,W] torch tensor in [0,1]
    """
    h = norm_01.detach().float().cpu().clamp(0, 1)
    if gamma != 1.0:
        h = h.pow(gamma)
    rgb = cm.get_cmap(cmap)(h.numpy())[...,:3]  # drop alpha, 0..1
    Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").save(out_path)

def norm_map_from_bchw(feat_bchw: torch.Tensor) -> torch.Tensor:
    """
    feat_bchw: [B,C,H,W] -> returns [B,H,W] L2 over channels, normalized per-image to [0,1]
    """
    m = torch.linalg.vector_norm(feat_bchw, dim=1)      # [B,H,W]
    # per-image min/max normalize
    m_flat = m.flatten(1)
    mn = m_flat.min(dim=1).values[:, None, None]
    mx = m_flat.max(dim=1).values[:, None, None]
    m01 = (m - mn) / (mx - mn + 1e-6)
    return m01

# ---------- setup
device = "cuda" if torch.cuda.is_available() else "cpu"
autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()

model_type = "vit_l"
sam = sam_model_registry[model_type](checkpoint="sam-hq/checkpoints/sam_hq_vit_l.pth").to(device)
sam.eval()

img_dir = "sam-hq/demo/input_imgs/"
save_dir = "output"
img_list = ["example0", "example1", "example2"]
ensure_dir(save_dir)

# ---------- preprocess all -> batch
transform = ResizeLongestSide(sam.image_encoder.img_size)  # 1024

x_1024_list = []
denorm_needed = []   # keep track for saving images
for name in img_list:
    img = Image.open(os.path.join(img_dir, f"{name}.png")).convert("RGB")
    im = np.array(img)                                      # [H,W,3] uint8

    im_resized = transform.apply_image(im)                  # [H',W',3] uint8

    # 0..255 float tensor, then SAM preprocess (pads + normalizes)
    x = torch.as_tensor(im_resized, dtype=torch.float32).permute(2,0,1)[None].to(device)  # [1,3,H',W']
    x_1024 = sam.preprocess(x)                                                 # [1,3,1024,1024]
    x_1024_list.append(x_1024)
    denorm_needed.append(name)

# stack to batch
x_1024_batch = torch.cat(x_1024_list, dim=0)  # [B,3,1024,1024]
B = x_1024_batch.size(0)


# ---------- forward once (batched)
with torch.no_grad(), autocast_ctx:
    feats, interms = sam.image_encoder(x_1024_batch)   # feats: [B,256,64,64]; interms: list of [B,H,W,C]
print(len(feats), len(interms))
print(feats.shape, interms[0].shape)
# (A) post-neck feature norms (batched)
norm_post = norm_map_from_bchw(feats)                  # [B,64,64]
norm_post_up = F.interpolate(norm_post[:,None], size=(1024,1024),
                             mode="bilinear", align_corners=False)[:,0]  # [B,1024,1024]
for b, name in enumerate(img_list):
    save_color_heatmap(norm_post_up[b], os.path.join(save_dir, f"{name}_postneck_norm_color.png"),
                       cmap="magma", gamma=0.9)

# (B) intermediate embeddings (loop layers, compute batched norms, then save per image)
if len(interms) == 0:
    print("[info] no intermediate embeddings found.")
else:
    for i, t_bhwc in enumerate(interms):
        # [B,H,W,C] -> [B,C,H,W]
        t_bchw = t_bhwc.permute(0,3,1,2).contiguous()
        norm_i = norm_map_from_bchw(t_bchw)  # [B,H,W], per-image normalized
        norm_i_up = F.interpolate(norm_i[:,None], size=(1024,1024),
                                  mode="bilinear", align_corners=False)[:,0]  # [B,1024,1024]
        for b, name in enumerate(img_list):
            out_path = os.path.join(save_dir, f"{name}_interm{i:02d}_norm_color.png")
            save_color_heatmap(norm_i_up[b], out_path, cmap="magma", gamma=0.9)

print(f"done. saved outputs to: {os.path.abspath(save_dir)}")
