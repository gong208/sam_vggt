import os, random, math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from segment_anything.modeling import MaskDecoder, TwoWayTransformer
# ---- helpers: losses ----
def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    # logits [B,H,W] or [1,H,W]; target [B,1,H,W] or [1,1,H,W] or [B,H,W]
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1,2,3))
    union = (probs + target).sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return (1 - dice).mean()

@torch.no_grad()
def tile_iou_from_logits(logits_tile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits_tile: [1, C, H, W]; target: [1, 1, H, W] or [1,H,W]
    if target.ndim == 3:
        target = target.unsqueeze(1)
    probs = torch.sigmoid(logits_tile)          # [1,C,H,W]
    preds = (probs > 0.5).float()
    inter = (preds * target).sum(dim=(0,2,3))   # [C]
    union = (preds + target - preds*target).sum(dim=(0,2,3)) + 1e-6
    return inter / union                        # [C]


def load_instance_mask(img_arr, target_instance_id=None, target_size=(256, 256)):
    if target_instance_id is not None:
        mask = (img_arr == target_instance_id).astype(np.float32)
    else:
        # or collapse to any non-zero foreground
        mask = (img_arr > 0).astype(np.float32)
    mask = torch.from_numpy(mask)[None, None, ...]   # [1,1,H,W]
    mask = F.interpolate(mask, size=target_size, mode="nearest")
    return mask

# ---- dataset for Hypersim-like directory with color/ & instance/ ----
class HypersimSceneDataset(Dataset):
    def __init__(self, root: str, scenes: List[str], N: int = 3, min_area: int = 256):
        self.N = N
        self.min_area = min_area
        self.items = []  # list of (image_paths[N], instance_paths[N])
        for scene in scenes:
            cdir = os.path.join(root, scene, "color")
            idir = os.path.join(root, scene, "instance")
            frames = sorted([f for f in os.listdir(cdir) if f.lower().endswith((".png",".jpg",".jpeg"))])
            # require exact pairing
            frames = [f for f in frames if os.path.exists(os.path.join(idir, f))]
            # sample sliding windows of length N (you can change sampling strategy)
            for i in range(0, len(frames) - N + 1):
                group = frames[i:i+N]
                self.items.append((
                    [os.path.join(cdir, g) for g in group],
                    [os.path.join(idir, g) for g in group],
                ))

    def __len__(self): return len(self.items)

    def _load_rgb(self, p: str):
        im = Image.open(p).convert("RGB")
        arr = np.array(im)  # H,W,3 uint8
        return arr, (arr.shape[0], arr.shape[1])  # (H,W)

    def _load_instance(self, p: str):
        # Keep integers (16/32-bit PNGs)
        arr = np.array(Image.open(p))  # H,W integer ids
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_paths, inst_paths = self.items[idx]
        N = self.N
        # choose prompt frame 0 (as requested)
        k = 0

        images_rgb, original_sizes = [], []
        for p in image_paths:
            rgb, hw = self._load_rgb(p)
            images_rgb.append(rgb)             # H,W,3
            original_sizes.append(hw)          # (H,W)

        inst_maps = [self._load_instance(p) for p in inst_paths]  # N x [H,W]

        # choose foreground instance ID from frame 0 (largest area >= min_area)
        inst0 = inst_maps[k]
        ids, counts = np.unique(inst0, return_counts=True)
        ids = ids[ids != 0]  # drop background id=0; adjust if dataset differs
        if len(ids) == 0:
            # fall back: skip sample or synthesize empty (here: raise to let sampler resample)
            raise RuntimeError("No instance in frame 0.")
        areas = {int(u): int((inst0 == u).sum()) for u in ids}
        valid = [u for u,a in areas.items() if a >= self.min_area] or list(areas.keys())
        inst_id = int(sorted(valid, key=lambda u: areas[u], reverse=True)[0])

        # build per-frame GT masks for this id (may be empty in some frames)
        gt_masks = []
        for t in range(N):
            m = (inst_maps[t] == inst_id).astype(np.float32)[None, ...]  # [1,H,W]
            gt_masks.append(torch.from_numpy(m))                          # keep CPU; move later

        # prompts: ONLY points on frame 0
        m0 = gt_masks[k][0].numpy()  # H,W in {0,1}
        pos = np.argwhere(m0 == 1)
        neg = np.argwhere(m0 == 0)
        pc_list, pl_list = [], []
        if len(pos):
            yx = pos[np.random.randint(len(pos))]
            pc_list.append([float(yx[1]), float(yx[0])])  # x,y
            pl_list.append(1)
        if len(neg):
            yx = neg[np.random.randint(len(neg))]
            pc_list.append([float(yx[1]), float(yx[0])])
            pl_list.append(0)
        point_coords = torch.tensor(pc_list, dtype=torch.float32) if pc_list else None
        point_labels = torch.tensor(pl_list, dtype=torch.int64) if pl_list else None

        return dict(
            image_paths=image_paths,             # [N] str
            original_sizes=original_sizes,       # [N] (H,W)
            gt_masks=gt_masks,                   # [N] [1,H,W] float tensor (cpu)
            prompt_frame_idx=k,                  # int (0)
            point_coords=point_coords,           # [Np,2] or None (cpu)
            point_labels=point_labels,           # [Np]   or None (cpu)
            # boxes/mask_input not used in this setting
        )

def collate_scenes(batch: List[Dict[str,Any]]) -> Dict[str,Any]:
    return {
        "batch_image_paths":   [b["image_paths"] for b in batch],         # [B][N]
        "batch_original_sizes":[b["original_sizes"] for b in batch],      # [B][N](H,W)
        "batch_gt_masks":      [b["gt_masks"] for b in batch],            # [B][N][1,H,W]
        "prompt_frame_idx":    torch.tensor([b["prompt_frame_idx"] for b in batch], dtype=torch.long),
        "point_coords_list":   [b["point_coords"] for b in batch],        # [B](None or [Np,2])
        "point_labels_list":   [b["point_labels"] for b in batch],        # [B](None or [Np])
    }

# ---- training loop ----
def train_sam_vggt(
    model,                                          # your SamVGGT instance
    train_loader: DataLoader,
    device: str = "cuda",
    epochs: int = 20,
    lr: float = 1e-4,
    lambda_iou: float = 0.5,
    multimask_output: bool = True,
    amp: bool = True,
):

    model = model.to(device)

    # Freeze encoders and prompt encoder, train fusion MLPs + decoder
    for p in model.sam.image_encoder.parameters(): p.requires_grad = False
    for p in model.vggt.parameters():              p.requires_grad = False
    for p in model.sam.prompt_encoder.parameters():p.requires_grad = False

    # Unfreeze the two MLPs and SAM decoder
    for p in model.embedding_fusion_mlp.parameters(): p.requires_grad = True
    for p in model.prompt_fusion_mlp.parameters():    p.requires_grad = True
    model.sam.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    for p in model.sam.mask_decoder.parameters():     p.requires_grad = True

    # (Optional) sanity assert
    def _trainable(name, mod):
        return any(p.requires_grad for p in mod.parameters())
    assert _trainable("fusion", model.embedding_fusion_mlp)
    assert _trainable("pfuse", model.prompt_fusion_mlp)
    assert _trainable("decoder", model.sam.mask_decoder)
    assert not any(p.requires_grad for p in model.sam.image_encoder.parameters())
    assert not any(p.requires_grad for p in model.vggt.parameters())
    assert not any(p.requires_grad for p in model.sam.prompt_encoder.parameters())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    model.train()
    for ep in range(epochs):
        running_loss, running_seg, running_iou = 0.0, 0.0, 0.0
        for it, batch in enumerate(train_loader):
            # unpack lists
            B = len(batch["batch_image_paths"])
            image_lists: List[List[str]] = batch["batch_image_paths"]
            orig_sizes_lists: List[List[Tuple[int,int]]] = batch["batch_original_sizes"]
            gt_masks_lists: List[List[torch.Tensor]] = batch["batch_gt_masks"]  # cpu
            k_list = batch["prompt_frame_idx"].tolist()
            pc_list = batch["point_coords_list"]
            pl_list = batch["point_labels_list"]

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp):
                # Forward: model should encode in batch internally and decode per-sample
                # Expect per-sample outputs: list of dicts with "masks" [1,C,H,W*N], "iou_predictions" [1,C]
                outs = model.forward_batched(
                    batch_image_paths=image_lists,
                    batch_original_sizes=orig_sizes_lists,
                    point_coords_list=pc_list,
                    point_labels_list=pl_list,
                    boxes_list=[None]*B,
                    mask_input_list=[None]*B,
                    prompt_frame_idx_list=k_list,
                    multimask_output=multimask_output,
                    return_embeddings=False,
                )

                # Loss over all frames (tiles), per sample
                seg_losses, iou_losses = [], []
                for b in range(B):
                    out_b = outs[b]

                    logits_wide = out_b["low_res_logits"]  # [1,C,256,256*N]
                    # For supervision: iterate all frames t
                    N = len(gt_masks_lists[b])
                    Hb, Wb = gt_masks_lists[b][0].shape[-2:]
                    seg_loss_b = 0.0
                    iou_loss_b = 0.0
                    valid_tiles = 0

                    for t in range(N):
                        gt_t = gt_masks_lists[b][t].to(logits_wide.device)  # [1,H,W]
                        if gt_t.sum() < 1:   # skip empty target tile to avoid degenerate IoU
                            continue
                        # slice tile t from wide logits (low-res). We train at low-res for speed:
                        # low_res logits: [1,C,256,256*N] -> ratemap to tiles
                        _, Cc, Hr, WrN = logits_wide.shape
                        tile_w_r = WrN // N
                        logits_tile_r = logits_wide[:, :, :, t*tile_w_r:(t+1)*tile_w_r]  # [1,C,Hr,Wr]
                        # Upscale tile to GT size to compute loss at original resolution
                        logits_tile = F.interpolate(logits_tile_r, size=(Hb, Wb), mode="bilinear", align_corners=False)  # [1,C,Hb,Wb]

                        # choose best channel if multimask
                        if Cc > 1:
                            with torch.no_grad():
                                ious_per_c = tile_iou_from_logits(logits_tile, gt_t)   # [C]
                            best_c = int(torch.argmax(ious_per_c))
                            chosen = logits_tile[:, best_c:best_c+1]  # [1,1,Hb,Wb]
                        else:
                            chosen = logits_tile[:, 0:1]             # [1,1,Hb,Wb]

                        bce = F.binary_cross_entropy_with_logits(chosen, gt_t)
                        dice = dice_loss_from_logits(chosen, gt_t)
                        seg_loss_b = seg_loss_b + (bce + dice)
                        valid_tiles += 1

                        # IoU head regression loss if available
                        if "iou_predictions" in out_b:
                            iou_pred = out_b["iou_predictions"]  # [1,C]
                            with torch.no_grad():
                                # recompute IoUs at low-res tile (to match iou head’s granularity)
                                ious_target = tile_iou_from_logits(logits_tile, gt_t).unsqueeze(0)  # [1,C]
                            iou_loss_b = iou_loss_b + F.mse_loss(iou_pred, ious_target)

                    if valid_tiles == 0:
                        continue
                    seg_losses.append(seg_loss_b / valid_tiles)
                    if "iou_predictions" in out_b:
                        iou_losses.append(iou_loss_b / valid_tiles)

                loss_seg = torch.stack(seg_losses).mean() if seg_losses else torch.tensor(0., device=device)
                loss_iou = torch.stack(iou_losses).mean() if iou_losses else torch.tensor(0., device=device)
                loss = loss_seg + lambda_iou * loss_iou

            scaler.scale(loss).step(optim)
            scaler.update()

            running_loss += loss.item()
            running_seg  += loss_seg.item()
            running_iou  += loss_iou.item() if iou_losses else 0.0

            if (it + 1) % 50 == 0:
                n = it + 1
                print(f"[ep {ep:02d} it {it+1:04d}] "
                      f"loss={running_loss/n:.4f} seg={running_seg/n:.4f} iouH={running_iou/n:.4f}")

        print(f"==> epoch {ep} done.")

# ----------------- usage sketch -----------------
# model = build_sam_vggt(...);   # your SamVGGT with forward_batched(...)
# train_set = HypersimSceneDataset(root="/path/to/hypersim", scenes=train_scenes, N=3)
# train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_scenes, drop_last=True)
# train_sam_vggt(model, train_loader, epochs=12, lr=1e-4, multimask_output=True)
