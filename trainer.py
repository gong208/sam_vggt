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
from torchvision import transforms
from sam_vggt_model import SamVGGT, build_sam_vggt
from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, LargeScaleJitter
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
                                # recompute IoUs at low-res tile (to match iou head's granularity)
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


# ---- Helper function to create non-distributed dataloader ----
def create_dataloader_non_distributed(name_im_gt_list, my_transforms=[], batch_size=1):
    """
    Create a dataloader without distributed sampling (for single-GPU training).
    Similar to create_dataloaders but without DistributedSampler.
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'sam-hq', 'train'))
    from utils.dataloader import OnlineDataset
    
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
    dataloader = DataLoader(
        gos_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers_
    )
    
    return dataloader, gos_dataset


# ---- Distributed training setup helper ----
def init_distributed_training():
    """
    Initialize distributed training. Returns rank, world_size, local_rank, and device.
    """
    import torch.distributed as dist
    
    # Check if distributed is already initialized
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Try to get from environment (set by torchrun or torch.distributed.launch)
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
            )
        else:
            # Single GPU mode
            rank = 0
            world_size = 1
            local_rank = 0
            return rank, world_size, local_rank, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


# ---- New training loop for SAM-VGGT with 4 continuous images ----
def train_sam_vggt_continuous(
    model,
    train_dataloader,
    optimizer,
    device: str = "cuda",
    num_frames: int = 4,
    amp: bool = True,
    distributed: bool = False,
):
    """
    Training loop for SAM-VGGT that:
    - Loads 4 continuous images from dataset per batch
    - Generates point prompts randomly from any of the 4 images
    - Adjusts point coordinates for concatenated layout
    - Feeds to model (no regression loss computation here)
    
    Args:
        model: SamVGGT model (should be wrapped with DDP if distributed=True)
        train_dataloader: DataLoader (should use DistributedSampler if distributed=True)
        optimizer: Optimizer
        device: Device string or torch.device
        num_frames: Number of continuous images per batch (default: 4)
        amp: Use mixed precision training
        distributed: Whether using distributed training
    """
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'sam-hq', 'train'))
    from utils.misc import masks_sample_points
    import random
    import tempfile
    
    # model.train()
    
    # Set epoch for DistributedSampler if using distributed training
    if distributed and hasattr(train_dataloader.sampler, 'set_epoch'):
        # This should be set at the beginning of each epoch
        pass  # Will be set by caller
    
    for data in train_dataloader:
        # data['image']: [B, C, 1024, 1024] - single image per sample
        # data['label']: [B, 1, 1024, 1024] - single mask per sample
        # We need to group 4 consecutive samples into one batch
        
        inputs = data['image']  # [B, C, 1024, 1024]
        labels = data['label']  # [B, 1, 1024, 1024]
        print(inputs.shape)
        print(labels.shape)
        # Get image paths if available in dataset
        im_paths = data.get('ori_im_path', None)
        if im_paths is None:
            # Fallback: need to get paths from dataset indices
            im_paths = [None] * inputs.shape[0]
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        batch_size = inputs.shape[0]
        
        # Process in groups of num_frames (4)
        for group_start in range(0, batch_size, num_frames):
            group_end = min(group_start + num_frames, batch_size)
            if group_end - group_start < num_frames:
                break  # Skip incomplete groups
            
            # Extract 4 continuous images and labels
            group_inputs = inputs[group_start:group_end]  # [4, C, 1024, 1024]
            group_labels = labels[group_start:group_end]  # [4, 1, 1024, 1024]
            
            # Convert images to numpy for saving/loading
            imgs_np = group_inputs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [4, 1024, 1024, C]
            
            # Generate point prompts randomly from any of the 4 images
            # Each frame gets exactly 2 points
            all_point_coords = []
            all_point_labels = []
            point_frame_indices = []  # Track which frame each point belongs to
            
            # Randomly select which frame(s) to sample points from
            # Sample points from 1-4 random frames, each with 2 points
            num_prompt_frames = random.randint(1, num_frames)
            prompt_frame_indices = random.sample(range(num_frames), num_prompt_frames)
            # order the prompt_frame_indices from small to large
            prompt_frame_indices.sort()
            for frame_idx in prompt_frame_indices:
                try:
                    # Sample points from this frame's mask
                    frame_labels = group_labels[frame_idx:frame_idx+1, 0, :, :]  # [1, 1024, 1024]
                    frame_points = masks_sample_points(frame_labels, k=2)  # [1, 2, 2]
                    
                    # Remove batch dimension: [1, 2, 2] -> [2, 2]
                    frame_points = frame_points.squeeze(0)  # [2, 2]
                    num_pts = frame_points.shape[0]

                    if num_pts == 0:
                        # No points sampled; skip this frame
                        continue
                    all_point_coords.append(frame_points)
                    all_point_labels.append(
                        torch.ones(num_pts, device=frame_points.device, dtype=torch.long)
                    )
                    point_frame_indices.extend([frame_idx] * num_pts)
                    
                except:
                    # Skip if not enough points
                    continue
            
            if len(all_point_coords) == 0:
                continue  # Skip if no valid points
            
            # Concatenate all points
            point_coords_concat = torch.cat(all_point_coords, dim=0)  # [N_points, 2]
            point_labels_concat = torch.cat(all_point_labels, dim=0)  # [N_points]
            point_frame_indices = torch.tensor(point_frame_indices, dtype=torch.long)  # [N_points]

            # Save images to temporary files for model to load
            # Model expects image paths, so we create temp files
            temp_dir = tempfile.mkdtemp()
            image_paths = []
            original_sizes = []
            
            for i in range(num_frames):
                # Save image to temp file
                temp_path = os.path.join(temp_dir, f"frame_{i}.png")
                img_pil = Image.fromarray(imgs_np[i])
                img_pil.save(temp_path)
                image_paths.append(temp_path)
                original_sizes.append((1024, 1024))  # After LargeScaleJitter, all are 1024x1024
            
            # Prepare point coords and labels for model
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
            
            # Clean up temp files
            shutil.rmtree(temp_dir)
            
            # TODO: Add loss computation and backward pass here
            # For now, just the data loading and model forward is done

# ----------------- usage sketch -----------------
# model = build_sam_vggt(...);   # your SamVGGT with forward_batched(...)
# train_set = HypersimSceneDataset(root="/path/to/hypersim", scenes=train_scenes, N=3)
# train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=8, collate_fn=collate_scenes, drop_last=True)
# train_sam_vggt(model, train_loader, epochs=12, lr=1e-4, multimask_output=True)



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
            LargeScaleJitter()  # Outputs 1024x1024 images
        ],
        batch_size=8  # Should be multiple of 4 (or num_frames)
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
    train_sam_vggt_continuous(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        device=device,
        num_frames=4,  # Number of continuous images per batch
        amp=True  # Use mixed precision
    )
    
    print("Training completed!")


def main_distributed():
    """
    Main function for distributed training with multiple GPUs.
    Launch with: torchrun --nproc_per_node=3 trainer.py
    Or: python -m torch.distributed.launch --nproc_per_node=3 trainer.py
    """
    # 1. Initialize distributed training
    rank, world_size, local_rank, device = init_distributed_training()
    is_main_process = (rank == 0)
    
    if is_main_process:
        print(f"Distributed training: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    # 2. Define datasets
    train_datasets = [
        {
            "name": "Dataset1",
            "im_dir": "./ai_001_001_00/color",
            "gt_dir": "./ai_001_001_00/instance",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        # Add more datasets...
    ]
    
    # 3. Create distributed dataloader (uses DistributedSampler)
    train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    train_dataloader, _ = create_dataloaders(
        train_im_gt_list,
        my_transforms=[
            RandomHFlip(),
            LargeScaleJitter()
        ],
        batch_size=8,  # Per GPU batch size (total = batch_size * world_size)
        training=True  # This will use DistributedSampler
    )
    
    # 4. Initialize model
    # model = build_sam_vggt()  # Your model initialization
    # model = model.to(device)
    model = None
    # 5. Wrap model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True  # Set to False if all parameters are used
    )
    model_without_ddp = model.module  # Access underlying model
    
    # 6. Set up optimizer (use model_without_ddp for optimizer)
    # Freeze encoders, train fusion MLPs and decoder
    for p in model_without_ddp.sam.image_encoder.parameters():
        p.requires_grad = False
    for p in model_without_ddp.vggt.parameters():
        p.requires_grad = False
    for p in model_without_ddp.sam.prompt_encoder.parameters():
        p.requires_grad = False
    
    for p in model_without_ddp.embedding_fusion_mlp.parameters():
        p.requires_grad = True
    for p in model_without_ddp.prompt_fusion_mlp.parameters():
        p.requires_grad = True
    for p in model_without_ddp.sam.mask_decoder.parameters():
        p.requires_grad = True
    
    trainable_params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0)
    
    # 7. Training loop with epochs
    num_epochs = 10
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    for epoch in range(num_epochs):
        if is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Set epoch for DistributedSampler (important for shuffling)
        if hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_sam_vggt_continuous(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            num_frames=4,
            amp=True,
            distributed=True
        )
        
        # Add validation, checkpointing, etc. here
        if is_main_process:
            print(f"Epoch {epoch+1} completed")
    
    # Cleanup
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # Check if running in distributed mode
    # if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
    #     main_distributed()
    # else:
    main()  # Single GPU mode

