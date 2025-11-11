import torch
import os
import numpy as np
import random
from PIL import Image
import imageio

class HypersimMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, root, scenes, N=3, min_inst_area=256, choose='largest'):
        self.items = []  # list of dicts per sample
        for scene in scenes:
            color_dir = f"{root}/{scene}/color"
            inst_dir  = f"{root}/{scene}/instance"
            # enumerate and sort frames; ensure pairing exists
            frames = sorted([f for f in os.listdir(color_dir) if f.endswith('.png') or f.endswith('.jpg')])
            # choose groups of N (sliding or sampling)
            for idxs in self._sample_groups(frames, N):
                paths = [f"{color_dir}/{frames[i]}" for i in idxs]
                inst_paths = [f"{inst_dir}/{frames[i]}" for i in idxs]
                self.items.append(dict(
                    image_paths=paths,
                    instance_paths=inst_paths,
                ))
        self.N = N
        self.min_inst_area = min_inst_area
        self.choose = choose

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        rec = self.items[i]
        # load one prompt frame k (e.g., middle)
        k = self.N // 2
        # load color shapes & original sizes
        image_paths = rec['image_paths']
        instance_paths = rec['instance_paths']
        original_sizes = []
        images_rgb = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            w, h = img.size
            original_sizes.append((h, w))
            images_rgb.append(np.array(img))  # keep as numpy uint8; model will to-tensor later

        # load instance map for frame k (keep integer IDs)
        inst_img = imageio.imread(instance_paths[k])  # 16/32-bit ints
        ids, counts = np.unique(inst_img, return_counts=True)
        # remove background id (assume 0; adjust if needed)
        valid = ids[ids != 0]
        if len(valid) == 0:
            # fallback: blank (skip in sampler or resample)
            raise RuntimeError("No instances in this frame.")
        # choose instance
        if self.choose == 'largest':
            areas = {int(u): int((inst_img == u).sum()) for u in valid}
            inst_id = max(areas, key=areas.get)
        else:  # random with area threshold
            valid_ids = [int(u) for u in valid if (inst_img == u).sum() >= self.min_inst_area]
            inst_id = random.choice(valid_ids) if valid_ids else int(valid[0])
        gt_mask = (inst_img == inst_id).astype(np.uint8)[None, ...]  # [1,H,W]

        # synthesize prompts (example: points)
        pos = np.argwhere(gt_mask[0] == 1)
        neg = np.argwhere(gt_mask[0] == 0)
        # sample some points
        Np_pos, Np_neg = 1, 1
        pc_list, pl_list = [], []
        if len(pos) > 0:
            yx = pos[np.random.randint(len(pos))]
            pc_list.append([yx[1], yx[0]])  # x,y
            pl_list.append(1)
        if len(neg) > 0:
            yx = neg[np.random.randint(len(neg))]
            pc_list.append([yx[1], yx[0]])
            pl_list.append(0)
        point_coords = torch.tensor(pc_list, dtype=torch.float32) if pc_list else None
        point_labels = torch.tensor(pl_list, dtype=torch.int64) if pl_list else None


        return dict(
            image_paths=image_paths,                 # [N] paths
            original_sizes=original_sizes,           # [(H,W)] * N
            prompt_frame_idx=k,
            point_coords=point_coords,               # [Np,2] or None
            point_labels=point_labels,               # [Np]   or None
            mask_input=None,                         # optional noisy prior
            gt_mask=torch.from_numpy(gt_mask).float()# [1,H,W]
        )

def collate_fn(batch):
    return {
        "batch_image_paths":   [b["image_paths"] for b in batch],
        "batch_original_sizes":[b["original_sizes"] for b in batch],
        "prompt_frame_idx":    torch.tensor([b["prompt_frame_idx"] for b in batch], dtype=torch.long),
        "point_coords_list":   [b["point_coords"] for b in batch],
        "point_labels_list":   [b["point_labels"] for b in batch],
        "gt_mask_list":        [b["gt_mask"] for b in batch],
    }
