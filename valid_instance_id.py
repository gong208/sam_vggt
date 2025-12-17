import os
import json
import torch
from skimage import io
from tqdm import tqdm
import os
import json
import torch
from skimage import io
from tqdm import tqdm

def compute_valid_ids_per_image(scene_path, area_ratio_threshold=0.05):
    """
    Only keep instance IDs whose area >= threshold * image_area.
    """
    inst_dir = os.path.join(scene_path, "instance")
    files = sorted(os.listdir(inst_dir))

    out_dir = os.path.join(scene_path, "valid_ids")
    os.makedirs(out_dir, exist_ok=True)

    for fname in tqdm(files, desc=f"Scene {scene_path}"):
        mask_path = os.path.join(inst_dir, fname)
        mask = io.imread(mask_path)

        if mask.ndim > 2:
            mask = mask[:, :, 0]

        H, W = mask.shape
        total_pixels = H * W
        min_pixels = int(area_ratio_threshold * total_pixels)

        mask_t = torch.tensor(mask, dtype=torch.int64)

        ids = torch.unique(mask_t)
        ids = ids[ids != 0]  # remove background

        valid_ids = []

        for idv in ids.tolist():
            area = torch.sum(mask_t == idv).item()
            if area >= min_pixels:
                valid_ids.append(idv)

        # save per-image valid instance ids
        save_path = os.path.join(out_dir, fname.replace(".png", ".json"))
        with open(save_path, "w") as f:
            json.dump({"ids": valid_ids}, f)


def preprocess_dataset(root_dir):
    scenes = sorted(os.listdir(root_dir))
    for scene in scenes:
        scene_path = os.path.join(root_dir, scene)
        if os.path.isdir(scene_path):
            compute_valid_ids_per_image(scene_path)

# Run once
preprocess_dataset("hypersim")
