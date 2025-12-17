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
from utils.loss_mask import loss_masks
from utils.misc import sample_points_for_instances
from torch.utils.data import DataLoader
from utils.dataloader import MultiSceneImageDataset, SceneBatchSampler, RandomHFlip, Resize, multiview_collate
from functools import partial

def create_dataloader(root_dir, batch_size=4, num_frames=8, my_transforms=[]):

    dataset = MultiSceneImageDataset(
        root_dir=root_dir,
        transform=transforms.Compose(my_transforms)
    )

    sampler = SceneBatchSampler(
        dataset, 
        batch_size=batch_size, 
        num_frames=num_frames
    )

    # Create a partial collate function with num_frames pre-filled
    collate_fn = partial(multiview_collate, num_frames=num_frames)

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=4,
    )
    return loader

def inspect_dataloader(dataloader, num_batches=2, num_frames=8):
    """
    Inspect the dataloader output for correctness:
    - batch size
    - number of views per group
    - shape correctness
    - scene consistency inside each group
    - instance ID range checks
    """
    print("\n================ DATALOADER INSPECTION ================")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        images = batch["images"]    # expected [B, 8, 3, H, W]
        labels = batch["labels"]    # expected [B, 8, 1, H, W]
        scene_ids = batch["scene_id"]  # expected length = B

        print(f"\n--- Batch {batch_idx} ---")
        print(f"Images shape: {tuple(images.shape)}")
        print(f"Labels shape: {tuple(labels.shape)}")

        if images.ndim != 5:
            print("❌ ERROR: images should be 5D [B, num_frames, 3, H, W]")
            return

        if labels.ndim != 5:
            print("❌ ERROR: labels should be 5D [B, num_frames, 1, H, W]")
            return

        B, NF, C, H, W = images.shape

        if NF != num_frames:
            print(f"❌ ERROR: num_frames mismatch: expected {num_frames}, got {NF}")
        else:
            print(f"✔ num_frames correct: {NF}")

        print(f"Batch size (groups): {B}")

        # ---------- Check scene consistency ----------
        print("\nScene ID consistency per group:")
        if isinstance(scene_ids, list):
            scene_ids = scene_ids  # from collate_fn
        elif isinstance(scene_ids, torch.Tensor):
            scene_ids = scene_ids.tolist()

        for i, sid in enumerate(scene_ids):
            print(f"  Group {i}: Scene ID = {sid}")

        # ---------- Instance ID range check ----------
        print("\nInstance ID ranges per group:")
        for i in range(B):
            group_labels = labels[i]  # [8, 1, H, W]
            instance_ids = group_labels.unique()
            min_id = int(instance_ids.min())
            max_id = int(instance_ids.max())
            print(f"  Group {i}: instance_id range = [{min_id}, {max_id}]")

        # ---------- Quick pixel inspection ----------
        print("\nRandom pixel check:")
        rand_img = images[0, 0]  # first sample, first view
        rand_label = labels[0, 0]

        print(f"  Sample image min/max: {float(rand_img.min())}/{float(rand_img.max())}")
        print(f"  Sample mask unique values (first few): {rand_label.unique()[:10]}")

    print("\n========================================================\n")



if __name__ == "__main__":

    train_loader = create_dataloader(
        root_dir="./hypersim",
        batch_size=4,
        num_frames=8,
        my_transforms=[
            Resize(size=[1024,1024]),
        ]
    )

    inspect_dataloader(train_loader, num_batches=2, num_frames=8)
