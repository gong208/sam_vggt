# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8


    if training:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):   
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size  # [H, W]

    def __call__(self, sample):
        image = sample["image"]             # float32 OK
        label = sample["label"]             # int32 or int64 FAILS in interpolate

        # --- Resize image (float32, bilinear) ---
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # --- Resize mask (convert to float → resize → cast back to int) ---
        label_float = label.float().unsqueeze(0)   # [1, 1, H, W]

        label_resized = F.interpolate(
            label_float,
            size=self.size,
            mode="nearest"
        ).squeeze(0)

        # convert back to integer instance IDs
        label = label_resized.to(torch.int64)

        sample["image"] = image
        sample["label"] = label
        sample["shape"] = torch.tensor(self.size)

        return sample



class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='nearest'),dim=0)
        
        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}






class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)

        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "shape": torch.tensor(im.shape[-2:]),
        }
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
class MultiSceneImageDataset(Dataset):
    """
    Dataset that returns ONE image + mask + per-image valid instance IDs.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.items = []   # list of dicts
        self.scenes = {}  # scene_id -> list of indices in self.items

        scene_dirs = sorted(os.listdir(root_dir))
        scene_idx = 0

        for scene_name in scene_dirs:
            scene_path = os.path.join(root_dir, scene_name)
            if not os.path.isdir(scene_path):
                continue

            color_dir = os.path.join(scene_path, "color")
            inst_dir  = os.path.join(scene_path, "instance")
            valid_dir = os.path.join(scene_path, "valid_ids")   # NEW

            color_files = sorted(os.listdir(color_dir))
            inst_files  = sorted(os.listdir(inst_dir))
            valid_files = sorted(os.listdir(valid_dir))

            assert len(color_files) == len(inst_files) == len(valid_files), \
                f"Scene {scene_name} mismatch between color, instance, valid_ids."

            self.scenes[scene_idx] = []

            for i, (im_name, gt_name, valid_name) in enumerate(zip(color_files, inst_files, valid_files)):
                im_path    = os.path.join(color_dir, im_name)
                gt_path    = os.path.join(inst_dir, gt_name)
                valid_path = os.path.join(valid_dir, valid_name)

                # load precomputed valid instance IDs
                with open(valid_path, "r") as f:
                    valid_ids = json.load(f)["ids"]

                self.items.append({
                    "image_path": im_path,
                    "label_path": gt_path,
                    "valid_ids": torch.tensor(valid_ids, dtype=torch.int64),   # NEW
                    "scene_id": scene_idx,
                })

                self.scenes[scene_idx].append(len(self.items) - 1)

            scene_idx += 1

        print(f"[Dataset] Loaded {len(self.items)} images across {len(self.scenes)} scenes.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        entry = self.items[index]

        # load image
        im = io.imread(entry["image_path"])
        if im.ndim == 2:
            im = np.repeat(im[:, :, None], 3, axis=2)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)

        # load instance mask
        mask = io.imread(entry["label_path"])
        if mask.ndim > 2:
            mask = mask[:, :, 0]

        im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.int32).unsqueeze(0)

        sample = {
            "image": im,
            "label": mask,
            "scene_id": entry["scene_id"],
            "index": index,
            "valid_ids": entry["valid_ids"],  # NEW
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


from torch.utils.data import Sampler

class SceneBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_frames=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size      # number of groups per batch
        self.num_frames = num_frames      # views per group
        self.shuffle = shuffle

        # Precompute all groups
        self.groups = []
        for scene_id, indices in dataset.scenes.items():
            if len(indices) < num_frames:
                continue

            idxs = list(indices)
            if shuffle:
                random.shuffle(idxs)

            # break scene into groups of num_frames
            for i in range(0, len(idxs) - num_frames + 1, num_frames):
                self.groups.append(idxs[i:i + num_frames])

        # print(f"[Sampler] Total groups = {len(self.groups)}")

    def __len__(self):
        return len(self.groups) // self.batch_size

    def __iter__(self):
        groups = self.groups.copy()
        if self.shuffle:
            random.shuffle(groups)

        # now produce batches of groups
        for i in range(0, len(groups), self.batch_size):
            batch_groups = groups[i:i + self.batch_size]

            if len(batch_groups) < self.batch_size:
                break  # drop last incomplete batch

            # flatten indices
            flat = [idx for g in batch_groups for idx in g]

            yield flat  # DataLoader sees the whole batch



def multiview_collate(batch, num_frames=8):
    """
    batch: list of samples = batch_size * num_frames
    Each sample contains:
        image: [3,H,W]
        label: [1,H,W]
        valid_ids: [K_i]
    """
    # stack images and labels
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    scene_ids = [b["scene_id"] for b in batch]

    # infer batch size
    B = len(batch) // num_frames

    # reshape to [B, N, ...]
    images = images.view(B, num_frames, *images.shape[1:])
    labels = labels.view(B, num_frames, *labels.shape[1:])

    # group-level scene id
    scene_ids = [scene_ids[i*num_frames] for i in range(B)]

    # ----------------------------
    # NEW: compute group-level valid ids
    # ----------------------------
    group_valid_ids = []

    for b in range(B):
        start = b * num_frames
        end   = (b + 1) * num_frames

        # collect valid_ids for this group
        ids_group = []
        for i in range(start, end):
            ids_group.append(batch[i]["valid_ids"])  # [K_i]

        # union across images (concatenate and unique)
        ids_union = torch.unique(torch.cat(ids_group, dim=0))
        group_valid_ids.append(ids_union)

    return {
        "images": images,
        "labels": labels,
        "scene_id": scene_ids,
        "valid_ids": group_valid_ids,   # list of length B, each entry is a tensor
    }

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

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=multiview_collate,
        num_workers=4,
    )
    return loader




