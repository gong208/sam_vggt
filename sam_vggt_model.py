"""
SAM-VGGT Model: Combines SAM and VGGT encoders for multi-view segmentation.

This model integrates:
- SAM (Segment Anything Model) image encoder
- VGGT (Visual Geometry Grounded Transformer) for multi-view feature aggregation
- Fusion MLPs to combine embeddings
- SAM prompt encoder and mask decoder for segmentation
"""
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Type, Union

from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
from segment_anything.utils.transforms import ResizeLongestSide
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def get_patch_size(model) -> int:
    """Extract patch size from VGGT model."""
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
    return 14


class PerPixelMLP(nn.Module):
    """Per-pixel MLP for fusing image embeddings using 1x1 convolutions."""
    
    def __init__(self, c_in: int, c_hidden: int, c_out: int, act: Type[nn.Module] = nn.GELU):
        """
        Args:
            c_in: Input channel dimension
            c_hidden: Hidden layer dimension
            c_out: Output channel dimension
            act: Activation function (default: GELU)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1, bias=True),
            act(),
            nn.Conv2d(c_hidden, c_out, kernel_size=1, bias=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W]
        Returns:
            [B, C_out, H, W]
        """
        return self.net(x)


class MLPBlock(nn.Module):
    """MLP block for fusing sparse prompt embeddings."""
    
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        out_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embedding_dim: Input embedding dimension
            mlp_dim: Hidden layer dimension
            out_dim: Output dimension
            act: Activation function (default: GELU)
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, out_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, embedding_dim]
        Returns:
            [B, N, out_dim]
        """
        return self.lin2(self.act(self.lin1(x)))


class SamVGGT(nn.Module):
    """
    SAM-VGGT: Multi-view segmentation model combining SAM and VGGT encoders.
    
    Architecture:
        1. SAM encoder extracts single-view features [B, 256, 64, 64]
        2. VGGT aggregator extracts multi-view features [1, N, 2048, 64, 64]
        3. Per-pixel MLP fuses concatenated features -> [N, 256, 64, 64]
        4. Features are concatenated spatially -> [1, 256, 64, 64*N]
        5. Pretrained SAM prompt encoder generates sparse and dense embeddings
        6. Dense embeddings and positional embeddings are concatenated along width for multi-frame input
        7. Prompt MLP fuses SAM prompts with VGGT camera tokens
        8. SAM mask decoder produces final segmentation masks
    """
    
    mask_threshold: float = 0.0
    
    def __init__(
        self,
        sam_model_type: str = "vit_l",
        sam_checkpoint: str = "sam-hq/checkpoints/sam_hq_vit_l.pth",
        vggt_checkpoint: str = "vggt/checkpoints/model.pt",
        vggt_img_size: int = 896,
        embed_fusion_hidden: int = 768,
        prompt_fusion_hidden: int = 768,
        device: str = "cuda",
        freeze_sam_encoder: bool = True,
        freeze_vggt: bool = True,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        """
        Args:
            sam_model_type: Type of SAM model ('vit_b', 'vit_l', 'vit_h')
            sam_checkpoint: Path to SAM checkpoint
            vggt_checkpoint: Path to VGGT checkpoint
            vggt_img_size: Image size for VGGT preprocessing (default: 896)
            embed_fusion_hidden: Hidden dimension for embedding fusion MLP
            prompt_fusion_hidden: Hidden dimension for prompt fusion MLP
            device: Device to load models on
            freeze_sam_encoder: Whether to freeze SAM encoder weights
            freeze_vggt: Whether to freeze VGGT weights
            pixel_mean: Mean values for SAM image normalization
            pixel_std: Std values for SAM image normalization
        """
        super().__init__()
        
        self.device = device
        self.vggt_img_size = vggt_img_size
        
        # Load SAM model
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).to(device)
        self.sam.eval()
        
        if freeze_sam_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        # Load VGGT model
        self.vggt = self._load_vggt(vggt_checkpoint, device)
        self.vggt.eval()
        
        if freeze_vggt:
            for param in self.vggt.parameters():
                param.requires_grad = False
        
        # Get dimensions
        self.sam_encoder_dim = 256  # SAM encoder output channels
        self.vggt_encoder_dim = 2048  # VGGT output channels (2 * embed_dim)
        self.fused_dim = self.sam_encoder_dim + self.vggt_encoder_dim  # 2304
        
        # Per-pixel MLP to fuse SAM and VGGT embeddings
        self.embedding_fusion_mlp = PerPixelMLP(
            c_in=self.fused_dim,
            c_hidden=embed_fusion_hidden,
            c_out=self.sam_encoder_dim,
        ).to(device)
        
        # MLP to fuse SAM sparse prompts with VGGT camera tokens
        self.prompt_fusion_mlp = MLPBlock(
            embedding_dim=self.sam_encoder_dim + self.vggt_encoder_dim,
            mlp_dim=prompt_fusion_hidden,
            out_dim=self.sam_encoder_dim,
        ).to(device)
        
        # SAM preprocessing
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
        # Get VGGT patch size
        self.vggt_patch_size = get_patch_size(self.vggt) or 14
        
    def _load_vggt(self, ckpt: str, device: str) -> VGGT:
        """Load VGGT model from checkpoint."""
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
        if missing:
            print("  (first few missing):", missing[:8])
        if unexpected:
            print("  (first few unexpected):", unexpected[:8])
            
        return model.to(device).eval()
    
    def preprocess_sam_images(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Preprocess images for SAM encoder.
        
        Args:
            images: List of images as tensors [3, H, W] in range [0, 255]
        
        Returns:
            Batched preprocessed images [B, 3, 1024, 1024] and list of original sizes
        """
        x_1024_list = []
        original_sizes = []
        
        for img in images:
            # Apply SAM transform
            img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            img_resized = self.transform.apply_image(img_np)
            
            # Convert to tensor and preprocess
            x = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
            x_1024 = self.sam.preprocess(x)
            x_1024_list.append(x_1024)
            original_sizes.append(img.shape[-2:])
        
        return torch.cat(x_1024_list, dim=0), original_sizes

    def preprocess_sam_images_batched(
        self, batch_images: List[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]]]:
        """
        Args:
            batch_images: List of length B; each item is a list of N tensors [3,H,W] (0..255)

        Returns:
            sam_pre: [B, N, 3, 1024, 1024]
            original_sizes: List[B][N] of (H, W)
        """
        B = len(batch_images)
        assert B > 0 and len(batch_images[0]) > 0
        N = len(batch_images[0])

        x_list = []
        original_sizes: List[List[Tuple[int, int]]] = []
        for b in range(B):
            assert len(batch_images[b]) == N, "All batch items must have same N"
            sizes_b = []
            x_list_b = []
            for img in batch_images[b]:
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_resized = self.transform.apply_image(img_np)
                x = torch.as_tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)[None].to(self.device)
                x_1024 = self.sam.preprocess(x)  # [1,3,1024,1024]
                x_list_b.append(x_1024)
                sizes_b.append((img.shape[-2], img.shape[-1]))  # (H,W)
            x_list.append(torch.cat(x_list_b, dim=0))  # [N,3,1024,1024]
            original_sizes.append(sizes_b)

        sam_pre = torch.stack(x_list, dim=0)  # [B,N,3,1024,1024]
        return sam_pre, original_sizes
    
    def preprocess_vggt_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Preprocess images for VGGT encoder.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            Preprocessed images [N, 3, H, W] in range [0, 1]
        """
        imgs = load_and_preprocess_images_square(image_paths, self.vggt_img_size)[0].to(self.device)
        return imgs
    
    def preprocess_vggt_images_batched(self, batch_image_paths: List[List[str]]) -> torch.Tensor:
        """
        Args:
            batch_image_paths: List of length B; each item is a list of N file paths.

        Returns:
            [B, N, 3, H, W] in [0, 1]
        """
        B = len(batch_image_paths)
        N = len(batch_image_paths[0])
        imgs_bn = []
        for b in range(B):
            assert len(batch_image_paths[b]) == N
            imgs_n = load_and_preprocess_images_square(batch_image_paths[b], self.vggt_img_size)[0].to(self.device)  # [N,3,H,W]
            imgs_bn.append(imgs_n)
        return torch.stack(imgs_bn, dim=0)  # [B,N,3,H,W]


    def encode_sam(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode images with SAM encoder.
        
        Args:
            images: [B, 3, 1024, 1024] preprocessed images
        
        Returns:
            feats: [B, 256, 64, 64] SAM embeddings
            interms: List of intermediate embeddings
        """
        with torch.no_grad():
            feats, interms = self.sam.image_encoder(images)
        return feats, interms
    
    def encode_sam_batched(
        self, sam_pre: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            sam_pre: [B, N, 3, 1024, 1024]

        Returns:
            feats_bn: [B, N, 256, 64, 64]
            interms:  whatever the encoder returns (list of tensors) for completeness
        """
        B, N = sam_pre.shape[:2]
        with torch.no_grad():
            feats, interms = self.sam.image_encoder(sam_pre.view(B * N, 3, 1024, 1024))  # [B*N,256,64,64]
        feats_bn = feats.view(B, N, 256, 64, 64)
        return feats_bn, interms

    def encode_vggt(
        self, 
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encode images with VGGT aggregator.
        
        Args:
            images: [N, 3, H, W] preprocessed images in range [0, 1]
        
        Returns:
            patch_features: [N, 2048, 64, 64] VGGT patch embeddings
            camera_tokens: [1, N, 2048] VGGT camera tokens
            patch_start_idx: Index where patch tokens start
        """
        N, _, H, W = images.shape
        imgs_batched = images.unsqueeze(0)  # [1, N, 3, H, W]
        
        # Calculate grid dimensions
        gh, gw = H // self.vggt_patch_size, W // self.vggt_patch_size
        assert gh > 0 and gw > 0, f"Bad grid: H,W={H,W}, patch={self.vggt_patch_size}"
        
        with torch.no_grad():
            agg_tokens_list, ps_idx = self.vggt.aggregator(imgs_batched)
        
        # Extract tokens from last aggregation layer
        tokens = agg_tokens_list[-1]  # [B, N, Tpf, 2C]
        B, Nf, Tpf, C2x = tokens.shape
        assert Nf == N
        C = C2x // 2
        
        # Extract camera tokens
        camera_tokens = tokens[:, :, 0]  # [1, N, 2048]
        
        # Extract patch tokens
        P = gh * gw
        patch_start = ps_idx
        patch_tok = tokens[:, :, patch_start:patch_start+P, :]  # [1, N, P, 2C]
        
        # Reshape to spatial format
        patch_tok = patch_tok.permute(0, 1, 3, 2).contiguous().reshape(B, N, C2x, gh, gw)
        patch_tok = patch_tok.squeeze(0)  # [N, 2048, 64, 64]
        
        return patch_tok, camera_tokens, patch_start
    
    def encode_vggt_batched(
        self, imgs_bn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            imgs_bn: [B, N, 3, H, W]

        Returns:
            patch_features: [B, N, 2048, 64, 64]
            camera_tokens:  [B, N, 2048]
            patch_start_idx: int
        """
        B, N, _, H, W = imgs_bn.shape
        gh, gw = H // self.vggt_patch_size, W // self.vggt_patch_size
        assert gh > 0 and gw > 0

        with torch.no_grad():
            agg_tokens_list, ps_idx = self.vggt.aggregator(imgs_bn)  # tokens per layer: [B,N,Tpf,2C]

        tokens = agg_tokens_list[-1]     # [B,N,Tpf,2C]
        C2x = tokens.shape[-1]
        C = C2x // 2
        cam = tokens[:, :, 0]            # [B,N,2C] -> but camera token is typically 2C; many impls store it as 2C; if not, adjust
        if cam.shape[-1] != C2x:
            # if aggregator already outputs 2C for tokens and cam is size 2C, we can keep it as is;
            pass
        # For downstream fusion we used 2048 (=2C). Keep it as is.
        camera_tokens = cam  # [B,N,2C] (typically 2048)

        P = gh * gw
        patch_start = ps_idx
        patch_tok = tokens[:, :, patch_start:patch_start + P, :]  # [B,N,P,2C]
        # -> [B,N,2C,P] -> [B,N,2C,gh,gw]
        patch_tok = patch_tok.permute(0, 1, 3, 2).contiguous().view(B, N, C2x, gh, gw)
        return patch_tok, camera_tokens, patch_start

    def fuse_embeddings(
        self, 
        sam_feats: torch.Tensor, 
        vggt_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse SAM and VGGT embeddings using per-pixel MLP.
        
        Args:
            sam_feats: [N, 256, 64, 64] SAM embeddings
            vggt_feats: [N, 2048, 64, 64] VGGT embeddings
        
        Returns:
            [N, 256, 64, 64] fused embeddings
        """
        # Concatenate channel-wise
        x = torch.cat([sam_feats, vggt_feats], dim=1)  # [N, 2304, 64, 64]
        
        # Apply fusion MLP
        fused = self.embedding_fusion_mlp(x)  # [N, 256, 64, 64]
        
        return fused

    def fuse_embeddings_batched(self, sam_feats, vggt_feats):
        # sam_feats:  [B, N, 256, 64, 64]
        # vggt_feats: [B, N, 2048, 64, 64]
        x = torch.cat([sam_feats, vggt_feats], dim=2)  # [B, N, 2304, 64, 64]
        B, N, C, H, W = x.shape
        fused = self.embedding_fusion_mlp(x.view(B * N, C, H, W))  # [B*N, 256, 64, 64]
        return fused.view(B, N, 256, H, W)

    
    def concatenate_embeddings_spatially(
        self, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate embeddings from multiple frames spatially.
        
        Args:
            embeddings: [N, 256, 64, 64] per-frame embeddings
        
        Returns:
            [1, 256, 64, 64*N] concatenated embeddings
        """
        N, C, H, W = embeddings.shape
        # Reshape to concatenate along width
        concat_embed = embeddings.reshape(1, C, H, W * N)
        return concat_embed
    
    def fuse_prompts(
        self,
        sam_sparse: torch.Tensor,        # [1, 7, 256]
        vggt_cam: torch.Tensor,          # [1, Nframes, 2048]
        prompt_frame_idx: Union[List[int], torch.Tensor],  # [6], only real points
    ) -> torch.Tensor:
        """
        Fuse only the real point tokens (first len(prompt_frame_idx)) with VGGT camera
        tokens, leave the last SAM padding token unchanged.

        sam_sparse: [B, Np, 256] (here B=1, Np=7)
        vggt_cam:   [1, Nframes, 2048]
        prompt_frame_idx: [N_real] (here N_real=6), frame index per real point
        """
        B, Np, D_sam = sam_sparse.shape
        _, Nframes, D_cam = vggt_cam.shape
        device = sam_sparse.device

        # Number of real points (exclude SAM's not-a-point token)
        N_real = len(prompt_frame_idx)

        # Convert frame indices to tensor: [N_real]
        if isinstance(prompt_frame_idx, list):
            frame_idx = torch.tensor(prompt_frame_idx, device=device, dtype=torch.long)
        else:
            frame_idx = prompt_frame_idx.to(device=device, dtype=torch.long)

        # 1) Take only the real point embeddings: [B, N_real, 256]
        sam_real = sam_sparse[:, :N_real, :]     # [1, 6, 256]

        # 2) Gather the matching camera token for each real point:
        #    vggt_cam: [1, Nframes, 2048]
        #    frame_idx: [N_real], values in [0, Nframes-1]
        cam_for_points = vggt_cam[:, frame_idx, :]    # [1, 6, 2048]

        # If B > 1 later, expand over batch
        if B > 1:
            cam_for_points = cam_for_points.expand(B, -1, -1)  # [B, 6, 2048]

        # 3) Concatenate and fuse: [B, 6, 2304] -> [B, 6, 256]
        fused_in = torch.cat([sam_real, cam_for_points], dim=-1)  # [1, 6, 2304]
        fused_real = self.prompt_fusion_mlp(fused_in)             # [1, 6, 256]

        # 4) Put the fused real points back, keep last token untouched
        fused_prompts = sam_sparse.clone()           # [1, 7, 256]
        fused_prompts[:, :N_real, :] = fused_real    # first 6 fused, last 1 unchanged

        return fused_prompts   # [1, 7, 256]

    
    def concatenate_prompt_embeddings(
        self,
        dense_embeddings: torch.Tensor,
        dense_pe: torch.Tensor,
        num_frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate dense embeddings and positional embeddings for multi-frame input.
        
        Since we're only using point prompts (no mask prompts), we can simply repeat
        the dense embeddings and positional embeddings along the width dimension.
        
        Args:
            dense_embeddings: [B, 256, 64, 64] dense embeddings from SAM prompt encoder
            dense_pe: [1, 256, 64, 64] positional embeddings from SAM prompt encoder
            num_frames: Number of frames concatenated
        
        Returns:
            Tuple of:
                - concatenated_dense_embeddings: [B, 256, 64, 64*num_frames]
                - concatenated_dense_pe: [1, 256, 64, 64*num_frames]
        """
        # Concatenate dense embeddings along width
        # [B, 256, 64, 64] -> [B, 256, 64, 64*N]
        concatenated_dense_embeddings = dense_embeddings.repeat(1, 1, 1, num_frames)
        
        # Concatenate positional embeddings along width
        # [1, 256, 64, 64] -> [1, 256, 64, 64*N]
        concatenated_dense_pe = dense_pe.repeat(1, 1, 1, num_frames)
        
        return concatenated_dense_embeddings, concatenated_dense_pe
    

    def forward(
        self,
        batch_image_paths: List[List[str]],
        point_coords_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i, 2] in original coords
        point_labels_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i]
        boxes_list: Optional[List[torch.Tensor]] = None,         # len B, each [Nb_i, 4] (optional)
        mask_input_list: Optional[List[torch.Tensor]] = None,    # len B, each [1, H', W'] (optional)
        point_frame_indices_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i] - frame index for each point
        multimask_output: bool = True,
        return_embeddings: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Multi-view, multi-sample forward:
            - Encodes all images across the batch at once with VGGT and SAM.
            - Then loops per sample to run prompt encoding, fusion, decoding, and postprocess.

        Returns:
            A list (len B) of dicts with keys: masks, iou_predictions, low_res_logits, (optional embeddings...)
        """
        B = len(batch_image_paths)
        assert B > 0
        N = len(batch_image_paths[0])
        for b in range(B):
            assert len(batch_image_paths[b]) == N, "All batch items must have same N"

        # 1) Load images for SAM (to keep consistent with your pipeline)
        #    Build tensors [B,N,3,H,W] (0..255)
        sam_images_bn: List[List[torch.Tensor]] = []
        original_sizes_bn: List[List[Tuple[int, int]]] = []
        for b in range(B):
            imgs_b = []
            sizes_b = []
            for p in batch_image_paths[b]:
                img_pil = Image.open(p).convert("RGB")
                sizes_b.append((img_pil.size[1], img_pil.size[0]))  # (H,W)
                img_np = np.array(img_pil)
                imgs_b.append(torch.from_numpy(img_np).permute(2, 0, 1).float())  # [3,H,W]
            sam_images_bn.append(imgs_b)
            original_sizes_bn.append(sizes_b)

        # 2) Preprocess SAM (batched)  -> [B,N,3,1024,1024]
        sam_pre, _ = self.preprocess_sam_images_batched(sam_images_bn)

        # 3) Preprocess VGGT (batched) -> [B,N,3,Hv,Wv]
        vggt_pre = self.preprocess_vggt_images_batched(batch_image_paths)

        # 4) Encode with SAM and VGGT in batch
        sam_feats_bn, sam_interms = self.encode_sam_batched(sam_pre)                      # [B,N,256,64,64]
        vggt_feats_bn, cam_tokens_bn, _ = self.encode_vggt_batched(vggt_pre)             # [B,N,2048,64,64], [B,N,2048]

        # 5) Fuse per-frame features (batched) -> [B,N,256,64,64]
        fused_bn = self.fuse_embeddings_batched(sam_feats_bn, vggt_feats_bn)

        # Prepare constant dense PE once
        dense_pe_1 = self.sam.prompt_encoder.get_dense_pe()  # [1,256,64,64]

        # 6) Loop per sample for prompts, decode, postprocess
        outputs: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            # (i) concat this sample's frames horizontally: [1,256,64,64*N]
            fused_b = fused_bn[b]                                        # [N,256,64,64]
            concat_embed_b = fused_b.reshape(1, 256, 64, 64 * N)         # [1,256,64,64N]

            # (ii) prompts: coordinates are already in 1024x1024 (or concatenated) space, no transformation needed
            points_tuple = None
            if point_coords_list is not None and point_labels_list is not None and point_coords_list[b] is not None:
                pc_t = point_coords_list[b].detach().cpu().float()            # [Np, 2] torch - already in 1024x1024 or concatenated space
                # Skip transformation since coordinates are already scaled
                pc_1024 = pc_t.to(self.device).unsqueeze(0)                   # [1, Np, 2] torch
                pl_1024 = point_labels_list[b].to(self.device).unsqueeze(0)   # [1, Np]
                points_tuple = (pc_1024, pl_1024)


            # (iii) SAM prompt encoder (per-sample)
            sparse_e, dense_e = self.sam.prompt_encoder(
                points=points_tuple,
                boxes=None,
                masks=None,
            )  # sparse: [1,Np,256], dense: [1,256,64,64]

            # (iv) tile dense & PE along width for N frames
            dense_e_cat = dense_e.repeat(1, 1, 1, N)        # [1,256,64,64N]
            dense_pe_cat = dense_pe_1.repeat(1, 1, 1, N)    # [1,256,64,64N]

            # (v) fuse sparse prompts with camera tokens - each point fuses with its corresponding frame's camera token
            cam_b = cam_tokens_bn[b:b+1]  # [1,N,2048]
            # Get per-point frame indices
            if point_frame_indices_list is not None and point_frame_indices_list[b] is not None:
                per_point_frame_indices = point_frame_indices_list[b].tolist()  # [Np] list of frame indices

            fused_prompts_b = self.fuse_prompts(sparse_e, cam_b, prompt_frame_idx=per_point_frame_indices)  # [1,Np,256]
            print(concat_embed_b.shape)
            print(dense_pe_cat.shape)
            print(fused_prompts_b.shape)
            print(dense_e_cat.shape)
            # (vi) decode masks for this sample
            low_res_masks_b, iou_pred_b = self.sam.mask_decoder(
                image_embeddings=concat_embed_b,         # [1,256,64,64N]
                image_pe=dense_pe_cat,                   # [1,256,64,64N]
                sparse_prompt_embeddings=fused_prompts_b,# [1,Np,256]
                dense_prompt_embeddings=dense_e_cat,     # [1,256,64,64N]
                multimask_output=multimask_output,
            )
            # (vii) postprocess back to the prompt frame’s original size (wide canvas)
            H0, W0 = original_sizes_bn[b][0]
            masks_b = self.sam.postprocess_masks(
                low_res_masks_b,
                input_size=(sam_pre.shape[-2], sam_pre.shape[-1] * N),          # (1024, 1024*N)
                original_size=(H0, W0 * N),
            )
            masks_b = masks_b > self.mask_threshold

            out_b = {
                "masks": masks_b,                     # [1,C,H0,W0*N]
                "iou_predictions": iou_pred_b,        # [1,C]
                "low_res_logits": low_res_masks_b,    # [1,C,256,256*N]
            }
            if return_embeddings:
                out_b["sam_vggt_embeddings"] = concat_embed_b
                out_b["fused_prompts"] = fused_prompts_b
                out_b["concatenated_dense_embeddings"] = dense_e_cat
                out_b["concatenated_dense_pe"] = dense_pe_cat

            outputs.append(out_b)

        return outputs

def build_sam_vggt(
    sam_model_type: str = "vit_l",
    sam_checkpoint: str = "sam-hq/checkpoints/sam_hq_vit_l.pth",
    vggt_checkpoint: str = "vggt/checkpoints/model.pt",
    device: str = "cuda",
    **kwargs
) -> SamVGGT:
    """
    Build SAM-VGGT model with default or custom parameters.
    
    Args:
        sam_model_type: Type of SAM model ('vit_b', 'vit_l', 'vit_h')
        sam_checkpoint: Path to SAM checkpoint
        vggt_checkpoint: Path to VGGT checkpoint
        device: Device to load models on
        **kwargs: Additional arguments for SamVGGT
    
    Returns:
        SamVGGT model
    """
    model = SamVGGT(
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
        vggt_checkpoint=vggt_checkpoint,
        device=device,
        **kwargs
    )
    return model

