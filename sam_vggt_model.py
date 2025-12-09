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
        self.vggt_camera_dim = 9  # VGGT camera token dimension
        self.fused_dim = self.sam_encoder_dim + self.vggt_encoder_dim  # 2304
        
        # Per-pixel MLP to fuse SAM and VGGT embeddings
        self.embedding_fusion_mlp = PerPixelMLP(
            c_in=self.fused_dim,
            c_hidden=embed_fusion_hidden,
            c_out=self.sam_encoder_dim,
        ).to(device)
        
        # MLP to fuse SAM sparse prompts with VGGT camera tokens
        self.prompt_fusion_mlp = MLPBlock(
            embedding_dim=self.sam_encoder_dim + self.vggt_camera_dim,
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

        pose_enc_list = self.vggt.camera_head(agg_tokens_list)
        cam_pose_enc = pose_enc_list[-1] # [B, N, 9]    
        # print("cam_pose_enc shape: ", cam_pose_enc.shape)
        tokens = agg_tokens_list[-1]     # [B,N,Tpf,2C]
        C2x = tokens.shape[-1]
 
        P = gh * gw
        patch_start = ps_idx
        patch_tok = tokens[:, :, patch_start:patch_start + P, :]  # [B,N,P,2C]
        # -> [B,N,2C,P] -> [B,N,2C,gh,gw]
        patch_tok = patch_tok.permute(0, 1, 3, 2).contiguous().view(B, N, C2x, gh, gw)
        return patch_tok, cam_pose_enc, patch_start


    def fuse_embeddings_batched(self, sam_feats, vggt_feats):
        # sam_feats:  [B, N, 256, 64, 64]
        # vggt_feats: [B, N, 2048, 64, 64]
        x = torch.cat([sam_feats, vggt_feats], dim=2)  # [B, N, 2304, 64, 64]
        B, N, C, H, W = x.shape
        fused = self.embedding_fusion_mlp(x.view(B * N, C, H, W))  # [B*N, 256, 64, 64]
        return fused.view(B, N, 256, H, W)

    
    def fuse_prompts(
        self,
        sam_sparse: torch.Tensor,          # [B, Np, 256]
        vggt_cam: torch.Tensor,            # [B, Nframes, 9]
        prompt_frame_idx: torch.Tensor,    # [B, N_real]
    ) -> torch.Tensor:
        """
        Batchified fusion: each batch element has N_real real points (fixed for all B).
        Fuses SAM point-token embeddings with per-frame camera tokens.
        """
        B, Np, D_sam = sam_sparse.shape
        _, Nframes, D_cam = vggt_cam.shape
        device = sam_sparse.device

        # Ensure long dtype for indexing
        frame_idx = prompt_frame_idx.to(device=device, dtype=torch.long)  # [B, N_real]
        N_real = frame_idx.shape[1]

        # ----------------------------------------------------------------------
        # 1) Extract the real SAM point embeddings:  [B, N_real, 256]
        # ----------------------------------------------------------------------
        sam_real = sam_sparse[:, :N_real, :]    # [B, N_real, 256]

        # ----------------------------------------------------------------------
        # 2) Gather camera tokens according to frame_idx
        #    vggt_cam:   [B, Nframes, 9]
        #    frame_idx:  [B, N_real]
        #
        # To gather with dim=1 we expand: [B, N_real] → [B, N_real, 9]
        # ----------------------------------------------------------------------
        cam_for_points = torch.gather(
            vggt_cam,
            dim=1,
            index=frame_idx.unsqueeze(-1).expand(B, N_real, D_cam)   # [B, N_real, 9]
        )

        # ----------------------------------------------------------------------
        # 3) Concatenate SAM real embeddings + camera embeddings
        #    fused_in: [B, N_real, 256+9 = 265]
        # ----------------------------------------------------------------------
        fused_in = torch.cat([sam_real, cam_for_points], dim=-1)   # [B, N_real, 265]

        # Run through your fusion MLP
        fused_real = self.prompt_fusion_mlp(fused_in)              # [B, N_real, 256]

        # ----------------------------------------------------------------------
        # 4) Write fused real tokens back into SAM token tensor
        # ----------------------------------------------------------------------
        fused_prompts = sam_sparse.clone()                         # [B, Np, 256]
        fused_prompts[:, :N_real, :] = fused_real                  # Replace real tokens

        return fused_prompts     # [B, Np, 256]    

    def forward(
        self,
        sam_pre: torch.Tensor,      # [B,N,3,1024,1024]
        point_coords_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i, 2] in original coords
        point_labels_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i]
        point_frame_indices_list: Optional[List[torch.Tensor]] = None,  # len B, each [Np_i] - frame index for each point
        multimask_output: bool = True,
        visualize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-view, multi-sample forward:
            - Encodes all images across the batch at once with VGGT and SAM.
            - Then loops per sample to run prompt encoding, fusion, decoding, and postprocess.

        Returns:
            A list (len B) of dicts with keys: masks, iou_predictions, low_res_logits, (optional embeddings...)
        """
        B, N, C, H, W = sam_pre.shape

        # Flatten batch + frames together
        sam_flat = sam_pre.reshape(B * N, C, H, W)

        # Resize spatial dimensions
        vggt_flat = F.interpolate(
            sam_flat,
            size=(896, 896),
            mode="bicubic",
            align_corners=False
        )

        # Restore multi-view shape
        vggt_pre = vggt_flat.reshape(B, N, C, 896, 896)

        sam_feats_bn, sam_interms = self.encode_sam_batched(sam_pre)                      # [B,N,256,64,64]
        vggt_feats_bn, cam_tokens_bn, _ = self.encode_vggt_batched(vggt_pre)             # [B,N,2048,64,64], [B,N,9]

        # 5) Fuse per-frame features (batched) -> [B,N,256,64,64]
        fused_bn = self.fuse_embeddings_batched(sam_feats_bn, vggt_feats_bn)

        # Prepare constant dense PE once
        dense_pe_1 = self.sam.prompt_encoder.get_dense_pe()  # [1,256,64,64]

        # ============================================================
        # 6) BATCHIFIED prompt encoding
        # ============================================================
        batch_points_tuple = None
        if point_coords_list is not None:
            # Convert lists → tensors

            pc_batch = torch.stack(point_coords_list, dim=0).to(self.device)   # [B,Np,2]
            pl_batch = torch.stack(point_labels_list, dim=0).to(self.device)   # [B,Np]
            batch_points_tuple = (pc_batch, pl_batch)
            
        # Prompt encoder (batched)
        sparse_e, dense_e = self.sam.prompt_encoder(
            points=batch_points_tuple,
            boxes=None,
            masks=None,
        )   # sparse_e=[B,Np,256], dense_e=[B,256,64,64]
        B, Np, _ = sparse_e.shape    # after prompt encoder
        # print("B: ", B)
        # print("Np: ", Np)
        # ============================================================
        # 7) TILE dense embeddings and PE across frames (batched)
        # ============================================================
        dense_e_cat = dense_e.repeat(1, 1, 1, N)        # [B,256,64,64N]
        dense_pe_cat = dense_pe_1.repeat(B, 1, 1, N)    # [B,256,64,64N]

        # ============================================================
        # 8) Flatten embeddings across frames for SAM decoder
        # ============================================================
        concat_embed_bn = torch.cat([fused_bn[:, i] for i in range(N)], dim=3)
        # ============================================================
        # 9) Batchify frame indices
        # ============================================================
        prompt_frame_idx = torch.stack(point_frame_indices_list, dim=0).to(self.device)  # [B,N_real]

        # ============================================================
        # 10) Fuse prompts (batched)
        # ============================================================
        fused_prompts = self.fuse_prompts(
            sam_sparse=sparse_e,          # [B,Np,256]
            vggt_cam=cam_tokens_bn,       # [B,N,9]
            prompt_frame_idx=prompt_frame_idx
        )                                 # [B,Np,256]

        # ============================================================
        # 11) Decode in batch
        # ============================================================
        low_res_masks_bn, iou_pred_bn = self.sam.mask_decoder(
            image_embeddings=concat_embed_bn,     # [B,256,64,64N]
            image_pe=dense_pe_cat,               # [B,256,64,64N]
            sparse_prompt_embeddings=fused_prompts,
            dense_prompt_embeddings=dense_e_cat,
            multimask_output=multimask_output,
        )
        outputs = {
            "iou_predictions": iou_pred_bn,
            "low_res_logits": low_res_masks_bn,
        }
        if visualize:
        # ============================================================
        # 12) Postprocess per sample (only step requiring loop)
        # ============================================================

            masks_b = self.sam.postprocess_masks(
                low_res_masks_bn,
                input_size=(sam_pre.shape[-2], sam_pre.shape[-1] * N),
                original_size=(H, W * N),
            )
            masks_b = masks_b > self.mask_threshold
            outputs["masks"] = masks_b

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

