# SAM-VGGT Model

A multi-view segmentation model that combines **SAM (Segment Anything Model)** and **VGGT (Visual Geometry Grounded Transformer)** for robust object segmentation across multiple camera views.

## Overview

The SAM-VGGT model integrates two powerful encoders:
- **SAM Image Encoder**: Extracts rich single-view features for segmentation
- **VGGT Aggregator**: Processes multi-view images with geometry-aware attention

The model fuses these complementary representations using learnable MLPs, enabling accurate segmentation with multi-view context.

## Architecture

```
Input: Multi-view images + point/box prompts
    │
    ├─→ SAM Encoder ──→ [N, 256, 64, 64]
    │                         │
    └─→ VGGT Aggregator ──→ [N, 2048, 64, 64]
                              │
                    ┌─────────┴─────────┐
                    │  Per-pixel MLP    │ (Fusion)
                    └─────────┬─────────┘
                              │
                        [N, 256, 64, 64]
                              │
                    ┌─────────┴─────────┐
                    │ Spatial Concat    │
                    └─────────┬─────────┘
                              │
                      [1, 256, 64, 64*N]
                              │
    ┌─────────────────────────┴─────────────────────────┐
    │                                                     │
    ├─→ Prompt Encoder + Camera Token Fusion             │
    │   [SAM sparse prompts + VGGT camera tokens]        │
    │                                                     │
    └─────────────────────────┬─────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Mask Decoder    │
                    └─────────┬─────────┘
                              │
                    Output: Segmentation Masks
```

## Key Components

### 1. **Dual Encoders**
- **SAM Encoder**: Pre-trained ViT-based encoder that extracts semantic features
- **VGGT Encoder**: Multi-view aggregator with alternating frame/global attention

### 2. **Fusion Modules**

#### Per-Pixel Embedding Fusion
Fuses SAM and VGGT spatial features using 1x1 convolutions:
- Input: `[N, 2304, 64, 64]` (256 + 2048 channels)
- Hidden: `[N, 768, 64, 64]`
- Output: `[N, 256, 64, 64]`

#### Prompt-Camera Fusion
Combines SAM prompt embeddings with VGGT camera tokens:
- Input: SAM sparse prompts `[B, Np, 256]` + Camera tokens `[1, N, 2048]`
- Hidden: `[B, Np, 768]`
- Output: `[B, Np, 256]` (geometry-aware prompts)

### 3. **Spatial Concatenation**
Multi-view features are concatenated along the spatial dimension:
- Per-frame: `[N, 256, 64, 64]`
- Concatenated: `[1, 256, 64, 64*N]`

This allows the decoder to process all views jointly while maintaining spatial coherence.

## Usage

### Installation

Ensure you have both SAM-HQ and VGGT repositories set up:

```bash
# Install SAM-HQ
cd sam-hq
pip install -e .

# Install VGGT
cd ../vggt
pip install -e .
```

### Basic Usage

```python
from sam_vggt_model import build_sam_vggt
import torch
import numpy as np

# Initialize model
model = build_sam_vggt(
    sam_model_type="vit_l",
    sam_checkpoint="sam-hq/checkpoints/sam_hq_vit_l.pth",
    vggt_checkpoint="vggt/checkpoints/model.pt",
    device="cuda",
)

# Prepare inputs
image_paths = ["view1.png", "view2.png", "view3.png", "view4.png"]
point_coords = torch.tensor([[[400, 200]]])  # [1, 1, 2] - (x, y)
point_labels = torch.tensor([[1]])           # [1, 1] - 1=foreground

# Forward pass
outputs = model(
    image_paths=image_paths,
    point_coords=point_coords,
    point_labels=point_labels,
    prompt_frame_idx=0,  # Prompt is on the first image
    multimask_output=True,
)

# Get results
masks = outputs["masks"]                     # [B, C, H, W*N]
iou_predictions = outputs["iou_predictions"] # [B, C]
low_res_logits = outputs["low_res_logits"]   # [B, C, 256, 256*N]
```

### Example Script

Run the provided example:

```bash
python example_sam_vggt.py
```

This will:
1. Load multi-view images from `data/kitchen/images/`
2. Apply a point prompt on the first frame
3. Generate segmentation masks across all views
4. Save visualizations

## Model Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sam_model_type` | str | `"vit_l"` | SAM model variant (`vit_b`, `vit_l`, `vit_h`) |
| `sam_checkpoint` | str | - | Path to SAM checkpoint file |
| `vggt_checkpoint` | str | - | Path to VGGT checkpoint file |
| `vggt_img_size` | int | `896` | Image size for VGGT preprocessing |
| `embed_fusion_hidden` | int | `768` | Hidden dim for embedding fusion MLP |
| `prompt_fusion_hidden` | int | `768` | Hidden dim for prompt fusion MLP |
| `freeze_sam_encoder` | bool | `True` | Whether to freeze SAM encoder weights |
| `freeze_vggt` | bool | `True` | Whether to freeze VGGT weights |

### Forward Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_paths` | List[str] | - | Paths to input images |
| `point_coords` | Tensor | None | Point prompts `[B, N, 2]` |
| `point_labels` | Tensor | None | Point labels `[B, N]` (1=fg, 0=bg) |
| `boxes` | Tensor | None | Box prompts `[B, 4]` |
| `mask_input` | Tensor | None | Mask prompts `[B, 1, H, W]` |
| `prompt_frame_idx` | int | `0` | Which frame contains the prompt |
| `multimask_output` | bool | `True` | Return multiple mask candidates |
| `return_embeddings` | bool | `False` | Return intermediate embeddings |

## Output Format

The model returns a dictionary with the following keys:

```python
{
    "masks": torch.Tensor,              # [B, C, H, W*N] - Binary masks (thresholded)
    "iou_predictions": torch.Tensor,    # [B, C] - Quality scores for each mask
    "low_res_logits": torch.Tensor,     # [B, C, 256, 256*N] - Low-res mask logits
    
    # Optional (if return_embeddings=True):
    "fused_embeddings": torch.Tensor,         # [N, 256, 64, 64]
    "concatenated_embeddings": torch.Tensor,  # [1, 256, 64, 64*N]
    "camera_tokens": torch.Tensor,            # [1, N, 2048]
    "sam_embeddings": torch.Tensor,           # [N, 256, 64, 64]
    "vggt_embeddings": torch.Tensor,          # [N, 2048, 64, 64]
}
```

## Key Features

### 1. **Multi-View Aware**
The model processes multiple camera views jointly, leveraging geometric relationships through VGGT's camera tokens and alternating attention.

### 2. **Flexible Prompting**
Supports all SAM prompt types:
- Point prompts (positive/negative)
- Bounding box prompts
- Mask prompts
- Combinations of the above

### 3. **Geometry-Aware Prompts**
Point/box prompts are enriched with VGGT's camera tokens, making them aware of the 3D geometry and camera viewpoints.

### 4. **Efficient Inference**
- Frozen encoders for fast inference
- Batch processing of multiple views
- GPU acceleration with mixed precision support

## Implementation Details

### Preprocessing

**SAM Images:**
- Resize longest side to 1024px
- Normalize with ImageNet statistics
- Pad to 1024×1024

**VGGT Images:**
- Resize to 896×896 (square)
- Normalize with ResNet statistics
- Range: [0, 1]

### Embedding Dimensions

- SAM encoder output: `256` channels
- VGGT aggregator output: `2048` channels (2× embed_dim)
- Fused embedding: `256` channels (matches SAM)
- Spatial resolution: `64×64` per frame

### Memory Requirements

For 4 views with ViT-L:
- SAM encoder: ~4GB
- VGGT encoder: ~8GB
- Fusion modules: ~0.1GB
- **Total: ~12GB VRAM**

## Training (Future Work)

The current implementation focuses on forward inference with frozen encoders. For training:

1. Unfreeze fusion MLPs:
```python
model = build_sam_vggt(
    freeze_sam_encoder=True,  # Keep frozen
    freeze_vggt=True,          # Keep frozen
)

# Only fusion MLPs are trainable
```

2. Fine-tune on multi-view segmentation datasets
3. Optionally unfreeze encoders for end-to-end training

## Differences from Notebook

This implementation provides a clean, modular version of the notebook code:

1. **Encapsulation**: All components in a single `SamVGGT` class
2. **Flexibility**: Configurable hyperparameters
3. **Reusability**: Can be imported and used in other projects
4. **Documentation**: Comprehensive docstrings and type hints
5. **Error Handling**: Input validation and meaningful error messages

## Files

- `sam_vggt_model.py`: Main model implementation
- `example_sam_vggt.py`: Example usage script
- `SAM_VGGT_README.md`: This documentation

## Citation

If you use this model, please cite the original works:

**SAM:**
```bibtex
@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

**VGGT:**
```bibtex
@article{wang2024vggt,
  title={Visual Geometry Grounded Deep Structure From Motion},
  author={Wang, Jianyuan and Karaev, Nikita and Rupprecht, Christian and Novotny, David},
  journal={arXiv preprint arXiv:2312.04563},
  year={2024}
}
```

## License

This implementation combines SAM (Apache 2.0) and VGGT (CC BY-NC 4.0). Please refer to the respective licenses for usage terms.


