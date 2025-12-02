import torch
from sam_vggt_model import SamVGGT, build_sam_vggt

model = build_sam_vggt()
model.eval()
# B=2 samples, each with N=3 frames
batch_paths = [
    ["data/kitchen/images/00.png","data/kitchen/images/01.png","data/kitchen/images/02.png"],
    ["data/kitchen/images/03.png","data/kitchen/images/04.png","data/kitchen/images/05.png"],
]

# Optional prompts per sample (can be None)
point_coords_list = [
    torch.tensor([[320., 240.]]),  # sample 0: 1 point
    torch.tensor([[100., 120.], [400., 300.]])  # sample 1: 2 points
]
point_labels_list = [
    torch.tensor([1]),
    torch.tensor([1, 0]),
]
prompt_frame_idx_list = [0, 2]  # prompts belong to frame 0 for sample 0, frame 2 for sample 1

outs = model.forward(
    batch_image_paths=batch_paths,
    point_coords_list=[point_coords_list],
    point_labels_list=[point_labels_list],
    point_frame_indices_list=[prompt_frame_idx_list],
    multimask_output=False,
    return_embeddings=False,
)
