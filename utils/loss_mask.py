import torch
from torch.nn import functional as F
from typing import List, Optional
import utils.misc as misc

# def point_sample(input, point_coords, **kwargs):
#     """
#     A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
#     Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
#     [0, 1] x [0, 1] square.
#     Args:
#         input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
#         point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
#         [0, 1] x [0, 1] normalized point coordinates.
#     Returns:
#         output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
#             features for points in `point_coords`. The features are obtained via bilinear
#             interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
#     """
#     add_dim = False
#     if point_coords.dim() == 3:
#         add_dim = True
#         point_coords = point_coords.unsqueeze(2)
#     output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
#     if add_dim:
#         output = output.squeeze(3)
#     return output

def point_sample(input, point_coords, mode="bilinear", **kwargs):
    # print("point sample mode: ", mode)
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, mode=mode, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.
    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.
    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords

# def dice_loss(
#         inputs: torch.Tensor,
#         targets: torch.Tensor,
#         num_masks: float,
#     ):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * (inputs * targets).sum(-1)
#     denominator = inputs.sum(-1) + targets.sum(-1)
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss.sum() / num_masks


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute DICE loss on full masks.
    inputs: logits, shape (N,1,H,W) or (N,H,W) etc.
    targets: binary labels, same spatial shape
    """
    inputs = inputs.sigmoid()

    # flatten BOTH
    inputs = inputs.flatten(1)
    targets = targets.float().flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks



dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        pos_weight: torch.Tensor = None,
    ):

    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", pos_weight=pos_weight)

    return loss.mean(1).sum() / num_masks



sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

# def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0, debug=False):
#     """Compute the losses related to the masks: the focal loss and the dice loss.
#     targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#     """

#     # No need to upsample predictions as we are using normalized coordinates :)

#     with torch.no_grad():
#         # sample point_coords
#         point_coords = get_uncertain_point_coords_with_randomness(
#             src_masks,
#             lambda logits: calculate_uncertainty(logits),
#             112 * 112,
#             oversample_ratio,
#             0.75,
#         )
#         # get gt labels
#         point_labels = point_sample(
#             target_masks.float(),
#             point_coords,
#             mode="nearest",
#             align_corners=False,
#         ).squeeze(1)

#     point_logits = point_sample(
#         src_masks,
#         point_coords,
#         mode="bilinear",
#         align_corners=False,
#     ).squeeze(1)
#     if debug:
#         with torch.no_grad():
#             print("LOSS point_labels shape:", tuple(point_labels.shape))
#             print("LOSS point_labels mean (fg frac):", (point_labels > 0.5).float().mean().item())
#             probs = point_logits.sigmoid()
#             print("LOSS probs mean:", probs.mean().item())
#             if (point_labels > 0.5).any():
#                 print("LOSS probs on fg:", probs[point_labels > 0.5].mean().item())

#     # with torch.no_grad():
#     #     p = point_labels.mean().clamp(1e-6, 1-1e-6)
#     #     pos_w = ((1 - p) / p).detach()  # scalar
#     # pos_weight = pos_w.unsqueeze(0)  # shape [1] works

#     loss_mask = sigmoid_ce_loss(point_logits, point_labels, num_masks, pos_weight=None)
#     # loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)
#     loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

#     del src_masks
#     del target_masks
#     return loss_mask, loss_dice

import torch
import torch.nn.functional as F

def _ensure_nchw_1(m: torch.Tensor) -> torch.Tensor:
    # Accept (N,H,W) or (N,1,H,W); return (N,1,H,W)
    if m.dim() == 3:
        return m.unsqueeze(1)
    assert m.dim() == 4, f"Expected 3D or 4D mask, got {m.shape}"
    return m

def _resize_logits_to(logits: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # logits: (N,1,H,W) -> (N,1,h,w) in logit space
    if logits.shape[-2:] == (h, w):
        return logits
    return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

def _resize_targets_to(targets: torch.Tensor, h: int, w: int) -> torch.Tensor:
    # targets: (N,1,H,W) -> (N,1,h,w) as labels
    if targets.shape[-2:] == (h, w):
        return targets
    return F.interpolate(targets.float(), size=(h, w), mode="nearest")

# def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0, debug=False):
#     """
#     CE: point-sampled (uncertainty-based)
#     Dice: full-mask (spatially aligned even if src/target have different sizes)
#     """
#     src_masks = _ensure_nchw_1(src_masks)
#     target_masks = _ensure_nchw_1(target_masks)

#     # ---------- point-sampled CE ----------
#     with torch.no_grad():
#         point_coords = get_uncertain_point_coords_with_randomness(
#             src_masks,
#             lambda logits: calculate_uncertainty(logits),
#             112 * 112,
#             oversample_ratio,
#             0.75,
#         )
#         point_labels = point_sample(
#             target_masks.float(),
#             point_coords,
#             mode="nearest",
#             align_corners=False,
#         ).squeeze(1)

#     point_logits = point_sample(
#         src_masks,
#         point_coords,
#         mode="bilinear",
#         align_corners=False,
#     ).squeeze(1)

#     if debug:
#         with torch.no_grad():
#             print("LOSS point_labels shape:", tuple(point_labels.shape))
#             print("LOSS point_labels mean (fg frac):", (point_labels > 0.5).float().mean().item())
#             probs = point_logits.sigmoid()
#             print("LOSS probs mean:", probs.mean().item())
#             if (point_labels > 0.5).any():
#                 print("LOSS probs on fg:", probs[point_labels > 0.5].mean().item())
#             with torch.no_grad():
#                 fg = point_logits[point_labels > 0.5]
#                 bg = point_logits[point_labels <= 0.5]
#                 print("sampled fg mean:", fg.mean().item() if fg.numel() else None)
#                 print("sampled bg mean:", bg.mean().item() if bg.numel() else None)
#                 print("sampled fg p95 :", fg.kthvalue(int(0.95*fg.numel()))[0].item() if fg.numel() else None)
#                 print("sampled bg p95 :", bg.kthvalue(int(0.95*bg.numel()))[0].item() if bg.numel() else None)


#     loss_mask = sigmoid_ce_loss(point_logits, point_labels, num_masks, pos_weight=None)


#     # ---------- full-mask Dice ----------
#     # Align to target resolution (common choice); resize logits in logit-space
#     Ht, Wt = target_masks.shape[-2], target_masks.shape[-1]
#     src_full = _resize_logits_to(src_masks, Ht, Wt)          # (N,1,Ht,Wt)
#     tgt_full = _resize_targets_to(target_masks, Ht, Wt)      # (N,1,Ht,Wt)
#     # dice_loss flattens inputs from dim=1 onward, so shapes match
#     loss_dice = dice_loss_jit(src_full, tgt_full, num_masks)

#     del src_masks
#     del target_masks
#     return loss_mask, loss_dice

import torch
import torch.nn.functional as F

def loss_masks(
    src_masks,
    target_masks,
    num_masks,
    oversample_ratio=3.0,
    debug=False,
    ce_mode: str = "point",          # "point" or "dense"
    use_pos_weight: bool = False,    # only used in dense CE
    dense_ce_on: str = "src",        # "src" uses src_masks resolution; "tgt" resizes to target resolution first
):
    """
    ce_mode="point":
        CE on uncertainty-sampled points (your current behavior)
    ce_mode="dense":
        CE on all pixels (dense supervision), optionally with pos_weight from fg fraction

    Dice:
        Full-mask Dice on aligned resolution
    """
    src_masks = _ensure_nchw_1(src_masks)         # [B,1,h,w] logits
    target_masks = _ensure_nchw_1(target_masks)   # [B,1,H,W] labels (0/1)

    # ----------------------------
    # 1) Mask loss (CE)
    # ----------------------------
    if ce_mode == "point":
        # ---------- point-sampled CE ----------
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                112 * 112,
                oversample_ratio,
                0.75,
            )
            point_labels = point_sample(
                target_masks.float(),
                point_coords,
                mode="nearest",
                align_corners=False,
            ).squeeze(1)  # [B,P]

        point_logits = point_sample(
            src_masks,
            point_coords,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [B,P]

        if debug:
            with torch.no_grad():
                print("LOSS[point] point_labels shape:", tuple(point_labels.shape))
                print("LOSS[point] fg frac:", (point_labels > 0.5).float().mean().item())
                probs = point_logits.sigmoid()
                print("LOSS[point] probs mean:", probs.mean().item())
                if (point_labels > 0.5).any():
                    print("LOSS[point] probs on fg:", probs[point_labels > 0.5].mean().item())

                fg = point_logits[point_labels > 0.5]
                bg = point_logits[point_labels <= 0.5]
                print("LOSS[point] sampled fg mean:", fg.mean().item() if fg.numel() else None)
                print("LOSS[point] sampled bg mean:", bg.mean().item() if bg.numel() else None)

        loss_mask = sigmoid_ce_loss(point_logits, point_labels, num_masks, pos_weight=None)

    elif ce_mode == "dense":
        # ---------- dense CE (all pixels) ----------
        # Choose resolution to compute CE on
        if dense_ce_on == "tgt":
            Ht, Wt = target_masks.shape[-2], target_masks.shape[-1]
            src_ce = _resize_logits_to(src_masks, Ht, Wt)          # [B,1,Ht,Wt]
            tgt_ce = _resize_targets_to(target_masks, Ht, Wt)      # [B,1,Ht,Wt]
        else:
            h, w = src_masks.shape[-2], src_masks.shape[-1]
            src_ce = src_masks                                   # [B,1,h,w]
            tgt_ce = _resize_targets_to(target_masks, h, w)      # [B,1,h,w]
        print("tgt fg frac (low-res):", (tgt_ce > 0.5).float().mean().item())

        pos_weight = None
        if use_pos_weight:
            # scalar pos_weight = (1-p)/p computed from current batch target fg frac
            with torch.no_grad():
                p = tgt_ce.float().mean().clamp(1e-6, 1 - 1e-6)   # foreground fraction
                pos_weight = ((1 - p) / p).to(src_ce.device)

        # BCEWithLogits over all pixels
        loss_mask = F.binary_cross_entropy_with_logits(
            src_ce, tgt_ce.float(), reduction="mean", pos_weight=pos_weight
        ) / float(num_masks)

        if debug:
            with torch.no_grad():
                fg = src_ce[tgt_ce > 0.5]
                bg = src_ce[tgt_ce <= 0.5]
                print("LOSS[dense] fg frac:", (tgt_ce > 0.5).float().mean().item())
                print("LOSS[dense] pos_weight:", float(pos_weight) if pos_weight is not None else None)
                print("LOSS[dense] fg mean:", fg.mean().item() if fg.numel() else None)
                print("LOSS[dense] bg mean:", bg.mean().item() if bg.numel() else None)
                # quick sanity on distribution width
                print("LOSS[dense] logits min/max:", src_ce.min().item(), src_ce.max().item())
                

    else:
        raise ValueError(f"Unknown ce_mode={ce_mode}. Use 'point' or 'dense'.")

    # ----------------------------
    # 2) Dice loss (full mask)
    # ----------------------------
    Ht, Wt = target_masks.shape[-2], target_masks.shape[-1]
    src_full = _resize_logits_to(src_masks, Ht, Wt)          # [B,1,Ht,Wt]
    tgt_full = _resize_targets_to(target_masks, Ht, Wt)      # [B,1,Ht,Wt]
    loss_dice = dice_loss_jit(src_full, tgt_full, num_masks)

    del src_masks
    del target_masks
    return loss_mask, loss_dice



def dice_loss_debug(inputs_logits: torch.Tensor, targets: torch.Tensor, num_masks: float, prefix=""):
    # inputs_logits, targets: [B, P] (point logits + point labels)
    probs = inputs_logits.sigmoid()

    # Sanity stats
    with torch.no_grad():
        print(f"{prefix} probs:   min={probs.min().item():.4f} max={probs.max().item():.4f} mean={probs.mean().item():.4f}")
        print(f"{prefix} target:  min={targets.min().item():.4f} max={targets.max().item():.4f} mean={targets.mean().item():.4f}")
        # How binary is target?
        near0 = (targets < 0.1).float().mean().item()
        near1 = (targets > 0.9).float().mean().item()
        mid  = ((targets >= 0.1) & (targets <= 0.9)).float().mean().item()
        print(f"{prefix} target binarity: near0={near0:.3f} near1={near1:.3f} mid={mid:.3f}")

    # Dice computation (same as your code)
    probs_f = probs.flatten(1)
    targets_f = targets.flatten(1)
    numerator = 2 * (probs_f * targets_f).sum(-1)
    denominator = probs_f.sum(-1) + targets_f.sum(-1)
    loss_per = 1 - (numerator + 1) / (denominator + 1)
    loss = loss_per.sum() / num_masks

    with torch.no_grad():
        print(f"{prefix} dice: numerator mean={numerator.mean().item():.4f}, denom mean={denominator.mean().item():.4f}")
        print(f"{prefix} dice_loss per-sample: {loss_per.detach().cpu().numpy()[:8]}")
        print(f"{prefix} dice_loss reduced: {loss.item():.6f} (num_masks={num_masks})")

    return loss



def loss_masks_debug(src_masks, target_masks, num_masks, oversample_ratio=3.0, debug_every=1, step=0):
    """
    src_masks:    [B,1,hp,wp] logits
    target_masks: [B,1,H,W]   float (0/1 preferred)
    """
    assert src_masks.ndim == 4 and target_masks.ndim == 4, (src_masks.shape, target_masks.shape)
    B = src_masks.shape[0]

    # IMPORTANT: in your training you often set num_masks=1.0; for stability/debug, use B
    if isinstance(num_masks, (int, float)) and num_masks == 1.0:
        pass  # keep as-is, but you should try num_masks=float(B)

    if step % debug_every == 0:
        print("\n================ loss_masks DEBUG ================")
        print("src_masks shape:", tuple(src_masks.shape), "dtype:", src_masks.dtype, "device:", src_masks.device)
        print("target_masks shape:", tuple(target_masks.shape), "dtype:", target_masks.dtype, "device:", target_masks.device)
        with torch.no_grad():
            print("src_logits stats:", src_masks.min().item(), src_masks.max().item(), src_masks.mean().item())
            print("target stats:", target_masks.min().item(), target_masks.max().item(), target_masks.mean().item())

    with torch.no_grad():
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            112 * 112,
            oversample_ratio,
            0.75,
        )
        # [B, P, 2] in [0,1]
        point_labels = point_sample(target_masks.float(), point_coords, mode="bilinear", align_corners=False).squeeze(1)  # [B,P]

        if step % debug_every == 0:
            print("point_coords:", tuple(point_coords.shape),
                  "min/max:", point_coords.min().item(), point_coords.max().item())
            print("point_labels:", tuple(point_labels.shape),
                  "min/max/mean:", point_labels.min().item(), point_labels.max().item(), point_labels.mean().item())
            # fraction of sampled points that are foreground-ish
            fg_frac = (point_labels > 0.5).float().mean().item()
            print("sampled FG fraction (labels>0.5):", fg_frac)

    point_logits = point_sample(src_masks, point_coords, mode = 'bilinear',align_corners=False).squeeze(1)  # [B,P]

    if step % debug_every == 0:
        with torch.no_grad():
            probs = point_logits.sigmoid()
            print("point_logits stats:", point_logits.min().item(), point_logits.max().item(), point_logits.mean().item())
            print("point_probs stats:", probs.min().item(), probs.max().item(), probs.mean().item())
            # quick correlation check
            corr = torch.corrcoef(torch.stack([probs.flatten(), point_labels.flatten()]))[0,1].item()
            print("corr(prob, label):", corr)

    # Compute losses
    loss_mask = sigmoid_ce_loss(point_logits, point_labels, num_masks)

    if step % debug_every == 0:
        loss_dice = dice_loss_debug(point_logits, point_labels, num_masks, prefix="[points] ")
    else:
        loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)

    # Optional: compare with FULL-MASK dice (upsample preds to GT size)
    if step % debug_every == 0:
        with torch.no_grad():
            pred_up = F.interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        full_dice = dice_loss_debug(pred_up.squeeze(1), target_masks.squeeze(1), num_masks, prefix="[full]  ")
        print("[compare] point-dice vs full-dice:", loss_dice.item(), full_dice.item())
        print("==================================================\n")

    return loss_mask, loss_dice
