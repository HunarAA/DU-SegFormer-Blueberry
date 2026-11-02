"""
Novelty candidate 1: learned (content-aware) upsampling in the SegFormer-B0 decoder.

Starts from the exact same 64-channel SegFormer-B0 used for the final recipe
(abl.build_segformer_dec64) and changes ONLY how features are upsampled:

    dysample        Stage 2-4 features are upsampled to 1/4 resolution with
                    DySample (Liu et al., ICCV 2023, "Learning to Upsample by
                    Learning to Sample", lp style with dynamic scope) instead of
                    bilinear interpolation. Output still at 1/4 resolution.
    dysample_half   As above, plus the fused 1/4-resolution features are
                    upsampled x2 with DySample before the classifier, so the
                    model predicts at 1/2 resolution (320 x 320).
    bilinear_half   Control for dysample_half: bilinear decoder, but fused
                    features upsampled x2 with bilinear interpolation before the
                    classifier. Separates "higher output resolution" from
                    "learned upsampling".

Works with transformers 4.x (decode_head.linear_c) and 5.x (linear_projections).

Quick check:  python model_novel.py
"""

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

import model_ablation as abl


class DySample(nn.Module):
    """DySample, 'lp' style with dynamic scope (DySample+)."""

    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        assert in_channels % groups == 0
        self.scale = scale
        self.groups = groups
        out_channels = 2 * groups * scale ** 2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.scope.weight)
        self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2)
        return grid.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.float().view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device, dtype=torch.float32) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=torch.float32) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([W, H], device=x.device, dtype=torch.float32).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
        coords = coords.view(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        out = F.grid_sample(x.float().reshape(B * self.groups, -1, H, W), coords,
                            mode="bilinear", align_corners=False, padding_mode="border")
        return out.view(B, -1, self.scale * H, self.scale * W).to(x.dtype)

    def forward(self, x):
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos.to(x.dtype)
        return self._sample(x, offset)


def _projections(head):
    return head.linear_c if hasattr(head, "linear_c") else head.linear_projections


def _decoder_forward(self, encoder_hidden_states, **kwargs):
    batch_size = encoder_hidden_states[-1].shape[0]
    target = encoder_hidden_states[0].shape[2:]
    feats = []
    for i, (state, proj) in enumerate(zip(encoder_hidden_states, _projections(self))):
        h, w = state.shape[2], state.shape[3]
        x = proj(state).transpose(1, 2).reshape(batch_size, -1, h, w)
        if i > 0:
            if self.use_dysample:
                x = self.stage_upsamplers[i - 1](x)
            else:
                x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
        feats.append(x)

    x = self.linear_fuse(torch.cat(feats[::-1], dim=1))
    x = self.activation(self.batch_norm(x))

    if self.final_upsample == "dysample":
        x = self.final_upsampler(x)
    elif self.final_upsample == "bilinear":
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    return self.classifier(self.dropout(x))


def _build(model_name, num_classes, use_dysample, final_upsample):
    model = abl.build_segformer_dec64(model_name=model_name, num_classes=num_classes)
    head = model.decode_head
    dim = head.classifier.in_channels  # 64
    head.use_dysample = use_dysample
    head.final_upsample = final_upsample
    if use_dysample:
        head.stage_upsamplers = nn.ModuleList([DySample(dim, scale=s) for s in (2, 4, 8)])
    if final_upsample == "dysample":
        head.final_upsampler = DySample(dim, scale=2)
    head.forward = MethodType(_decoder_forward, head)
    return model


def build_dysample(model_name="nvidia/mit-b0", num_classes=2):
    return _build(model_name, num_classes, use_dysample=True, final_upsample=None)


def build_dysample_half(model_name="nvidia/mit-b0", num_classes=2):
    return _build(model_name, num_classes, use_dysample=True, final_upsample="dysample")


def build_bilinear_half(model_name="nvidia/mit-b0", num_classes=2):
    return _build(model_name, num_classes, use_dysample=False, final_upsample="bilinear")


if __name__ == "__main__":
    import os
    # Tip: after the first run, set HF_HUB_OFFLINE=1 to work without internet.
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, 3, 640, 640)
    print(f"{'Variant':<22}{'Params (M)':>12}{'GMACs':>8}   Output")
    for name, builder in [("baseline 64-ch", abl.build_segformer_dec64), ("dysample", build_dysample),
                          ("dysample_half", build_dysample_half), ("bilinear_half", build_bilinear_half)]:
        m = builder().eval()
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            out = m(pixel_values=x).logits
        params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"{name:<22}{params:>12.3f}{counter.get_total_flops() / 2e9:>8.2f}   {tuple(out.shape)}")
