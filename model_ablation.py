"""
Ablation variants for the SFAF decoder (Step 1 of the revision).

The original model_sfaf.py is NOT modified. This file adds:
    segformer_dec64   - standard SegFormer-B0 with a 64-channel MLP decoder (instead of 256)
    sfaf_add          - SFAF with the learned gate replaced by addition      F = D + S
    sfaf_concat       - SFAF with the gate replaced by concat + 1x1 conv     F = Conv([D, S])
    sfaf_noreinject   - SFAF (with gate) but WITHOUT concatenating P1 before refinement

Run `python model_ablation.py` once to check that every variant builds and to
print parameter counts. The "sfaf_gate" check must match the original SFAF
(about 3.394 M parameters).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerForSemanticSegmentation

from model_sfaf import DepthwiseSeparableConv


class AblationSFAFDecoder(nn.Module):
    def __init__(self, hidden_sizes, decoder_dim=64, num_classes=2,
                 fusion="gate", reinject=True):
        super().__init__()
        if fusion not in ("gate", "add", "concat"):
            raise ValueError(f"Unknown fusion: {fusion}")

        self.fusion = fusion
        self.reinject = reinject
        c1, c2, c3, c4 = hidden_sizes

        self.proj1 = nn.Conv2d(c1, decoder_dim, 1, bias=False)
        self.proj2 = nn.Conv2d(c2, decoder_dim, 1, bias=False)
        self.proj3 = nn.Conv2d(c3, decoder_dim, 1, bias=False)
        self.proj4 = nn.Conv2d(c4, decoder_dim, 1, bias=False)

        self.detail_fusion = DepthwiseSeparableConv(decoder_dim * 2, decoder_dim)
        self.semantic_fusion = DepthwiseSeparableConv(decoder_dim * 2, decoder_dim)

        if fusion == "gate":
            # Identical to the original SFAF gate
            self.gate = nn.Sequential(
                nn.Conv2d(decoder_dim * 2, decoder_dim, 1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.GELU(),
                nn.Conv2d(decoder_dim, decoder_dim, 1),
                nn.Sigmoid(),
            )
        elif fusion == "concat":
            self.concat_proj = nn.Sequential(
                nn.Conv2d(decoder_dim * 2, decoder_dim, 1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.GELU(),
            )
        # fusion == "add" needs no extra layers

        refine_in = decoder_dim * 2 if reinject else decoder_dim
        self.refine = DepthwiseSeparableConv(refine_in, decoder_dim)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        # Same initialization as the original SFAF decoder
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features):
        f1, f2, f3, f4 = features
        target_size = f1.shape[-2:]

        p1 = self.proj1(f1)
        p2 = F.interpolate(self.proj2(f2), size=target_size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=target_size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=target_size, mode="bilinear", align_corners=False)

        detail = self.detail_fusion(torch.cat([p1, p2], dim=1))
        semantic = self.semantic_fusion(torch.cat([p3, p4], dim=1))

        if self.fusion == "gate":
            gate = self.gate(torch.cat([detail, semantic], dim=1))
            fused = gate * detail + (1.0 - gate) * semantic
        elif self.fusion == "add":
            fused = detail + semantic
        else:  # concat
            fused = self.concat_proj(torch.cat([detail, semantic], dim=1))

        if self.reinject:
            fused = torch.cat([fused, p1], dim=1)

        fused = self.refine(fused)
        fused = self.dropout(fused)
        return self.classifier(fused)


class AblationSFAFSegFormer(nn.Module):
    def __init__(self, model_name="nvidia/mit-b0", num_classes=2, decoder_dim=64,
                 fusion="gate", reinject=True):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(model_name)
        self.decode_head = AblationSFAFDecoder(
            hidden_sizes=self.encoder.config.hidden_sizes,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
            fusion=fusion,
            reinject=reinject,
        )

    def forward(self, pixel_values):
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.decode_head(outputs.hidden_states[-4:])
        return SimpleNamespace(logits=logits, aux_logits=None)


# ------------------------------------------------------------
# Builders (same signature as build_model in your other files)
# ------------------------------------------------------------

def build_segformer_dec64(model_name="nvidia/mit-b0", num_classes=2):
    """Standard SegFormer-B0, only the decoder width changed from 256 to 64."""
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        decoder_hidden_size=64,
        id2label={0: "background", 1: "blueberry"},
        label2id={"background": 0, "blueberry": 1},
    )


def build_sfaf_add(model_name="nvidia/mit-b0", num_classes=2):
    return AblationSFAFSegFormer(model_name, num_classes, fusion="add", reinject=True)


def build_sfaf_concat(model_name="nvidia/mit-b0", num_classes=2):
    return AblationSFAFSegFormer(model_name, num_classes, fusion="concat", reinject=True)


def build_sfaf_noreinject(model_name="nvidia/mit-b0", num_classes=2):
    return AblationSFAFSegFormer(model_name, num_classes, fusion="gate", reinject=False)


# ------------------------------------------------------------
# Quick check: python model_ablation.py
# ------------------------------------------------------------

if __name__ == "__main__":
    from model_sfaf import build_model as build_original_sfaf

    variants = {
        "original SFAF (reference)": build_original_sfaf,
        "sfaf_gate (should match)": lambda **kw: AblationSFAFSegFormer(fusion="gate", reinject=True),
        "segformer_dec64": build_segformer_dec64,
        "sfaf_add": build_sfaf_add,
        "sfaf_concat": build_sfaf_concat,
        "sfaf_noreinject": build_sfaf_noreinject,
    }

    x = torch.randn(1, 3, 640, 640)
    print(f"{'Variant':<28}{'Params (M)':>12}   Output shape")
    for name, builder in variants.items():
        model = builder(model_name="nvidia/mit-b0", num_classes=2).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            out = model(pixel_values=x).logits
        print(f"{name:<28}{params:>12.3f}   {tuple(out.shape)}")
