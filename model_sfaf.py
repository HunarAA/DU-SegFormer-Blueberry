from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SmallFruitAdaptiveFusionDecoder(nn.Module):
    def __init__(self, hidden_sizes, decoder_dim=64, num_classes=2):
        super().__init__()
        c1, c2, c3, c4 = hidden_sizes

        self.proj1 = nn.Conv2d(c1, decoder_dim, 1, bias=False)
        self.proj2 = nn.Conv2d(c2, decoder_dim, 1, bias=False)
        self.proj3 = nn.Conv2d(c3, decoder_dim, 1, bias=False)
        self.proj4 = nn.Conv2d(c4, decoder_dim, 1, bias=False)

        self.detail_fusion = DepthwiseSeparableConv(decoder_dim * 2, decoder_dim)
        self.semantic_fusion = DepthwiseSeparableConv(decoder_dim * 2, decoder_dim)

        self.gate = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, 1),
            nn.Sigmoid(),
        )

        self.refine = DepthwiseSeparableConv(decoder_dim * 2, decoder_dim)
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self):
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
        gate = self.gate(torch.cat([detail, semantic], dim=1))
        fused = gate * detail + (1.0 - gate) * semantic
        fused = self.refine(torch.cat([fused, p1], dim=1))
        fused = self.dropout(fused)
        return self.classifier(fused)


class SmallFruitSegFormer(nn.Module):
    def __init__(self, model_name="nvidia/mit-b0", num_classes=2, decoder_dim=64):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained(model_name)
        self.decode_head = SmallFruitAdaptiveFusionDecoder(
            hidden_sizes=self.encoder.config.hidden_sizes,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, pixel_values):
        outputs = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = self.decode_head(outputs.hidden_states[-4:])
        return SimpleNamespace(logits=logits, aux_logits=None)


def build_model(model_name="nvidia/mit-b0", num_classes=2):
    return SmallFruitSegFormer(model_name=model_name, num_classes=num_classes, decoder_dim=64)
