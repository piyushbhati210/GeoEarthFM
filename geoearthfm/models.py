import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class GeoEncoder(nn.Module):
    """
    Lightweight shared GeoAI encoder.
    It can receive RGB (3 ch) or STTORM multispectral (10 ch)
    through modality-specific input adapters, then produces a
    shared representation.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.rgb_adapter = nn.Sequential(
            nn.Conv2d(3, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.ms_adapter = nn.Sequential(
            nn.Conv2d(10, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.stage1 = ConvBlock(32, 64)
        self.stage2 = ConvBlock(64, 128)
        self.stage3 = ConvBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward_features(self, x, modality="rgb"):
        x = self.rgb_adapter(x) if modality == "rgb" else self.ms_adapter(x)
        x = self.stage1(x)
        x = F.max_pool2d(x, 2)
        x = self.stage2(x)
        x = F.max_pool2d(x, 2)
        x = self.stage3(x)
        return x

    def forward(self, x, modality="rgb"):
        f = self.forward_features(x, modality)
        z = self.proj(self.pool(f).flatten(1))
        return F.normalize(z, dim=1)

class LandCoverModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder = GeoEncoder(256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        z = self.encoder(x, "rgb")
        return self.classifier(z), z

class ChangeDetector(nn.Module):
    """
    Siamese change detector using the same GeoEncoder weights for
    before/after images. The decoder predicts a binary change mask.
    """
    def __init__(self):
        super().__init__()
        self.encoder = GeoEncoder(256)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, before, after):
        fb = self.encoder.forward_features(before, "ms")
        fa = self.encoder.forward_features(after, "ms")
        diff = torch.cat([torch.abs(fb-fa), fb+fa], dim=1)
        y = self.decoder(diff)
        return F.interpolate(y, size=before.shape[-2:], mode="bilinear", align_corners=False)
