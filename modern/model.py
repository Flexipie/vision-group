import torch
import torch.nn as nn
import torchvision.models as models
import config


class FatigueNet(nn.Module):
    """
    CNN-LSTM architecture for temporal fatigue detection.

    Backbone : MobileNetV2, optionally pretrained on ImageNet, gives one
               1280-dim feature vector per frame.
    Temporal : 2-layer LSTM over a sliding window of FRAME_BUFFER_SIZE frames
    Head     : FC → 2 classes (0=alert, 1=drowsy)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        pretrained=False,
    ):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────────
        weights = None
        if pretrained:
            try:
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            except AttributeError:
                weights = models.MobileNet_V2_Weights.DEFAULT

        try:
            backbone = models.mobilenet_v2(weights=weights)
        except Exception as e:
            print(f"[FatigueNet] Could not load pretrained weights ({e}); using random init.")
            backbone = models.mobilenet_v2(weights=None)
        # Remove the classifier head; keep feature extractor
        self.cnn = backbone.features          # output: (B, 1280, 4, 4) for 112x112 input
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 1280, 1, 1)
        self.cnn_out_dim = 1280

        # ── LSTM ──────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, C, H, W)
        Returns logits: (batch, num_classes)
        """
        B, T, C, H, W = x.shape

        # Extract CNN features for all frames
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)               # (B*T, 1280, h, w)
        feats = self.pool(feats)          # (B*T, 1280, 1, 1)
        feats = feats.view(B, T, -1)      # (B, T, 1280)

        return self.forward_from_features(feats)

    def forward_from_features(self, feats):
        """
        feats: (batch, seq_len, 1280) precomputed CNN features.
        Returns logits: (batch, num_classes)
        """
        B, T, D = feats.shape

        # Temporal modelling
        lstm_out, _ = self.lstm(feats)    # (B, T, hidden)
        last = lstm_out[:, -1, :]         # use last timestep

        return self.classifier(last)      # (B, num_classes)

    def predict_proba(self, x):
        """Returns drowsy probability for a batch."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
        return probs[:, 1]               # drowsy probability
