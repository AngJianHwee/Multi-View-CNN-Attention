import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleViewCNN(nn.Module):
    def __init__(self, channels_in, reshape_size, output_dim=10):
        super(SingleViewCNN, self).__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(channels_in, 64, 3, 1, 1, padding_mode="reflect")

        # Attention mechanism
        self.norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(64, num_heads=1, batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))

        # Additional Convolution Layers
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        # Individual output layer
        self.dropout = nn.Dropout(0.5)
        # self.fc_out = nn.Linear(128 * 4 * 4, output_dim)  # Assuming 32x32 input -> 4x4 after convolutions

        # case not 32
        self.fc_out = nn.Linear(
            128 * (reshape_size // 8) * (reshape_size // 8), output_dim
        )  # Assuming 32x32 input -> 4x4 after convolutions
        print(f"reshape_size: {reshape_size}")
        print(f"self.fc_out: {self.fc_out}")

    def use_attention(self, x):
        bs, c, h, w = x.shape
        x_att = x.reshape(bs, c, h * w).transpose(1, 2)  # BSxHWxC
        x_att = self.norm(x_att)
        att_out, att_map = self.mha(x_att, x_att, x_att)
        return att_out.transpose(1, 2).reshape(bs, c, h, w), att_map

    def forward(self, x):
        print(f"Input dimension: {x.shape}")
        x = self.conv1(x)
        print(f"Dimension after conv1: {x.shape}")
        x = self.scale * self.use_attention(x)[0] + x
        print(f"Dimension after attention: {x.shape}")
        x = F.relu(x)
        print(f"Dimension after relu: {x.shape}")
        x = F.relu(self.bn1(self.conv2(x)))
        print(f"Dimension after conv2: {x.shape}")
        x = F.relu(self.bn2(self.conv3(x)))
        print(f"Dimension after conv3: {x.shape}")
        x = F.relu(self.bn3(self.conv4(x)))
        print(f"Dimension after conv4: {x.shape}")

        features = x.reshape(x.shape[0], -1)  # BS x (128 * 4 * 4)
        print(f"Dimension after flattening: {features.shape}")

        # features_drop = self.dropout(features)
        # output = self.fc_out(features_drop)
        output = self.fc_out(features)
        print(f"Output dimension: {output.shape}")

        return output, features


class ThreeViewCNN(nn.Module):
    def __init__(
        self, channels_ins, output_dims_individual, reshape_size, output_dim=10
    ):
        super(ThreeViewCNN, self).__init__()

        # Three parallel view modules with their own output layers
        self.view1 = SingleViewCNN(
            channels_in=channels_ins[0],
            reshape_size=reshape_size,
            output_dim=output_dims_individual[0],
        )
        self.view2 = SingleViewCNN(
            channels_in=channels_ins[1],
            reshape_size=reshape_size,
            output_dim=output_dims_individual[1],
        )
        self.view3 = SingleViewCNN(
            channels_in=channels_ins[2],
            reshape_size=reshape_size,
            output_dim=output_dims_individual[2],
        )

        # Fusion mechanism using a FC layer
        feature_dim = (
            self.view1.fc_out.in_features
            + self.view2.fc_out.in_features
            + self.view3.fc_out.in_features
        )

        total_feature_dim = feature_dim  # Concatenated features from all views

        self.fusion = nn.Sequential(
            nn.Linear(
                total_feature_dim, 64
            ),  # Learnable fusion of concatenated features
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(64, output_dim),  # Final output
        )

    def forward(self, x1, x2, x3):
        # Get individual predictions and features
        pred1, feat1 = self.view1(x1)  # pred: BS x output_dim, feat: BS x (128*4*4)
        pred2, feat2 = self.view2(x2)
        pred3, feat3 = self.view3(x3)

        # Concatenate features for fusion
        fused_features = torch.cat((feat1, feat2, feat3), dim=1)  # BS x (128*4*4*3)
        print(f"Fused features shape: {fused_features.shape}")

        # Learnable fusion
        fused_pred = self.fusion(fused_features)

        return fused_pred, (feat1, feat2, feat3)

    def get_individual_predictions(self, x1, x2, x3):

        pred1, feat1 = self.view1(x1)
        pred2, feat2 = self.view2(x2)
        pred3, feat3 = self.view3(x3)

        return (pred1, pred2, pred3), (feat1, feat2, feat3)
