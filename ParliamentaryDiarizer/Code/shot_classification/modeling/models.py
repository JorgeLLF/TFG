
import torch.nn as nn

# MODELO DE CLASIFICACIÓN DE PLANOS
class ShotClassificationModel(nn.Module):

    def __init__(self):
        super(ShotClassificationModel, self).__init__()
        self._conv_block_1 = self._normalizing_conv_block(in_channels=3, out_channels=96, kernel_size=3)
        self._conv_block_2 = self._normalizing_conv_block(in_channels=96, out_channels=256, kernel_size=3)
        self._conv_block_3 = self._normalizing_conv_block(in_channels=256, out_channels=384, kernel_size=3)
        self._flatten_stage = nn.Flatten()
        self._linear_block_1 = self._linear_block(in_features=384*26*26, out_features=512)
        self._linear_block_2 = self._linear_block(in_features=512, out_features=512)
        self._classification_layer = nn.Linear(in_features=512, out_features=4)

    def _normalizing_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.LocalResponseNorm(size=2)
        )

    def _linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        x = self._conv_block_1(x)
        x = self._conv_block_2(x)
        x = self._conv_block_3(x)
        x = self._flatten_stage(x)
        x = self._linear_block_1(x)
        x = self._linear_block_2(x)
        x = self._classification_layer(x)
        return x