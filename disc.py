import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    CNN Block that encapsulates convolution, batch normalization, and LeakyReLU activation.

    Parameters:
    - in_channels (int): Number of channels in the input image
    - out_channels (int): Number of channels produced by the convolution
    - stride (int, optional): Stride of the convolution.
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Forward pass of the CNNBlock.

        Parameters:
        - x (torch.Tensor): Input tensor

        Returns:
        - torch.Tensor: Output tensor
        """
        return self.conv(x)


class Discriminator(nn.Module):
    """
    The Discriminator network for DroGAN, utilizing CNNBlocks for the internal layers.

    Parameters:
    - in_channels (int): Number of channels in the input images
    """
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,  
                features[0],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature))
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Forward pass of the Discriminator.

        Parameters:
        - x (torch.Tensor): Input tensor from domain X
        - y (torch.Tensor): Input tensor from domain Y

        Returns:
        - torch.Tensor: Output tensor representing the discriminator's prediction
        """
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
