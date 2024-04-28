import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DualConvolution(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DualConvolution, self).__init__()
        self.dual_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, inputs):
        return self.dual_conv(inputs)

class UNetModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, feature_sizes=[64, 128, 256, 512]):
        super(UNetModel, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for size in feature_sizes:
            self.encoder.append(DualConvolution(input_channels, size))
            input_channels = size
        
        for size in reversed(feature_sizes):
            self.decoder.append(
                nn.ConvTranspose2d(size * 2, size, kernel_size=2, stride=2))
            self.decoder.append(DualConvolution(size * 2, size))
        
        self.core = DualConvolution(feature_sizes[-1], feature_sizes[-1] * 2)
        self.final_layer = nn.Conv2d(feature_sizes[0], output_channels, kernel_size=1)
    
    def forward(self, inputs):
        skips = []
        
        for layer in self.encoder:
            inputs = layer(inputs)
            skips.append(inputs)
            inputs = self.downsample(inputs)
        
        inputs = self.core(inputs)
        skips = skips[::-1]
        
        for index in range(0, len(self.decoder), 2):
            inputs = self.decoder[index](inputs)
            connection = skips[index // 2]
            
            if inputs.shape != connection.shape:
                inputs = TF.resize(inputs, size=connection.shape[2:])
            
            inputs = torch.cat((connection, inputs), dim=1)
            inputs = self.decoder[index + 1](inputs)
        
        return self.final_layer(inputs)

  