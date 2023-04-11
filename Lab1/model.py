import torch.nn as nn


class FirstConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FirstConvLayer, self).__init__()
        self.sequential = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size))
    
    def forward(self, x):
        return self.sequential(x)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, want_shortcut, downsample, last_layer, pool_type):
        super(ConvolutionalBlock, self).__init__()

        self.want_shortcut = want_shortcut
        if self.want_shortcut:
            self.shortcut == nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.sequential = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLu(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLu()
        )

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if downsample:
            if last_layer:
                self.want_shortcut = False
                self.sequential.append(nn.AdaptiveMaxPool2d(2))
            else:
                if pool_type == 'convolution':
                    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, ..., padding=1, bias=False)
                elif pool_type == 'kmax':
                    channels = [64, 128, 256, 512]
                    dimension = [511, 256, 128]
                    index = channels.index(in_channels)
                    self.sequential.append(nn.AdaptiveMaxPool2d(dimension[index]))
                else:
                    self.sequential.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                
        self.relu = nn.ReLu()

    def forward(self, x):
        if self.want_shortcut:
            short = x
            out = self.conv1(x)
            out = self.sequential(out)
            if out.shape != short.shape:
                short = self.shortcut(short)
            out = self.relu(short + out)
            return out
        else:
            out = self.conv1(x)
            return self.sequential(out)
    

class FullyConnectedBlock(nn.Model):
    def __init__(self, n_class):
        super(FullyConnectedBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(2048, 1024)
            nn.ReLu(),
            nn.Linear(1024, 1024),
            nn.ReLu(),
            nn.Linear(1024, n_class),
            # nn.Softmax(dim=1)
        )


    def forward(self, x):
        return self.sequential(x)















