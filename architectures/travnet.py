import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class TravNetUp3NNRGB(nn.Module):
    def __init__(self, pretrained=True, output_size=(240, 424), bottleneck_dim=256, output_channels=1):
        super().__init__()
        model = models.resnet18(pretrained=pretrained)
        self.out_dim = (output_size[0], output_size[1])
        self.num_classes = output_channels
        
        # encoder
        self.block1 = nn.Sequential(*(list(model.children())[:3]))
        self.block2 = nn.Sequential(model.maxpool, model.layer1)
        self.block3 = model.layer2
        self.block4 = model.layer3
        self.block5 = model.layer4
        
        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bottleneck_dim, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.convUp1A = nn.Sequential(
            nn.Conv2d(in_channels=512+256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.convUp1B = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.convUp2A = nn.Sequential(
            nn.Conv2d(in_channels=256+128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.convUp2B = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.convUp3A = nn.Sequential(
            nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.convUp3B = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.convUp4A = nn.Sequential(
            nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.convUp4B = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.convUp5A = nn.Sequential(
            nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.convUp5B = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(in_channels=32, out_channels=output_channels, kernel_size=1, stride=1)
        self.activation = nn.Sigmoid()
        
    def get_name(self):
        return "travnetup3nnrgb"

    def forward(self, x):
        # Encoder
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        # Bottleneck
        x = self.bottleneck(out5)

        # Decoder
        x = torch.cat((x, out5), dim=1)
        x = self.convUp1B(F.interpolate(self.convUp1A(x), out4.shape[2:]))
        x = torch.cat((x, out4), dim=1)
        x = self.convUp2B(F.interpolate(self.convUp2A(x), out3.shape[2:]))
        x = torch.cat((x, out3), dim=1)
        x = self.convUp3B(F.interpolate(self.convUp3A(x), out2.shape[2:]))
        x = torch.cat((x, out2), dim=1)
        x = self.convUp4B(F.interpolate(self.convUp4A(x), out1.shape[2:]))
        x = torch.cat((x, out1), dim=1)
        x = self.convUp5B(F.interpolate(self.convUp5A(x), self.out_dim))
        output = self.activation(self.final(x))
        
        if self.num_classes == 1:
            return {'prediction': output[:,0]}
        else:
            return {'prediction': output}
    
    def load_pretrained(self, weights):
        # Load from pretrained (but ignore "final" layer for semantic segmentation pretraining)
        state_dict = self.state_dict()
        state_dict.update({k: v for k, v in weights.items() if not k.startswith('final')})
        self.load_state_dict(state_dict)
