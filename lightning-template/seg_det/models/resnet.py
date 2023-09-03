import torch
import torchvision.models as models
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, layer, pretrained):
        super(ResNet, self).__init__()
        resnet_models = {18 : models.resnet18,
                         34 : models.resnet34,
                         50 : models.resnet50}
        self.resnet = resnet_models[layer](pretrained)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x