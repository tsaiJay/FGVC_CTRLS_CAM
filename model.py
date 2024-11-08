import torch
import torch.nn as nn
import torchvision.models as models
from cm_module import camModule


def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        # init.constant_(m.bias.data, 0.0)


class resnet(nn.Module):
    def __init__(self, model, num_classes, cm_args=None):
        super().__init__()

        self.cm_args = cm_args

        if model == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif model == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

        # if init:
        #     self.model.fc.apply(weight_init_kaiming)

        if self.cm_args:
            self.cm_module = camModule(cm_args=cm_args, n_features=n_features)

    def forward(self, x, label):
        # out = self.model(x)  # same forward as below

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        logit = self.model.layer4(x)

        if self.cm_args:
            hint_map = self.cm_module(logit, self.model.fc.weight, label)

        out = self.model.avgpool(logit)
        out = out.flatten(1)
        out = self.model.fc(out)

        return out, hint_map


# class resnet18(nn.Module):
#     def __init__(self, num_classes, init: bool = False):
#         super().__init__()

#         self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

#         n_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(n_features, num_classes)

#         if init:
#             self.model.fc.apply(weight_init_kaiming)

#     def forward(self, x):
#         # out = self.model(x)  # same forward as below

#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         logit = self.model.layer4(x)

#         out = self.model.avgpool(logit)
#         out = out.flatten(1)
#         out = self.model.fc(out)

#         return out, logit