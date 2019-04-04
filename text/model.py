import torch
from torch import nn
import torch.nn.functional as F

import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_labels, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.last_conv = nn.Conv2d(512, num_labels, kernel_size=(2, 3), padding=(0, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last_conv(x).squeeze(2)
        # bs, number of classes, length

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), False)
    return model


def build_model(num_labels):
    model = resnet18(num_labels=num_labels)
    return model

def loss_calculation(predicted, targets, target_lengths):
    #, dtype=torch.int32
    logit = F.log_softmax(predicted, dim=1).permute(2, 0, 1).contiguous()
    input_lengths = torch.full((logit.size(1),), logit.size(0), dtype=torch.int).to('cuda')
    # print(logit[0])
    # print(input_lengths.size())
    # print(targets.size())
    # print(torch.sum(target_lengths))
    
    #print(logit)
    loss = F.ctc_loss(logit, targets, input_lengths, target_lengths, zero_infinity=True)
    # print(loss)
    # return

    return loss

def get_accuracy(predicted, targets, ind_to_class, to_file=False):
    #bs, classes, length
    #list of target tensors
    correct = 0
    strings = []
    probability = torch.argmax(F.log_softmax(predicted, dim=1), dim=1)
    for batch in probability:
        tmp = []
        for index in batch:
            char = ind_to_class[index.item()]
            if len(tmp) == 0 or tmp[-1] == '<_>' or tmp[-1] != char:
                tmp.append(char)
        strings.append(''.join(tmp).replace('<_>', ''))
    targets = [tensor_to_str(tensor, ind_to_class) for tensor in targets]
    for target_idx, target in enumerate(targets):
        for idx, char in enumerate(target):
            try:
                if strings[target_idx][idx] == char:
                    # print('add')
                    correct += 1
            except:
                pass
    if to_file:
        with open('./results.txt', 'a') as f:
            for string, target in zip(strings, targets):
                f.write('{} : {}\n'.format(string, target))
    # print(strings[0], ' : ', targets[0])
    # print(correct)
    return correct

def tensor_to_str(tensor, ind_to_class):
    tmp = []
    for idx in  tensor:
        tmp.append(ind_to_class[idx.item()])

    return ''.join(tmp)