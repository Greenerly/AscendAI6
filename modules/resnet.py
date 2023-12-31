import mindspore.nn as nn# import torch.nn as nn
import math
import mindspore.dataset
# import torch.utils.model_zoo as model_zoo
import mindspore.ops as F# import torch.nn.functional as F

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size =1,stride =stride,has_bias=True)
# def conv1x1(in_planes,out_planes,stride=1):
#     return nn.Conv2d(in_planes,out_planes,kernel_size =1,stride =stride,bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, has_bias=True)
# def conv3x3(in_planes, out_planes, stride=1):
#     "3x3 convolution with padding"
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)



class BasicBlock(nn.Cell):# class BasicBlock(nn.Module)
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        # 与pytorch关系为momentum=1-momentum
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)# batch normalization使得一批(Batch)的feature map满足均值为0，方差为1的分布规律
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    # 貌似不用改
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 到这里
class ResNet(nn.Cell):# class ResNet(nn.Module)

    def __init__(self, block, layers, strides, compress_layer=True):
        self.inplanes = 32
        super(ResNet, self).__init__()
        # padding=1会有如下报错：For 'Conv2D', the 'pad' must be zero when 'pad_mode' is not 'pad', but got 'pad': 1 and 'pad_mode': same.
        self.conv1_new = nn.Conv2d(3, 32, kernel_size=3, stride=strides[0], padding=0,
                               has_bias=True)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.9)# 也可以不写
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 32, layers[0],stride=strides[1])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4])        
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5])

        self.compress_layer = compress_layer        
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1), has_bias =True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace = True))

        # for m in self.modules():
        for m in self.name_cells():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.SequentialCell(*layers)

    def forward(self, x, multiscale = False):
        out_features = []
        x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_shape = x.size()[2:]
        x = self.layer1(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer2(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer3(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer4(x)
        if x.size()[2:] != tmp_shape:
            tmp_shape = x.size()[2:]
            out_features.append(x)
        x = self.layer5(x)
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.size()[2:] != tmp_shape:
                tmp_shape = x.size()[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)
        return out_features

def resnet45(strides, compress_layer):
    model = ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer)
    return model
