from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torchsummary import summary


model_urls = {}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cardinality=4, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # print('######')
        # print('width:',width)
        # print('######')
        self.width = width
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes)
            )
        self.stride = stride
        self.cardinality = cardinality
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.ag = nn.Sequential(
            nn.Linear(width,width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width,width),
            nn.BatchNorm1d(width),
            nn.ReLU()
        )
        self.branches = nn.ModuleList()
        for i in range(1,self.cardinality+1):
            branch = nn.Sequential()
            for j in range(i):
                scope = 'lite3x3_b'+str(i)+'_'+str(j+1)
                branch.add_module(scope,self.lite3x3(self.width,self.width))
            self.branches.append(branch)
    
    def lite3x3(self, inplanes, planes, stride=1):
        return nn.Sequential(
            conv1x1(inplanes,inplanes), #pointwise 1x1
            conv3x3(inplanes, planes, groups=inplanes), #depthwise 3x3
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
    
    def _gate(self,in_tensor,branches,branch_idx):
        out = branches[branch_idx](in_tensor)
        out_gap = self.gap(out)
        out_gap = out_gap.view(out_gap.size(0),-1)
        out_ag = self.ag(out_gap)
        # print('out_ag:',out_ag.shape)
        # print('out:',out.shape)
        out = (out.reshape(out.size(0)*out.size(1),out.size(2),out.size(3)) \
                .transpose(0,2)*(out_ag.flatten().float())) \
                .transpose(2,0) \
                .reshape(out.size())
        # out = (out.transpose(0,2)*out_ag).transpose(0,2)
        return out

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out1 = self.branches[0](out) #out shape: 64,64,64
        # out1_gap = self.gap(out1)
        # out1_gap = out1_gap.view(out1_gap.size(0),-1)
        # out1_ag = self.ag(out1_gap)
        # out1 = (out1.transpose(0,2)*out1_ag).transpose(0,2)
  
        # print('out:',out.shape)
        out1 = self._gate(out,self.branches,0)
        out2 = self._gate(out,self.branches,1)
        out3 = self._gate(out,self.branches,2)
        out4 = self._gate(out,self.branches,3)

        out = out1+out2+out3+out4
        # print('summed_out:',out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class OSNet(nn.Module):

    def __init__(self, num_classes, loss, block=Bottleneck, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_stride=2, fc_dims=None, dropout_p=None, **kwargs):
        super(OSNet, self).__init__()
        print('###num_classes:',num_classes)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes),
            self.relu,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            block(self.inplanes, self.inplanes),
            block(self.inplanes, 256),
        )

        self.transition1 = nn.Sequential(
            conv1x1(256, 256),
            norm_layer(256),
            self.relu,
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.conv3 = nn.Sequential(
            block(256, 256),
            block(256,384),
        )

        self.transition2 = nn.Sequential(
            conv1x1(384,384),
            norm_layer(384),
            self.relu,
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.conv4 = nn.Sequential(
            block(384,384),
            block(384,512),
        )

        self.conv5 = conv1x1(512,512)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            self.relu
        )
        self.classifier = nn.Linear(512,num_classes)
        

    def forward(self, x):

        out = self.conv1(x)
        # print('conv1:',out.size())
        out = self.conv2(out)
        # print('conv2:',out.size())
        out = self.transition1(out)
        # print('transition1:',out.size())
        out = self.conv3(out)
        # print('conv3:',out.size())
        out = self.transition2(out)
        # print('transition2:',out.size())
        out = self.conv4(out)
        # print('conv4:',out.size())
        out = self.conv5(out)
        # print('conv5:',out.size())
        out = self.gap(out)
        out = out.view(out.size(0),-1)
        # print('gap:',out.size())
        out = self.fc(out)

        if not self.training:
            return out
        
        out = self.classifier(out)

        return out
        
        # if self.loss == 'softmax':
        #     return y
        # elif self.loss == 'triplet':
        #     return y, v
        # else:
        #     raise KeyError("Unsupported loss: {}".format(self.loss))

def osnet(num_classes, loss='softmax', pretrained=False, **kwargs):
    model = OSNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    summary(model,input_size=(3,256,256))
    return model
