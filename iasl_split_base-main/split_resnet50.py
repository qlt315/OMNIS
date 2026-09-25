import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from boxcoder import boxcoder

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1

"""
Making custom resnet backbone for FasterRCNN object detection model
Steps:
    -- remove layer0
    -- remove first few blocks in layer1
    -- add bottleneck in layer1
    -- remove avgpool, fc layers at end (bc we only want the feature map, not a classification)

Training
1) make teacher resnet (train on imagenet dataset), then student is phsyically split resnet
    -- phys split: train head + encoder/decoder, freeze tail
    -- phys split: reverse the freeze, only train tail
    -- now we have fully trained reg resnet50, and phys split resnet50
2) now, train 2 models
    -- time for obj det model, don't train backbone (freeze it), only obj det heads
    -- model 1: reg obj det
    -- model 2: obj det model + phys split backbone
3) add early exit
    -- determine whether we should go thru with the split (is our pred (bb + class) good enough yet?)
    -- perform another round of knowledge distillation, train up the early exit
"""
class ResidualBlock50(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock50, self).__init__()
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
                        torch.nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        torch.nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = torch.nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        # x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        residual = x
        out = self.conv1(x)
        # print(out)
        out = self.conv2(out)
        # print(out)
        out = self.conv3(out)
        # print(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        # print(out)
        return out

###################################################################

class encoder(nn.Module):
    def __init__(self, bottleneck_channel=12, is_training=True, is_testing=False): # we will manually set is_testing
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn1 = GDN1(32)
        # self.conv2 = nn.Conv2d(48, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # self.gdn2 = GDN1(32)

        #IMPORTANT:: BE SURE TO CHANGE CHANNELS FOR NON ENTROPY BOTTLENECKS (and for OD 12 channel encoder)
        # normal = 16
        # entropy = 24 for 24channel bottleneck(may have higher values)
        channel_value = 16

        self.conv2 = nn.Conv2d(32, channel_value, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn2 = GDN1(channel_value)
        self.conv3 = nn.Conv2d(channel_value, bottleneck_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.entropy_bb = EntropyBottleneck(bottleneck_channel)

        self.is_training = is_training
        self.is_testing = is_testing
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        # x = self.gdn3(x)
        # x = self.conv4(x)

        if self.is_testing:
            size = [x.shape[2], x.shape[3]]
            y_hat = self.entropy_bb.compress(x)
            y_likelihoods = size
        else:   
            y_hat, y_likelihoods = self.entropy_bb(x, self.is_training)

        # print(y_hat.shape)

        return y_hat, y_likelihoods
    
class standard_encoder(nn.Module):
    def __init__(self, bottleneck_channel=12, is_training=True):
        super(standard_encoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, bottleneck_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channel)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x
    
class new_standard_encoder(nn.Module):
    def __init__(self, bottleneck_channel=12, is_training=True):
        super(new_standard_encoder, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, bottleneck_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(bottleneck_channel)

        self.quant_thing = boxcoder(bottleneck_channel)
        self.to_bits = False
        self.packet_size_value = 128
        self.drop_prob = 0.25899
        self.ber = (10**(-6))

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        self.self_target = x

        if self.to_bits:
            x, x_min, x_max = self.quant_thing.forward_quant(x)
            x = self.quant_thing.encode_to_bitstream(x, x_min, x_max, (self.packet_size_value+16))
            x = self.quant_thing.add_error(x, self.ber)#1, 95)
            x, x_min, x_max = self.quant_thing.decode_to_tensor(x, (self.packet_size_value+16), self.drop_prob)
            x = self.quant_thing.forward_dequant(x, x_min, x_max)
        else:
            x = self.quant_thing(x)
        x = self.relu(x)

        return x
    
###################################################################

class decoder(nn.Module):
    def __init__(self, bottleneck_channel=12, is_testing=None): # we will manually set is_testing
        super(decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(bottleneck_channel, 512, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.gdn1 = GDN1(512, inverse=True)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.gdn2 = GDN1(256, inverse=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.gdn3 = GDN1(256, inverse=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gdn3 = GDN1(256, inverse=True)
        # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False)

        self.entropy_bb_for_decode = EntropyBottleneck(bottleneck_channel)
        self.is_testing = is_testing
        self.apply_grad = False

    def forward(self, x, size=None):
        if self.is_testing:
            x = self.entropy_bb_for_decode.decompress(x, size)

        if self.apply_grad:
            x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
            # x.requires_grad = True

        # x = F.relu(x)
        x = self.conv1(x)
        # if self.apply_grad:
        #     x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        x = self.gdn1(x)
        # if self.apply_grad:
        #     x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        x = self.conv2(x)
        # if self.apply_grad:
        #     x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        x = self.gdn2(x)
        # if self.apply_grad:
        #     xx = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        x = self.conv3(x)
        # if self.apply_grad:
        #     x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        x = self.gdn3(x)
        # if self.apply_grad:
        #     xx = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)
        x = self.conv4(x)
        if self.apply_grad:
            x = torch.nan_to_num(x, nan=0.0001, posinf=0.0001, neginf=0.0001)

        # print(torch.isnan(x).any())

        # print(x.shape)
        
        return x

class standard_decoder(nn.Module):
    def __init__(self, bottleneck_channel=12):
        super(standard_decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(bottleneck_channel, 512, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, size=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        return x

###################################################################

class ResNetHead(torch.nn.Module):
    def __init__(self, bottelneck_channel= 12, is_training=True, entropy_split=True, new_standard=False):
        super(ResNetHead, self).__init__()
        self.inplanes = 64
        self.conv1 = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        torch.nn.BatchNorm2d(64),
                        torch.nn.ReLU())
        self.mp = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.entropy_split = entropy_split and not new_standard
        if entropy_split:
            self.encoder = encoder(bottelneck_channel, is_training)
        elif new_standard:
            self.encoder = new_standard_encoder(bottelneck_channel)
        else:
            self.encoder = standard_encoder(bottelneck_channel)

    def forward(self, x):
        self.conv1_out = self.conv1(x)
        self.mp_out = self.mp(self.conv1_out)

        if self.entropy_split:
            self.y_hat, self.y_likelihoods = self.encoder(self.mp_out)                 
            return self.y_hat, self.y_likelihoods
        else:
            return self.encoder(self.mp_out)

###################################################################

class ResNetTail(torch.nn.Module):
    def __init__(self, block, layers, num_classes, bottelneck_channel=12, keep_heads=False, entropy_split=True):
        super(ResNetTail, self).__init__()
        self.inplanes = 64
        if entropy_split:
            self.decoder = decoder(bottelneck_channel)
        else:
            self.decoder = standard_decoder(bottelneck_channel)

        self.use_gt = False
        self.gt_decoder = None

        self.keep_heads = keep_heads
        self.layer2  = self._make_layer(block, 512, layers[2], stride = 2)
        self.layer3  = self._make_layer(block, 1024, layers[3], stride = 2)
        self.layer4  = self._make_layer(block, 2048, layers[2], stride = 2)

        if keep_heads:
            self.avgpool = nn.AvgPool2d(7, stride=1)
            self.fc      = nn.Linear(512, num_classes)

        # for computing loss in 1st round of training
        self.decoder_out = None
        self.l2_out      = None
        self.l3_out      = None
        self.l4_out      = None

        # self.bit_process = True
        # if self.bit_process:
        #     self.bit_processor =  nn.Sequential(block(256,256,1,None),block(256,256,1,None))

    def forward(self, x, size=None):
        if self.use_gt:
            self.decoder_out = self.gt_decoder(x, size)
        else:
            self.decoder_out = self.decoder(x, size)
        # if self.bit_process:
        #     self.decoder_out = self.bit_processor(self.decoder_out)
        # print(torch.isnan(self.decoder_out).any())
        # print('----')
        # print(self.decoder_out.shape)
        self.l2_out = self.layer2(self.decoder_out)
        # print(self.l2_out)
        self.l3_out = self.layer3(self.l2_out)
        # print(self.l3_out)
        self.l4_out = self.layer4(self.l3_out)
        x = self.l4_out
        # print(self.l4_out)

        if self.keep_heads:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)
    

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 224, 224)
    head = ResNetHead()
    tail = ResNetTail(ResidualBlock50, [3, 4, 6, 3], 1e3)
    h_out = head(input_tensor)
    t_out = tail(h_out)
    print("Backbone OUT", t_out.shape)