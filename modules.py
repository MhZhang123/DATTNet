import os
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler,RandomSampler
from torch.cuda import amp
from PIL import Image
from torch import nn
from torch import optim
import torchvision
from torchvision import transforms
from PIL import Image

#model pre-train
from torch.cuda import amp
from torch.optim import lr_scheduler
import torchvision.models as models
from torchvision.models import vgg16_bn, resnet18, resnet34, resnet50, resnet152, resnet101
from torchvision import models

class SAM(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(SAM,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi, psi



def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

def img2seq(x): # B, C, H, W
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x

def seq2img(x): # B, L, C
    B, L, C = x.shape
    H = W = int(L ** 0.5)
#     print(H, W)
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x



class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out
    

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)
    
    
class MultiScaleConv(nn.Module):
    def __init__(self, cha, patch_size=[1, 3, 5, 7]):
        super().__init__()
        self.msc = nn.ModuleList()
        for ps in patch_size:
            self.msc.append(nn.Conv2d(cha, cha, kernel_size=ps, stride=1, padding=(ps - 1) // 2, bias=False))
        self.proj = nn.Conv2d(cha * 4, cha, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x): # B, C, H, W
        xs = []
        for blk in self.msc:
            conv_x = blk(x)
            xs.append(conv_x)
            
        x = torch.concat(xs, dim=1)
        x = self.proj(x)
        return x
    


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(dim, int(dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim*4)) 
        else:
            self.mlp = MLP_FFN(dim, int(dim*4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx
    



def img2seq(x): # B, C, H, W
    x = x.flatten(2).transpose(1, 2).contiguous()
    return x

def seq2img(x): # B, L, C
    B, L, C = x.shape
    H = W = int(L ** 0.5)
#     print(H, W)
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)
    

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)+x    


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim*2, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim,reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)
            
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out
    
class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio=[1, 2, 4, 8, 16]):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[4], reduction_ratio[4])
        self.sr1 = nn.Conv2d(dim*2, dim*2, reduction_ratio[3], reduction_ratio[3])
        self.sr2 = nn.Conv2d(dim*4, dim*4, reduction_ratio[2], reduction_ratio[2])
        self.sr3 = nn.Conv2d(dim*8, dim*8, reduction_ratio[1], reduction_ratio[1])
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        tem0 = x[:,:50176,:].reshape(B, 224, 224, C).permute(0, 3, 1, 2) 
        tem1 = x[:,50176:75264,:].reshape(B, 112, 112, C*2).permute(0, 3, 1, 2)
        tem2 = x[:,75264:87808,:].reshape(B, 56, 56, C*4).permute(0, 3, 1, 2)
        tem3 = x[:,87808:94080,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        tem4 = x[:,94080:95648,:]

        sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
        sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
        sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)
        sr_3 = self.sr3(tem3).reshape(B, C, -1).permute(0, 2, 1)


        reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, sr_3, tem4], -2))
        
        return reduce_out
    

######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

class AttU_BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratio=[1, 2, 4, 8, 16]):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratio=[1, 2, 4, 8, 16])
        self.norm2 = nn.LayerNorm(dims)
        
        self.mixffn1 = MixFFN_skip(dims,dims*4)
        self.mixffn2 = MixFFN_skip(dims*2,dims*8)
        self.mixffn3 = MixFFN_skip(dims*4,dims*16)
        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        self.mixffn5 = MixFFN_skip(dims*8,dims*32)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4, c5 = inputs
            B, C, _, _= c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f, c5f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)


        tem1 = tx[:,:50176,:].reshape(B, -1, C) 
        tem2 = tx[:,50176:75264,:].reshape(B, -1, C*2)
        tem3 = tx[:,75264:87808,:].reshape(B, -1, C*4)
        tem4 = tx[:,87808:94080,:].reshape(B, -1, C*8)
        tem5 = tx[:,94080:95648,:].reshape(B, -1, C*8)

        m1f = self.mixffn1(tem1, 224, 224).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 112, 112).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 56, 56).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 28, 28).reshape(B, -1, C)
        m5f = self.mixffn4(tem5, 14, 14).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f, m5f], -2)
        
        tx2 = tx1 + t1

        return tx2

class BridgeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratio=[1, 2, 4, 8, 16]):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_4(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_4(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c1, c2, c3, c4, c5 = x
        B, C, _, _= c1.shape
        c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
        c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
        c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)

        res_x = torch.cat([c1f, c2f, c3f, c4f, c5f], -2)        
        
        
        
        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1


        B,_,C = bridge2.shape
        outs = []
        
        sk1 = bridge4[:,:50176,:].reshape(B, 224, 224, C).permute(0, 3, 1, 2) 
        sk2 = bridge4[:,50176:75264,:].reshape(B, 112, 112, C*2).permute(0, 3, 1, 2)
        sk3 = bridge4[:,75264:87808,:].reshape(B, 56, 56, C*4).permute(0, 3, 1, 2)
        sk4 = bridge4[:,87808:94080,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = bridge4[:,94080:95648,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)
        outs.append(sk5)
        
        return outs    
    

#### block4-block5
class AttU_BridgeLayer_5(nn.Module):
    def __init__(self, dims, head, reduction_ratios=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratio=None)
        self.norm2 = nn.LayerNorm(dims)
        

        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        self.mixffn5 = MixFFN_skip(dims*8,dims*32)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c4, c5 = inputs
  
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            inputs = torch.cat([c4f, c5f], -2)

        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem4 = tx[:,:6272,:].reshape(B, -1, C*8)

        tem5 = tx[:,6272:,:].reshape(B, -1, C*8)


        m4f = self.mixffn4(tem4, 28, 28).reshape(B, -1, C)
        m5f = self.mixffn4(tem5, 14, 14).reshape(B, -1, C)


        t1 = torch.cat([m4f, m5f], -2)

        
        tx2 = tx1 + t1

        return tx2

class BridgeBlock_5(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_5(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_5(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1

        B,_,C = bridge2.shape
        outs = []
        
        sk4 = bridge2[:,:6272,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = bridge2[:,6272:,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs    

class BridgeBlock_5_1(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_5(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x

        outs = []
        
        sk4 = bridge1[:,:6272,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = bridge1[:,6272:,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs       

    
class BridgeBlock_5_2(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_5(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_5(dims, head, reduction_ratio)
        self.bridge_layer3 = AttU_BridgeLayer_5(dims, head, reduction_ratio)
        self.bridge_layer4 = AttU_BridgeLayer_5(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1
        bridge3 = self.bridge_layer3(bridge2) + bridge2
        bridge4 = self.bridge_layer4(bridge3) + bridge3
  
        outs = []
        
        sk4 = bridge4[:,:6272,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = bridge4[:,6272:,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs        
    
    
    
class BridgeBlock_5_3(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_5(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_5(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)      

        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)

        B,_,C = bridge2.shape
        outs = []
        
        sk4 = bridge2[:,:6272,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = bridge2[:,6272:,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs        
######################################################################################################
######################################################################################################
######################################################################################################
###block5-block6

class AttU_BridgeLayer_6(nn.Module):
    def __init__(self, dims, head, reduction_ratios=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratio=None)
        self.norm2 = nn.LayerNorm(dims)
        

        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        self.mixffn5 = MixFFN_skip(dims*8,dims*32)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c4, c5 = inputs
  
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c4f, c5f], -2)
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem4 = tx[:,:1568,:].reshape(B, -1, C*8)
        tem5 = tx[:,1568:,:].reshape(B, -1, C*8)

        m4f = self.mixffn4(tem4, 14, 14).reshape(B, -1, C)
        m5f = self.mixffn4(tem5, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m4f, m5f], -2)
        
        tx2 = tx1 + t1

        return tx2

class BridgeBlock_6(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_6(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_6(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1

        B,_,C = bridge2.shape
        outs = []
        
        sk4 = bridge2[:,:1568,:].reshape(B, 14, 14, C*8).permute(0, 3, 1, 2)
        sk5 = bridge2[:,1568:,:].reshape(B, 7, 7, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs 
    
class AttU_BridgeLayer_7(nn.Module):
    def __init__(self, dims, head, reduction_ratios=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)

        self.mixffn4 = MixFFN_skip(dims*2,dims*8)
        self.mixffn5 = MixFFN_skip(dims*4,dims*16)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c4, c5 = inputs
  
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 784*64
            
            inputs = torch.cat([c4f, c5f], -2)  # 2352*64
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem4 = tx[:,:1568,:].reshape(B, -1, C*2)
        tem5 = tx[:,1568:,:].reshape(B, -1, C*4)

        m4f = self.mixffn4(tem4, 28, 28).reshape(B, -1, C)  # 
        m5f = self.mixffn5(tem5, 14, 14).reshape(B, -1, C)  # 

        t1 = torch.cat([m4f, m5f], -2)
        
        tx2 = tx1 + t1

        return tx2
    
    
class BridgeBlock_7(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_7(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_7(dims, head, reduction_ratio)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1

        B,_,C = bridge2.shape
        outs = []
        
        sk4 = bridge2[:,:1568,:].reshape(B, 28, 28, C*2).permute(0, 3, 1, 2)
        sk5 = bridge2[:,1568:,:].reshape(B, 14, 14, C*4).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs       

class AttU_BridgeLayer_8(nn.Module):
    def __init__(self, dims, head, reduction_ratios=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)

        self.mixffn4 = MixFFN_skip(dims*8,dims*32)
        self.mixffn5 = MixFFN_skip(dims*16,dims*64)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            c4, c5 = inputs
  
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 6272*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c4f, c5f], -2)  # 9408*64
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem4 = tx[:,:6272,:].reshape(B, -1, C*8)
        tem5 = tx[:,6272:,:].reshape(B, -1, C*16)

        m4f = self.mixffn4(tem4, 28, 28).reshape(B, -1, C)  # 
        m5f = self.mixffn5(tem5, 14, 14).reshape(B, -1, C)  # 
        t1 = torch.cat([m4f, m5f], -2)
        tx2 = tx1 + t1

        return tx2
    
    
class BridgeBlock_8(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_8(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_8(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1

        B,_,C = bridge2.shape
        outs = []
        
        sk4 = bridge2[:,:6272,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)
        sk5 = bridge2[:,6272:,:].reshape(B, 14, 14, C*16).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs   
    
#### block3-block4
class AttU_BridgeLayer_9(nn.Module):
    def __init__(self, dims, head, reduction_ratios=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)

        self.mixffn4 = MixFFN_skip(dims*4,dims*16)
        self.mixffn5 = MixFFN_skip(dims*8,dims*32)
        
    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c4, c5 = inputs
  
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 12544*64
            c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)  # 6272*64
            
            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c4f, c5f], -2)  # 18816*64
        else:
            B,_,C = inputs.shape 

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem4 = tx[:,:12544,:].reshape(B, -1, C*4)
        tem5 = tx[:,12544:,:].reshape(B, -1, C*8)

        m4f = self.mixffn4(tem4, 56, 56).reshape(B, -1, C)  # 
        m5f = self.mixffn5(tem5, 28, 28).reshape(B, -1, C)  # 

        t1 = torch.cat([m4f, m5f], -2)
        tx2 = tx1 + t1

        return tx2


class BridgeBlock_9(nn.Module):
    def __init__(self, dims, head, reduction_ratio=None):
        super().__init__()
        self.bridge_layer1 = AttU_BridgeLayer_9(dims, head, reduction_ratio)
        self.bridge_layer2 = AttU_BridgeLayer_9(dims, head, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x[0].shape[0]
        C = 64
        c4, c5 = x

        c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64
        c5f = c5.permute(0, 2, 3, 1).reshape(B, -1, C)
        # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
        res_x = torch.cat([c4f, c5f], -2)        

        bridge1 = self.bridge_layer1(x) + res_x
        bridge2 = self.bridge_layer2(bridge1) + bridge1

        B,_,C = bridge2.shape
        outs = []
        
        sk4 = bridge2[:,:12544,:].reshape(B, 56, 56, C*4).permute(0, 3, 1, 2)
        sk5 = bridge2[:,12544:,:].reshape(B, 28, 28, C*8).permute(0, 3, 1, 2)

        outs.append(sk4)
        outs.append(sk5)
        
        return outs       


class DATTNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=3):
        super().__init__()
        self.encoder = vgg16_bn(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.eca1 = eca_layer(64)
        self.Att1 = SAM(F_g=64,F_l=64,F_int=32)
        
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.eca2 = eca_layer(128)
        self.Att2 = SAM(F_g=128,F_l=128,F_int=64)
        
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.eca3 = eca_layer(256)
        self.Att3 = SAM(F_g=256,F_l=256,F_int=128)
        
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.eca4 = eca_layer(512)
        self.Att4 = SAM(F_g=512,F_l=512,F_int=256)

        self.block5 = nn.Sequential(*self.encoder[27:34])
        self.eca5 = eca_layer(512)
        self.Att5 = SAM(F_g=512,F_l=512,F_int=256)


        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.ecab = eca_layer(512)
        self.Attb = SAM(F_g=512,F_l=512,F_int=256)        
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.Att6 = SAM(F_g=512,F_l=512,F_int=256)
        self.conv6 = conv(512 + 512, 512)

        self.up_conv7 = up_conv(512, 256)
        self.Att7 = SAM(F_g=256,F_l=256,F_int=128)
        self.conv7 = conv(256 + 512, 256)

        self.up_conv8 = up_conv(256, 128)
        self.Att8 = SAM(F_g=128,F_l=128,F_int=64)
        self.conv8 = conv(128 + 256, 128)

        self.up_conv9 = up_conv(128, 64)
        self.Att9 = SAM(F_g=64,F_l=64,F_int=32)
        self.conv9 = conv(64 + 128, 64)

        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

        self.bridge = BridgeBlock_5(dims=64, head=1)
        

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        block1 = self.block1(x)
        block1 = self.eca1(block1)
        block1, p1 = self.Att1(g=block1, x=block1)
        
        block2 = self.block2(block1)
        block2 = self.eca2(block2)
        block2, p2 = self.Att2(g=block2, x=block2)
        
        block3 = self.block3(block2)
        block3 = self.eca3(block3)
        block3, p3 = self.Att3(g=block3, x=block3)
        
        block4_i = self.block4(block3)
        block4 = self.eca4(block4_i)
        block4, p4 = self.Att4(g=block4, x=block4)

        block5_i = self.block5(block4)
        block5 = self.eca5(block5_i)
        block5, p5 = self.Att5(g=block5, x=block5)        

        bottleneck = self.bottleneck(block5)
        bottleneck = self.ecab(bottleneck)
        bottleneck, pb = self.Attb(g=bottleneck, x=bottleneck)        
        x = self.conv_bottleneck(bottleneck)
        
        block4, block5 = self.bridge([block4_i, block5_i])

        x = self.up_conv6(x)
        x, u6 = self.Att6(g=x,x=x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x, u7  = self.Att7(g=x,x=x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x, u8  = self.Att8(g=x,x=x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x, u9  = self.Att9(g=x,x=x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)
        x = self.conv11(x)
        
        return x, [p1, p2, p3, p4, p5, pb, u6, u7, u8, u9]

