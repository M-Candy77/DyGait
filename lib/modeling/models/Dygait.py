import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks



def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(1,3,3),
                     stride=stride,
                     padding=(0,1,1),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
        
    


class BasicBlockDy(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1g = conv3x3x3(in_planes, planes, stride)
        self.conv1d = conv3x3x3(in_planes, planes, stride)
        self.conv1s = conv1x3x3(in_planes, planes, stride)

        self.relu = nn.ReLU(inplace=True)

        self.conv2g = conv3x3x3(planes, planes)
        self.conv2d = conv3x3x3(planes, planes)
        self.conv2s = conv3x3x3(planes, planes)


        self.stride = stride

    def forward(self, x):
        residual = x

        out1g = self.conv1g(x)
        out1g = self.relu(out1g)

        out1mean = torch.mean(x, 2).unsqueeze(2)
        out1d = x - out1mean
        out1d = self.conv1d(out1d)
        out1s = self.conv1s(x)
        out1ds = out1d + out1s
        out1ds = self.relu(out1ds)        
        out1 = out1g + out1ds

        out2g = self.conv2g(out1)
        out2g = self.relu(out2g)

        out2mean = torch.mean(out1, 2).unsqueeze(2)
        out2d = out1 - out2mean
        out2d = self.conv2d(out2d)
        out2s = self.conv2s(out1)
        out2ds = out2d + out2s
        out2ds = self.relu(out2ds) 
        
        out = out2g + out2ds
        out = out+residual

        return out


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class Dygait(BaseModel):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, *args, **kargs):
        super(Dygait, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']

        if dataset_name in ['OUMVLP','GREW','Gait3D-128pixel']:
            # For OUMVLP and GREW
            self.conv3d0 = BasicBlockDy(1, in_c[0])
            
            self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv3d1 = BasicBlockDy(in_c[0], in_c[1])
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[1], in_c[1], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.conv3d2 = BasicBlockDy(in_c[1], in_c[2])
            self.MaxPool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.conv3d3 = BasicBlockDy(in_c[2], in_c[3])

            self.conv3d4 = BasicBlockDy(in_c[3], in_c[3])
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = BasicBlockDy(in_c[0], in_c[1])
            self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = BasicBlockDy(in_c[1], in_c[2])
            self.GLConvB2 = BasicBlockDy(in_c[2], in_c[2])

        if dataset_name in ['OUMVLP','GREW']:
            self.Head0 = SeparateFCs(31, in_c[-1], in_c[-1])
            self.HPP = GeMHPP(bin_num=[16,8,4,2,1])
        else:
            self.Head0 = SeparateFCs(32, in_c[-1], in_c[-1])
            self.HPP = GeMHPP(bin_num=[32])            
        self.TP = PackSequenceWrapper(torch.max)
        # self.HPP = GeMHPP()
        
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        #print("---sils-",sils.shape)
        outs = self.conv3d0(sils)
        
        # outs = self.LTA(outs)
        outs = self.MaxPool0(outs)

        outs = self.conv3d1(outs)
        # outs = self.MaxPool0(outs)
        outs = self.LTA(outs)


        outs = self.conv3d2(outs)
        outs = self.MaxPool1(outs)
        outs = self.conv3d3(outs)

        outs = self.conv3d4(outs)  # [n, c, s, h, w]
        # print('-conv3d3-',outs.shape)
        outs = self.TP(outs, dim=2, seq_dim=2, seqL=seqL)[0]  # [n, c, h, w]
        #print('-TP前-', outs.shape)
        #outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        #print('-TP后-', outs.shape)
        outs = self.HPP(outs)  # [n, c, p]
        outs = outs.permute(2, 0, 1).contiguous()  # [p, n, c]
        # print('-outs-',outs.shape)
        embed_1 = self.Head0(outs)  # [p, n, c]
        # print('-Head0-',outs.shape)
        # gait = gait.permute(1, 2, 0).contiguous()  # [n, c, p]
        # gait = gait.permute(0, 2, 1).contiguous()  # [n, p, c]



        # feat = self.HPP(outs)  # [n, c, p]
        # feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]

        # embed_1 = self.FCs(feat)  # [p, n, c]
        embed_2, logits = self.BNNecks(embed_1)  # [p, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed_2 = embed_2.permute(1, 0, 2).contiguous()  # [n, p, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p, c]
        embed = embed_1


        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
