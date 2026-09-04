import torch
from torch import nn
from .models import register


@register('guidenet')
class GuideNetwork(nn.Module):
    def __init__(self, model_type, in_dim=None, hidden_dim=None, out_dim=None,
                 kernel_size=None, stride_lst=None, padding_lst=None, pool=None, dilation_lst=None):
        super(GuideNetwork, self).__init__()

        self.model_type = model_type
        if model_type == 'identity':
            self.network = nn.Identity()
        else:
            self.network = []
            if model_type == 'fc':
                if hidden_dim != None:
                    prev_dim = in_dim
                    for dim in hidden_dim:
                        self.network.extend([nn.Linear(prev_dim, dim),
                                             nn.ReLU()])
                        prev_dim = dim
                    self.network.append(nn.Linear(prev_dim, out_dim))
                else:
                    self.network.append(nn.Linear(in_dim, out_dim))
            elif model_type == 'conv':
                if hidden_dim != None:
                    prev_dim = in_dim
                    for dim, kernel, stride, pad, dil in zip(hidden_dim, kernel_size[:-1], stride_lst[:-1], padding_lst[:-1], dilation_lst[:-1]):
                        self.network.extend([nn.Conv2d(prev_dim, dim, kernel, stride, padding=pad, dilation=dil),
                                            nn.InstanceNorm2d(dim), # nn.BatchNorm2d(dim) : oe, ue 마다 exposure 값이 다르고 그에 따라 신뢰도가 다름. 정규화는 반드시 진행하되, 노출/대비를 ref 의 스타일로 인식하고 batch norm 대신 instance norm 을 우선 적용
                                            nn.ReLU()])
                        prev_dim = dim
                    self.network.append(nn.Conv2d(prev_dim, out_dim, kernel_size[-1], stride_lst[-1], padding_lst[-1], dilation_lst[-1]))
                    if pool == 'True':
                        self.network.append(nn.AdaptiveAvgPool2d(1))
                else:
                    self.network.append(nn.Conv2d(in_dim, out_dim, 3))
            self.network = nn.Sequential(*self.network)

    def forward(self, x):
        if self.model_type == 'conv':
            b, q, c, h, w = x.shape
            x = x.contiguous().view(-1, c, h, w) 
        out = self.network(x)
        if len(out.shape) == 4: # conv layer output
            if out.shape[-1] == w:
                out = out[:,:,h//2:h//2+1, w//2:w//2+1] # extract center point
            out = out.view(out.size(0), -1)
            out = out.view(b, q, -1)
        return out