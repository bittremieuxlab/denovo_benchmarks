import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

class SelfMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 rel_size: int,
                 key_size: int,
                 rel_feature_num: int):
        super().__init__()
        assert hidden_size % 8 == 0
        assert hidden_size % rel_size == 0
        assert rel_size % rel_feature_num == 0
        self.rel_size = rel_size
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.rel_feature_num = rel_feature_num
        self.linear_qkv = nn.Linear(hidden_size, rel_size*key_size*3)
        self.output_layer = nn.Linear(rel_size*key_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self,x,pos):
        q, k, v = self.linear_qkv(x).view(x.size(0), -1, self.rel_size, self.key_size*3).chunk(3, dim=-1)
        k = k/torch.linalg.vector_norm(k,dim=-1,keepdim=True)
        q, k = self.apply_rope(q, pos), self.apply_rope(k, pos)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        postx = F.scaled_dot_product_attention(q,k,v,scale=1).transpose(1,2).flatten(2,3)
        postx = self.output_layer(postx)
        x = self.ln(x + postx)
        return x
    
    @staticmethod
    def apply_rope(x, dis):
        dis_sin, dis_cos = dis.chunk(2,dim=-1)
        x0, x1 = x.chunk(2,dim=-1)
        return torch.concat([x0*dis_cos-x1*dis_sin,\
                             x0*dis_sin+x1*dis_cos], dim = -1)

class FFNGLU(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        assert hidden_size%3==0
        # 根据“GLU Variants Improve Transformer”，采用GEGLU结构做FFN.
        self.pre_ffn_gate = nn.Sequential(nn.Linear(hidden_size, 2*hidden_size//3, bias=False),
                                          nn.SiLU())
        self.pre_ffn = nn.Linear(hidden_size, 2*hidden_size//3, bias=False)
        self.post_ffn = nn.Linear(2*hidden_size//3, hidden_size, bias=False)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        postx = self.post_ffn(self.pre_ffn_gate(x)*self.pre_ffn(x))
        x = self.ln(x + postx)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, rel_size, key_size, rel_feature):
        super().__init__()
        self.mha = SelfMultiHeadAttn(hidden_size, rel_size, key_size, rel_feature)
        self.ffn = FFNGLU(hidden_size)

    def forward(self,x,pos):
        x = self.mha(x,pos)
        x = self.ffn(x)
        return x
