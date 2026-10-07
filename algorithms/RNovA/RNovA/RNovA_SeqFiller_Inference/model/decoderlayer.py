import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
from torch.nn.attention import sdpa_kernel, SDPBackend

class SelfMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int):
        super().__init__()
        assert hidden_size % 8 == 0
        assert hidden_size % num_heads == 0
        assert hidden_size // num_heads % 8 == 0
        self.num_heads = num_heads
        self.head_size = hidden_size//num_heads
        self.linear_qkv = nn.Linear(hidden_size, hidden_size*3)
        self.linear_candidate = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self,seq,candidate_aa,cache_seqlens,k_cache,v_cache):
        q, k, v = self.linear_qkv(seq).view(-1, 1, self.num_heads, self.head_size*3).chunk(3,dim=-1)
        candidate_aa = self.linear_candidate(candidate_aa)
        k = k/torch.linalg.vector_norm(k,dim=-1,keepdim=True)
        k = k.to(torch.float16)
        postx = flash_attn_with_kvcache(q,k_cache,v_cache,k,v,
                                        cache_seqlens=cache_seqlens,softmax_scale=1).flatten(2,3)
        postx = self.output_layer(postx)
        candidate_aa = self.ln(postx + candidate_aa)
        seq = self.ln(seq + postx)
        return seq, candidate_aa
    
class CrossMultiHeadAttn(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int):
        super().__init__()
        assert hidden_size % 8 == 0
        assert hidden_size % num_heads == 0
        assert hidden_size // num_heads % 8 == 0
        assert num_heads % 4 == 0
        self.num_heads = num_heads
        self.head_size = hidden_size//num_heads
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self,x,x_mass,k_cache,v_cache):
        q = self.linear_q(x).view(x.size(0), -1, self.num_heads, self.head_size)
        q = self.apply_rope(q, x_mass).transpose(1,2)
        postx = F.scaled_dot_product_attention(q,k_cache,v_cache,scale=1).transpose(1,2).flatten(2,3)
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

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.mha = SelfMultiHeadAttn(hidden_size, num_heads)
        self.cross_mha = CrossMultiHeadAttn(hidden_size, num_heads)
        self.ffn = FFNGLU(hidden_size)

    def forward(self,seq,seq_mass,
                candidate_aa,candidate_aa_mass,
                cache_seqlens,
                k_cache_self,v_cache_self,
                k_cache_cross,v_cache_cross):
        seq, candidate_aa = self.mha.forward(seq,candidate_aa,cache_seqlens,k_cache_self,v_cache_self)
        seq = self.cross_mha(seq,seq_mass,k_cache_cross,v_cache_cross)
        seq = self.ffn(seq)
        candidate_aa = self.cross_mha(candidate_aa,candidate_aa_mass,k_cache_cross,v_cache_cross)
        candidate_aa = self.ffn(candidate_aa)
        return seq, candidate_aa
