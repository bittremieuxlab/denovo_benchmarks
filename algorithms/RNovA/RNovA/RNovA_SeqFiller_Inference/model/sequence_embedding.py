import torch
import torch.nn as nn

from math import ceil

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim, lambda_max, lambda_min) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        base = lambda_max/(2*torch.pi)
        scale = lambda_min/lambda_max
        div_term = base*scale**(torch.arange(0, dim, 2, dtype=torch.float)/dim)
        self.register_buffer('div_term', div_term)

    def forward(self, mass_position):
        pe_sin = torch.sin(mass_position.unsqueeze(dim=-1) / self.div_term)
        pe_cos = torch.cos(mass_position.unsqueeze(dim=-1) / self.div_term)
        return torch.concat([pe_sin, pe_cos],dim=-1).float()

class SequenceAbosoluteEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_embedding = nn.Embedding(23, cfg.model.hidden_size)
        self.seq_pos_embedding = nn.Embedding(ceil(cfg.data.peptide_max_len*4),cfg.model.hidden_size)
        self.seq_iter_embedding = nn.Embedding(ceil(cfg.data.max_iter*10),cfg.model.hidden_size)
    
    def forward(self,seq,seq_pos,seq_iter):
        seq = self.seq_embedding(seq)
        seq_pos = self.seq_pos_embedding(seq_pos.clip(max=self.cfg.data.peptide_max_len*4-1))
        seq_iter = self.seq_iter_embedding(seq_iter.clip(max=self.cfg.data.max_iter*10-1))
        return seq+seq_pos+seq_iter
    
class SequenceRelativeEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        assert cfg.model.decoder.num_heads % 4 == 0

        self.cfg = cfg
        self.num_heads = cfg.model.decoder.num_heads
        self.seq_mass_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size//cfg.model.decoder.num_heads, 
                                                              cfg.embedding.mass_lambda_max, 
                                                              cfg.embedding.mass_lambda_min)
    
    def forward(self,seq_mass_forward):
        seq_mass_forward_1 = self.seq_mass_embedding(seq_mass_forward)
        seq_mass_forward_2 = self.seq_mass_embedding(seq_mass_forward/2)
        return torch.stack([seq_mass_forward_1,seq_mass_forward_2],dim=2).repeat_interleave(self.num_heads//2,dim=2)

class SequenceEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.seq_abosolute_embedding = SequenceAbosoluteEmbedding(cfg)
        self.seq_relative_embedding = SequenceRelativeEmbedding(cfg)
    
    def forward(self,seq,seq_pos,seq_iter,seq_mass_forward,candidate_aa,candidate_aa_mass):
        seq_len = seq.size(1)
        candidate_aa_size = candidate_aa.size(1)
        candidate_aa = candidate_aa.repeat(1,seq_len)
        candidate_aa_pos = seq_pos.repeat_interleave(candidate_aa_size,1)
        candidate_aa_iter = seq_iter.repeat_interleave(candidate_aa_size,1)
        candidate_aa_with_seq_mass = candidate_aa_mass.repeat(1,seq_len)+seq_mass_forward.repeat_interleave(candidate_aa_size,1)
        
        seq = self.seq_abosolute_embedding(seq,seq_pos,seq_iter)
        seq_mass = self.seq_relative_embedding(seq_mass_forward)
        
        candidate_aa = self.seq_abosolute_embedding(candidate_aa,candidate_aa_pos,candidate_aa_iter)
        candidate_aa_mass = self.seq_relative_embedding(candidate_aa_with_seq_mass)
        return seq,seq_mass,candidate_aa,candidate_aa_mass