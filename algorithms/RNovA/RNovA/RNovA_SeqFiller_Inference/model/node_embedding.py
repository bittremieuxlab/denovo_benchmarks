import torch
from torch import nn

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

class NodeAbosoluteInputEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.intensity_embedding = nn.Sequential(SinusoidalPositionEmbedding(cfg.model.hidden_size, 
                                                                             cfg.embedding.intensity_lambda_max, 
                                                                             cfg.embedding.intensity_lambda_min),
                                                 nn.Linear(cfg.model.hidden_size, cfg.model.hidden_size))
        
        self.charge_embedding = nn.Embedding(10, cfg.model.hidden_size)
        self.class_embedding = nn.Embedding(6, cfg.model.hidden_size)
        self.peak_intensity_rank_embedding = nn.Embedding(cfg.data.ms2_max_peak_count, cfg.model.hidden_size)
    
    def forward(self, intensity, peak_intensity_rank, charge, peak_class):
        intensity = self.intensity_embedding(intensity)
        charge = self.charge_embedding(charge)
        peak_class = self.class_embedding(peak_class)
        peak_intensity_rank = self.peak_intensity_rank_embedding(peak_intensity_rank)
        return intensity+charge+peak_class+peak_intensity_rank

class NodeRelativeInputEncoderEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.relative_mass_embedding = SinusoidalPositionEmbedding(cfg.model.encoder.key_size, 
                                                                   cfg.embedding.mass_lambda_max, 
                                                                   cfg.embedding.mass_lambda_min)

        self.relative_intensity_embedding = SinusoidalPositionEmbedding(cfg.model.encoder.key_size, 
                                                                        cfg.embedding.intensity_lambda_max, 
                                                                        cfg.embedding.intensity_lambda_min)
    
    def forward(self, moverz, mass, intensity):
        relative_moverz = self.relative_mass_embedding(moverz)
        relative_mass = self.relative_mass_embedding(mass)
        relative_intensity = self.relative_intensity_embedding(intensity)
        return torch.stack([relative_mass,relative_intensity,relative_moverz],dim=2).repeat_interleave(self.cfg.model.encoder.rel_size//3,dim=2)
    
class NodeEmbedding(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.node_abosolute_embedding = NodeAbosoluteInputEmbedding(cfg)
        self.node_relative_embedding = NodeRelativeInputEncoderEmbedding(cfg)
    
    def forward(self, node_mass, peak_intensity_rank, peak_moverz, node_intensity, charge, node_class):
        node = self.node_abosolute_embedding(node_intensity, peak_intensity_rank, charge, node_class)
        node_relative_pos = self.node_relative_embedding(peak_moverz, node_mass, node_intensity)
        return node, node_relative_pos

class NodeRelativeMassDecoderEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.relative_mass_embedding = SinusoidalPositionEmbedding(cfg.model.hidden_size//cfg.model.decoder.num_heads, 
                                                                   cfg.embedding.mass_lambda_max, 
                                                                   cfg.embedding.mass_lambda_min)
    
    def forward(self, mass):
        relative_mass = self.relative_mass_embedding(mass).unsqueeze(2)
        return relative_mass
