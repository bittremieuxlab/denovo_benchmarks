import torch
from torch import nn
from .encoderlayer import EncoderLayer
from .decoderlayer import DecoderLayer
from .node_embedding import NodeEmbedding, NodeRelativeMassDecoderEmbedding
from .sequence_embedding import SequenceEmbedding
from math import ceil

class Decoder_Cross_Projector(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.decoder_cross_kv = nn.Linear(cfg.model.hidden_size,
                                          cfg.model.hidden_size*cfg.model.decoder.num_layers*2)
        self.node_mass_decoder_embedding = NodeRelativeMassDecoderEmbedding(cfg)
    
    def forward(self, node, node_mass):
        k_cache, v_cache = self.decoder_cross_kv(node).view(node.size(0),-1,
                                                            self.cfg.model.encoder.rel_size*self.cfg.model.decoder.num_layers*2,
                                                            self.cfg.model.encoder.key_size).chunk(2,dim=-2)
        node_mass = self.node_mass_decoder_embedding(node_mass)
        dis_sin, dis_cos = node_mass.chunk(2,dim=-1)
        x0, x1 = k_cache.chunk(2,dim=-1)
        k_cache = torch.concat([x0*dis_cos-x1*dis_sin,x0*dis_sin+x1*dis_cos], dim = -1)
        return k_cache.transpose(1,2).chunk(self.cfg.model.decoder.num_layers, dim=1), v_cache.transpose(1,2).chunk(self.cfg.model.decoder.num_layers, dim=1)

class RNovA(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.node_embedding = NodeEmbedding(cfg)
        self.node_decoder_embedding = Decoder_Cross_Projector(cfg)
        self.seq_embedding = SequenceEmbedding(cfg)
        self.encoder = nn.ModuleList([EncoderLayer(cfg.model.hidden_size, 
                                                   cfg.model.encoder.rel_size,
                                                   cfg.model.encoder.key_size, 
                                                   cfg.model.encoder.rel_feature_num) \
                                                   for _ in range(cfg.model.encoder.num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg.model.hidden_size, 
                                                   cfg.model.decoder.num_heads) \
                                                   for _ in range(cfg.model.decoder.num_layers)])
        self.output = nn.Linear(cfg.model.hidden_size, 1)
    
    @torch.no_grad()
    @torch.autocast('cuda',dtype=torch.float16)
    def encoder_forward(self, node_mass, peak_intensity_rank, peak_moverz, node_intensity, charge, node_class):
        node, node_relative_pos = self.node_embedding(node_mass, peak_intensity_rank, peak_moverz, node_intensity, charge, node_class)
        for encoder_layer in self.encoder: node = encoder_layer(node, node_relative_pos)
        k_cache, v_cache = self.node_decoder_embedding(node, node_mass)
        return k_cache, v_cache

    @torch.no_grad()
    @torch.autocast('cuda',dtype=torch.float16)
    def forward(self, sequence_input, cache_seqlens, k_cache_self, v_cache_self, k_cache_cross, v_cache_cross):
        seq,seq_mass,candidate_aa,candidate_aa_mass = self.seq_embedding(**sequence_input)
        for i, decoder_layer in enumerate(self.decoder):
            seq, candidate_aa = decoder_layer.forward(seq,seq_mass,
                                                      candidate_aa,candidate_aa_mass,
                                                      cache_seqlens,
                                                      k_cache_self[i], v_cache_self[i],
                                                      k_cache_cross[i],v_cache_cross[i])
        candidate_aa = self.output(candidate_aa)
        return candidate_aa
