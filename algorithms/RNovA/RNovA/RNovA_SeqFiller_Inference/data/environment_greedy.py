import torch
import numpy as np
from utils.BasicClass import Candidate_Residue_AA, Residual_AA
from .knapsack_build import knapsack_build_vsize32, knapsack_build_vsize64, next_aa_mask_builder

from math import ceil
import os
import hashlib

VERIFY_MASK = os.environ.get('RNOVA_VERIFY_MASK') == '1'
TIMING = os.environ.get('RNOVA_TIMING') == '1'
# Precursor tolerance of the knapsack mass constraint, in ppm of the precursor mass.
# Default 10 ppm (originally 5): tuned on the 9-species human sample, where ~7% of the
# database-search labels sit 5-10 ppm from their reported precursor.
PRECURSOR_PPM = float(os.environ.get('RNOVA_PRECURSOR_PPM', '10'))
# Refinement iterations. Default 8 (originally configs' max_iter, 4); see Environment.
DEFAULT_MAX_ITER = 8
import collections
TIMERS = collections.defaultdict(float)


KNAPSACK_CACHE_DIR = os.environ.get(
    'RNOVA_KNAPSACK_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'rnova_knapsack'))


def build_knapsack_cached(candidate_mass_cpu, mass_max, aa_resolution):
    """Knapsack reachability table, reused across runs via an on-disk cache.

    Building the table walks mass_max * aa_resolution columns once per candidate
    amino acid - 1.28e9 iterations for a 32-residue alphabet, about 49 s - and the
    result depends only on the arguments, so a run over several mgf files used to
    pay it again for every file.
    """
    key = hashlib.md5(np.round(np.asarray(candidate_mass_cpu, dtype=np.float64), 6).tobytes()
                      + repr((float(mass_max), float(aa_resolution))).encode()).hexdigest()
    path = os.path.join(KNAPSACK_CACHE_DIR, f'knapsack_{key}.npy')
    if os.path.exists(path):
        try:
            return np.load(path)
        except (OSError, ValueError):
            pass  # unreadable or truncated cache entry: fall through and rebuild
    builder = knapsack_build_vsize32 if len(candidate_mass_cpu) <= 32 else knapsack_build_vsize64
    matrix = builder(candidate_mass_cpu, mass_max, aa_resolution)
    try:
        os.makedirs(KNAPSACK_CACHE_DIR, exist_ok=True)
        tmp = path + f'.{os.getpid()}.tmp'
        with open(tmp, 'wb') as f:   # file handle, so np.save does not append '.npy'
            np.save(f, matrix)
        os.replace(tmp, path)        # atomic, so concurrent runs cannot read a partial file
    except OSError:
        pass                         # cache is an optimisation, never fatal
    return matrix



KNAPSACK_MASS_MAX = 4000
AA_RESOLUTION = 10000


def resolve_candidates(candidate_amino_acids):
    """Candidate names -> Residual_AA objects, sorted the way the decoder indexes them.

    Names absent from AA_PTM_Mol_Formula must be ``X[delta_mass]`` and are registered
    on the fly. The sort order fixes both the decoder's vocabulary order and the
    knapsack cache key, so build-time prebuilds must go through this function too.
    """
    db = Candidate_Residue_AA()
    candidate_aa = []
    for aa in candidate_amino_acids:
        try:
            candidate_aa.append(db[aa])
        except KeyError:
            if '[' not in aa or not aa.endswith(']'):
                raise ValueError(f"unknown candidate {aa!r}: not in AA_PTM_Mol_Formula and not X[delta_mass]")
            new_temp_aa = Residual_AA(aa[:aa.find('[')],
                                      n_terminal_PTM='',
                                      c_terminal_PTM='',
                                      r_group_PTM=aa[aa.find('[')+1:-1],
                                      embedding_db_index=len(db)+3,
                                      composition=None,
                                      mass=float(aa[aa.find('[')+1:-1])+db[aa[0]].mass,
                                      full_name=aa)
            db.add_residue(aa, new_temp_aa)
            candidate_aa.append(db[aa])
    return sorted(candidate_aa)


def knapsack_for(candidate_aa):
    """Knapsack table for resolved candidates, from the on-disk cache when possible."""
    if len(candidate_aa) > 64:
        raise ValueError(
            f"{len(candidate_aa)} candidate amino acids given; the knapsack table only supports up to 64.")
    masses = np.array([aa.mass for aa in candidate_aa])
    return build_knapsack_cached(masses, KNAPSACK_MASS_MAX, AA_RESOLUTION)


class Environment(object):
    def __init__(self, cfg, model, inference_dl, device, candidate_amino_acids):
        self.cfg = cfg
        self.model = model
        self.inference_dl_ori = inference_dl
        self.device = device
        # Refinement iterations (RNOVA_MAX_ITER, default DEFAULT_MAX_ITER). This may exceed
        # configs' max_iter (4): the model's iteration embedding has max_iter*10 slots, sized
        # from the config, so it still loads; 8 raised AA AUC on the tuning samples, and gains
        # stopped beyond it.
        self.max_iter = int(os.environ.get('RNOVA_MAX_ITER', DEFAULT_MAX_ITER))
        if self.max_iter >= cfg.data.max_iter*10:
            raise ValueError(f"RNOVA_MAX_ITER={self.max_iter} exceeds the iteration embedding ({cfg.data.max_iter*10} slots)")
        
        # Input prepration.
        self.seq = torch.ones([self.cfg.train.batch_size,1],dtype=torch.long,device=self.device)
        self.seq_pos = torch.zeros_like(self.seq)
        self.seq_mass_forward = torch.zeros_like(self.seq,dtype=torch.float)
        self.iter_num = torch.ones_like(self.seq)

        # Reserve 2 times of max len position for ensure 
        # there is enough space in cache tensor.
        # And fill in first seq token.
        self.candidate_ptm_aa = Candidate_Residue_AA()
        candidate_aa = resolve_candidates(candidate_amino_acids)
        self.candidate_aa = torch.tensor([aa.embedding_db_index for aa in candidate_aa],device='cuda',dtype=torch.long).unsqueeze(0)
        self.basic_candidate_aa = torch.tensor([self.candidate_ptm_aa[aa.amino_acid_name].embedding_db_index if aa.amino_acid_name!='C' else self.candidate_ptm_aa['C|UniMod:4'].embedding_db_index for aa in candidate_aa],device='cuda',dtype=torch.long).unsqueeze(0)
        self.candidate_mass = torch.tensor([aa.mass for aa in candidate_aa],device='cuda',dtype=torch.float).unsqueeze(0)
        self.candidate_mass_cpu = np.array([aa.mass for aa in candidate_aa])
        self.candidate_aa_num = self.candidate_aa.size(1)
        self.result_seq_ntoc = torch.zeros([self.cfg.train.batch_size,self.cfg.data.peptide_max_len*20], device=self.device, dtype=torch.long)
        self.result_seq_score_ntoc = torch.zeros_like(self.result_seq_ntoc, dtype=torch.float)
        self.result_seq_cton = torch.zeros_like(self.result_seq_ntoc)
        self.result_seq_score_cton = torch.zeros_like(self.result_seq_score_ntoc)

        self.aa_resolution = AA_RESOLUTION
        self.knapsack_matrix = knapsack_for(candidate_aa)

        # Mirror of the knapsack table on the device. The per-step mask used to be
        # built in a Python loop over the batch, which forced a GPU->CPU sync on
        # every decode step; see inference_mask_gpu.
        self.knapsack_gpu = torch.from_numpy(self.knapsack_matrix.view(np.int64)).to(self.device)
        self.knapsack_len = self.knapsack_gpu.numel()
        self.knapsack_window = 2048  # power of two; grown on demand in inference_mask_gpu
        self.knapsack_offsets = torch.arange(self.knapsack_window, device=self.device, dtype=torch.long)
        self.knapsack_bits = torch.arange(len(self.candidate_mass_cpu), device=self.device, dtype=torch.long)
        
        self.k_cache_self = torch.zeros(self.cfg.model.decoder.num_layers,
                                        self.cfg.train.batch_size,
                                        ceil(self.cfg.data.peptide_max_len*self.max_iter*4),
                                        self.cfg.model.decoder.num_heads,
                                        self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads,
                                        dtype=torch.float16,
                                        device=device)
        self.v_cache_self = torch.zeros(self.cfg.model.decoder.num_layers,
                                        self.cfg.train.batch_size,
                                        ceil(self.cfg.data.peptide_max_len*self.max_iter*4),
                                        self.cfg.model.decoder.num_heads,
                                        self.cfg.model.hidden_size//self.cfg.model.decoder.num_heads,
                                        dtype=torch.float16,
                                        device=device)

    def __iter__(self):
        self.inference_dl = iter(self.inference_dl_ori)
        return self

    def __next__(self):
        if not TIMING:
            candidate_aa, decoder_step_input, title = self.exploration_initializing()
            decoder_step_input = self.next_aa_choice(candidate_aa, decoder_step_input)

            while (self.iter_num<=self.max_iter).any():
                candidate_aa = self.model(**decoder_step_input)
                decoder_step_input = self.next_aa_choice(candidate_aa, decoder_step_input)
            result, result_score = self.result_generator()
            return result, result_score, title

        import time as _t
        def _sync(): torch.cuda.synchronize()
        t0=_t.perf_counter()
        candidate_aa, decoder_step_input, title = self.exploration_initializing(); _sync()
        TIMERS['init'] += _t.perf_counter()-t0
        t0=_t.perf_counter()
        decoder_step_input = self.next_aa_choice(candidate_aa, decoder_step_input); _sync()
        TIMERS['mask'] += _t.perf_counter()-t0
        while (self.iter_num<=self.max_iter).any():
            TIMERS['steps'] += 1
            t0=_t.perf_counter()
            candidate_aa = self.model(**decoder_step_input); _sync()
            TIMERS['model'] += _t.perf_counter()-t0
            t0=_t.perf_counter()
            decoder_step_input = self.next_aa_choice(candidate_aa, decoder_step_input); _sync()
            TIMERS['mask'] += _t.perf_counter()-t0
        t0=_t.perf_counter()
        result, result_score = self.result_generator(); _sync()
        TIMERS['result'] += _t.perf_counter()-t0
        TIMERS['batches'] += 1
        return result, result_score, title
        
    def exploration_initializing(self):
        node_input, precursor_mass, title = next(self.inference_dl)
        self.precursor_mass = precursor_mass
        
        self.iter_num[:] = 1
        
        self.seq = self.seq[:len(title)]
        self.seq_pos = self.seq_pos[:len(title)]
        self.iter_num = self.iter_num[:len(title)]
        self.seq_mass_forward = self.seq_mass_forward[:len(title)]
        self.k_cache_self = self.k_cache_self[:,:len(title)]
        self.v_cache_self = self.v_cache_self[:,:len(title)]
        self.result_seq_ntoc = self.result_seq_ntoc[:len(title)]
        self.result_seq_score_ntoc = self.result_seq_score_ntoc[:len(title)]
        self.result_seq_cton = self.result_seq_cton[:len(title)]
        self.result_seq_score_cton = self.result_seq_score_cton[:len(title)]

        sequence_input = {
            'seq': self.seq,
            'seq_pos': self.seq_pos,
            'seq_iter': self.iter_num,
            'seq_mass_forward': self.seq_mass_forward,
            # Constant Value for a batch
            'candidate_aa': self.basic_candidate_aa,
            'candidate_aa_mass': self.candidate_mass
        }

        k_cache, v_cache = self.model.encoder_forward(**node_input)
        candidate_aa = self.model(sequence_input, 0, self.k_cache_self, self.v_cache_self, k_cache, v_cache)

        decoder_step_input = {
            'cache_seqlens': 0,
            'sequence_input': sequence_input,
            'k_cache_self': self.k_cache_self,
            'v_cache_self': self.v_cache_self,
            'k_cache_cross': k_cache,
            'v_cache_cross': v_cache
        }
        self.result_seq_ntoc.zero_()
        self.result_seq_score_ntoc.zero_()
        self.result_seq_cton.zero_()
        self.result_seq_score_cton.zero_()
        return candidate_aa, decoder_step_input, title
    
    def inference_mask_gpu(self, remain_mass, ms1_threshold):
        """Per-sequence knapsack mask for a whole batch, without leaving the GPU.

        Reproduces the original numpy slice semantics exactly, edge cases included:
        a window reaching past the end of the table admits every amino acid, while
        one whose start goes negative (numpy read that as an empty slice) or whose
        width is zero admits none.
        """
        x = remain_mass.reshape(-1)
        y = ms1_threshold.reshape(-1)
        start, width = x - y, 2 * y
        need = int(width.max())
        if need > self.knapsack_window:
            # Heavy, highly charged precursors (or a wide ppm tolerance) need a wider window;
            # grow to the next power of two so the pairwise OR below stays exact.
            self.knapsack_window = 1 << (need - 1).bit_length()
            self.knapsack_offsets = torch.arange(self.knapsack_window, device=self.device, dtype=torch.long)

        idx = start.unsqueeze(1) + self.knapsack_offsets.unsqueeze(0)
        valid = (self.knapsack_offsets.unsqueeze(0) < width.unsqueeze(1)) \
            & (idx >= 0) & (idx < self.knapsack_len) & (start >= 0).unsqueeze(1)
        vals = torch.where(valid, self.knapsack_gpu[idx.clamp(0, self.knapsack_len - 1)],
                           torch.zeros((), dtype=self.knapsack_gpu.dtype, device=self.device))

        while vals.size(1) > 1:                      # pairwise OR-reduce the window
            vals = vals[:, 0::2] | vals[:, 1::2]
        acc = vals.squeeze(1)

        mask = ((acc.unsqueeze(1) >> self.knapsack_bits) & 1).bool()
        return mask | ((x + y) > self.knapsack_len).unsqueeze(1)

    def inference_mask_cpu(self, remain_mass, ms1_threshold):
        """Original per-sequence loop, kept so the vectorised path can be checked."""
        out = []
        for x, y in zip(remain_mass.cpu(), ms1_threshold.cpu()):
            if x + y > len(self.knapsack_matrix):
                out += [torch.ones(self.candidate_mass.size(1), dtype=bool)]
            else:
                out += [torch.from_numpy(next_aa_mask_builder(
                    self.knapsack_matrix[x - y:x + y], len(self.candidate_mass_cpu)))]
        return torch.stack(out).to(self.device)

    # @torch.compile  (disabled: recompile thrash)
    def next_aa_choice(self, candidate_aa, decoder_step_input):
        ms1_threshold = (self.precursor_mass*PRECURSOR_PPM*1e-6*self.aa_resolution).round().long()
        remain_mass = self.precursor_mass-decoder_step_input['sequence_input']['seq_mass_forward']
        remain_mass = (remain_mass*self.aa_resolution).round().long()
        inference_mask = self.inference_mask_gpu(remain_mass, ms1_threshold)
        if VERIFY_MASK:
            reference = self.inference_mask_cpu(remain_mass, ms1_threshold)
            if not torch.equal(inference_mask, reference):
                raise AssertionError("GPU knapsack mask differs from the CPU reference")
        candidate_aa = candidate_aa.to(torch.float).squeeze(-1)
        candidate_aa = candidate_aa.masked_fill(~inference_mask,-float('inf'))
        next_aa_score, next_aa = candidate_aa.max(1,keepdim=True)
        seq = decoder_step_input['sequence_input']['candidate_aa'][0,next_aa]
        seq_mass = decoder_step_input['sequence_input']['seq_mass_forward']+decoder_step_input['sequence_input']['candidate_aa_mass'][0,next_aa]
        seq_pos = decoder_step_input['sequence_input']['seq_pos']
        seq_result = self.candidate_aa[0,next_aa]
        if self.max_iter % 2:
            final_max_iter_ntoc = (self.iter_num == self.max_iter).squeeze(1)
            final_max_iter_cton = (self.iter_num == self.max_iter-1).squeeze(1)
        else:
            final_max_iter_ntoc = (self.iter_num == self.max_iter-1).squeeze(1)
            final_max_iter_cton = (self.iter_num == self.max_iter).squeeze(1)
        self.result_seq_ntoc[final_max_iter_ntoc] = self.result_seq_ntoc[final_max_iter_ntoc].scatter(-1,seq_pos[final_max_iter_ntoc],seq_result[final_max_iter_ntoc])
        self.result_seq_score_ntoc[final_max_iter_ntoc] = self.result_seq_score_ntoc[final_max_iter_ntoc].scatter(-1,seq_pos[final_max_iter_ntoc],next_aa_score[final_max_iter_ntoc])
        self.result_seq_cton[final_max_iter_cton] = self.result_seq_cton[final_max_iter_cton].scatter(-1,seq_pos[final_max_iter_cton],seq_result[final_max_iter_cton])
        self.result_seq_score_cton[final_max_iter_cton] = self.result_seq_score_cton[final_max_iter_cton].scatter(-1,seq_pos[final_max_iter_cton],next_aa_score[final_max_iter_cton])

        seq_pos = seq_pos + 1

        finish_iter_flag = (self.precursor_mass-seq_mass) < 10
        self.iter_num[finish_iter_flag] += 1
        if (self.iter_num>self.max_iter).all():
            return decoder_step_input
        else:
            n_term_inference_flag = (self.iter_num%2).bool()
            seq = torch.where(finish_iter_flag & n_term_inference_flag, 1, seq)
            seq = torch.where(finish_iter_flag & ~n_term_inference_flag, 2, seq)
            seq_mass = torch.where(finish_iter_flag, 0, seq_mass)
            seq_pos = torch.where(finish_iter_flag, 0, seq_pos)
            
            decoder_step_input['sequence_input']['seq'] = seq
            decoder_step_input['sequence_input']['seq_pos'] = seq_pos
            decoder_step_input['sequence_input']['seq_iter'] = self.iter_num
            decoder_step_input['sequence_input']['seq_mass_forward'] = seq_mass
            decoder_step_input['cache_seqlens'] += 1
            return decoder_step_input
    
    def result_generator(self):
        score_max_flag = self.result_seq_score_ntoc.sum(1,keepdim=True)>self.result_seq_score_cton.sum(1,keepdim=True)
        result_seq = torch.where(score_max_flag,self.result_seq_ntoc,self.result_seq_cton).cpu().tolist()
        result_seq_score = torch.where(score_max_flag,self.result_seq_score_ntoc,self.result_seq_score_cton).cpu().tolist()

        result = [[] for _ in range(self.result_seq_ntoc.size(0))]
        result_score = [[] for _ in range(self.result_seq_ntoc.size(0))]
        for i in range(self.result_seq_ntoc.size(0)):
            result_seq_row = result_seq[i]
            result_seq_score_row = result_seq_score[i]
            result_seq_row = [aa for aa in result_seq_row if aa!=0]
            result_seq_score_row = [aa for aa in result_seq_score_row if aa!=0]
            if not score_max_flag[i]: 
                result_seq_row = result_seq_row[::-1]
                result_seq_score_row = result_seq_score_row[::-1]
            for aa, aa_score in zip(result_seq_row,result_seq_score_row):
                result[i] += [self.candidate_ptm_aa[aa]]
                result_score[i] += [aa_score]
        return result, result_score