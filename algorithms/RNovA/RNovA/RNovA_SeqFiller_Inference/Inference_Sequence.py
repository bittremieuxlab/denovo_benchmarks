import logging
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import RNovA
from data import RnovaDataset, RnovaCollator, DataPrefetcher, Environment

from hydra import initialize, compose

logger = logging.getLogger(__name__)

# Sequences whose mean per-residue score is below this are not written at all. Fixed on
# purpose (no option or environment variable): on the 9-species human, mouse and tomato
# benchmark samples it dropped 19-35% of predictions and not one correct peptide.
MIN_MEAN_SCORE = -2.0

def read_mgf(mgf_file):
    spectra = []
    with open(mgf_file) as f:
        for line in f:
            if line.startswith('BEGIN IONS'):
                charge = 2
                moverz = []
                intensity = []
            elif line.startswith('TITLE'): title = line.strip().split('=')[-1]
            elif line.startswith('PEPMASS'): precursor_moverz = float(line.strip().split('=')[-1].split(' ')[0])
            elif line.startswith('CHARGE'): 
                line = line.strip().split('=')[-1]
                if line.endswith('+'):
                    charge = int(line[:-1])
                elif line.endswith('-'):
                    charge = -int(line[:-1])
                else:
                    charge = int(line)
            elif line.startswith('SCANS'): scan = line.strip().split('=')[-1]
            elif line.startswith('RTINSECONDS'): rt = line.strip().split('=')[-1]
            elif line.startswith('END IONS'):
                try:
                    spectra += [{'title':scan, 'precursor_moverz':precursor_moverz,'precursor_charge':charge,'product_ion_moverz':np.array(moverz),'product_ion_intensity_log':np.log(np.array(intensity))}]
                except:
                    spectra += [{'title':title.split('.')[-2], 'precursor_moverz':precursor_moverz,'precursor_charge':charge,'product_ion_moverz':np.array(moverz),'product_ion_intensity_log':np.log(np.array(intensity))}]
            elif line[0].isnumeric():
                mz, peak_intensity = line.strip().split(' ')
                mz = float(mz)
                peak_intensity = float(peak_intensity)
                moverz.append(mz)
                intensity.append(peak_intensity)
    return spectra

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    # RNOVA_BATCH_SIZE overrides configs/train/train.yaml, e.g. for GPUs below ~16 GB
    overrides = [f"train.batch_size={os.environ['RNOVA_BATCH_SIZE']}"] if os.environ.get('RNOVA_BATCH_SIZE') else []
    with initialize(config_path="configs", version_base=None): cfg = compose(config_name="config", overrides=overrides)
    local_rank = 0
    torch.cuda.set_device(local_rank)
    model = RNovA(cfg).to(local_rank)
    model_checkpoint = torch.load('save/rnova.pt',map_location={'cuda:0': f'cuda:{local_rank}'},weights_only=True)
    model.load_state_dict(model_checkpoint)
    
    mgf_files, candidate_amino_acids = sys.argv[1:-1], sys.argv[-1]
    candidate_amino_acids = candidate_amino_acids.split(';')
    for mgf_file in mgf_files:
        logger.info("Start analysing %s", mgf_file)
        spectra = read_mgf(mgf_file)

        ds = RnovaDataset(cfg,spectra)
        collator = RnovaCollator(cfg)
        train_dl = DataLoader(ds,batch_size=cfg.train.batch_size,collate_fn=collator,num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=4)
        train_dl = DataPrefetcher(train_dl, local_rank)
        environment = Environment(cfg, model, train_dl, local_rank, candidate_amino_acids)

        fw = open(mgf_file[:-4]+'_rnova_denovo_seq.csv', 'w')
        fw.write('title,sequence,score\n')
        n_written = n_dropped = 0
        for results, results_score, titles in environment:
            for result, result_score, title in zip(results, results_score, titles):
                result_str = []
                for aa in result:
                    name = aa.amino_acid_name
                    # 只保留非空的修饰项
                    mods = [mod for mod in [aa.r_group_PTM, aa.n_terminal_PTM, aa.c_terminal_PTM] if mod]
                    if mods: name += f"[{'|'.join(mods)}]"
                    result_str.append(name)
                result_str = ''.join(result_str)
                scores = [f"{s:.4f}" for s in result_score]
                # judged on the values as written, so readers of the file see the same mean
                values = [float(s) for s in scores]
                if not values or sum(values) / len(values) < MIN_MEAN_SCORE:
                    n_dropped += 1
                    continue
                fw.write(f"{title},{result_str},{';'.join(scores)}\n")
                n_written += 1
        fw.close()
        logger.info("Results saved to %s_rnova_denovo_seq.csv (%d written, %d below mean score %.1f not written)",
                    mgf_file[:-4], n_written, n_dropped, MIN_MEAN_SCORE)

if __name__ == "__main__":
    main()
