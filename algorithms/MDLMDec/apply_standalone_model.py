from load_standalone_model import load_model
from loader_ import LoaderHF
import torch
from denovo_base.utils import Dict2dev
from tqdm import tqdm
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to set EOS as last non-AA token and then fill to the right side with NT
def fill_null_after_first_eos_token(intseq, EOS, NT):
    if len(intseq.shape) == 1:
        intseq = intseq[None]
    bs, sl = intseq.shape
    
    #length = ((intseq == EOS)|(intseq == NT)).int().argmax(1)
    #intseq[torch.arange(bs), length] = EOS
    #mask = (length > 0)[:,None]
    
    terminal = (intseq == EOS) | (intseq == NT)
    has_terminal = terminal.any(dim=1)
    length = terminal.int().argmax(dim=1)
    rows = torch.arange(bs, device=intseq.device)
    intseq[rows[has_terminal], length[has_terminal]] = EOS
    mask = has_terminal[:, None]
    
    index_array = torch.arange(sl)[None].repeat([bs, 1]).to(intseq.device)
    boolean_array = index_array > length[:, None]
    intseq[mask & boolean_array] = NT

    return intseq

# Function to turn integer outputs of model into variable length list of lists of strings
def to_list_of_strings(intseq, reverse, revdic, EOS, NT):
    if len(intseq.shape) == 1:
        intseq = intseq[None]
    is_reverse = lambda x: x[::-1] if reverse else x
    return [
        is_reverse([
            revdic[int(n)] for n in m if n not in [NT, EOS]
        ])
        for m in intseq
    ]

# Function to reverse the to_list_of_strings function
def to_list_of_numbers(array, aaseqlist, reverse):
    if len(array.shape) == 1:
        array = array[None]
    is_reverse = lambda x: x[::-1] if reverse else x
    return [
        is_reverse([
            array[i, j].item() for j, n in enumerate(m)
        ])
        for i, m in enumerate(aaseqlist)
    ]

def main():
    # Create and load model
    model = load_model("./experiment_directory", device=device)
    
    # Create data loader
    loader = LoaderHF(
        "parquet",
        synonyms=[['I','L']],
        reverse=model.reverse,
        batch_size=100,
        num_workers=2,
    )

    pbar = tqdm(loader.dataloader['test'], total=loader.size//100)
    output = {
        'spectrum_id': [],
        'filename': [],
        'index': [],
        'prediction': [],
        'aa_prob': [],
        'pep_prob': [],
        'aa_prob_min': [],
        'pep_prob_min': [],
    }
    # Iterate through dataset/dataloader
    for batch in pbar:
        # Add list data to output
        output['spectrum_id'].extend(batch['id'].tolist())
        output['filename'].extend(batch['file'].tolist())
        output['index'].extend(batch['index'].tolist())
        # Move pytorch data to device
        batch = Dict2dev(batch, device)
        # Predict peptides
        outdict = model.predict_sequence(batch, save_p=True, save_x=True, n=1)
        # Post-processing for predicted integer sequence
        prediction = outdict.pop('prediction')
        prediction = fill_null_after_first_eos_token(prediction, model.decoder.EOS, model.decoder.NT)
        pred_strings = to_list_of_strings(prediction, model.reverse, model.decoder.rev_outdict, model.decoder.EOS, model.decoder.NT)
        # Post-processing for logits and other confidence metrics
        logits = outdict.pop('logits').softmax(-1).gather(-1, prediction[...,None]).squeeze()
        aaprob_list = to_list_of_numbers(logits, pred_strings, model.reverse)
        pepprob = [np.prod(lst).item() for lst in aaprob_list]
        aamin = outdict.pop('aa_prob_min')
        aamin_list = to_list_of_numbers(aamin, pred_strings, model.reverse)
        pepmin = outdict.pop("pep_prob_min")

        # Add to output
        output['prediction'].extend(pred_strings)
        output['aa_prob'].extend(aaprob_list)
        output['aa_prob_min'].extend(aamin_list)
        output['pep_prob'].extend(pepprob)
        output['pep_prob_min'].extend(pepmin.cpu().tolist())
    
    # Write to output.parquet
    pd.DataFrame(output).to_parquet("output.parquet")

if __name__ == '__main__':
    main()
