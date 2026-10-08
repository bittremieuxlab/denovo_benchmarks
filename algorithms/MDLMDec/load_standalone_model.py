import yaml
import os
import torch
from loader_ import LoaderObj
from denovo_base.models.seq2seq import Seq2SeqMDLM
from glob import glob
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to instantiate Seq2SeqMDLM model and load previous weights
def load_model(project_directory, regex_extension='weights/*high*wts', device=None):
    
    with open(os.path.join(project_directory, "yaml/config.yaml")) as stream:
        config = yaml.safe_load(stream)

    # token dictionary
    L = LoaderObj()
    amod_dic = L.create_sequence_dictionary(config['loader']['dictionary_path'])
    synonyms = config['loader']['synonyms']
    if synonyms is not None:
        for pair in synonyms:
            letter_a, letter_b = pair
            amod_dic = L.synonym(letter_a, letter_b, amod_dic)
    amod_dic_rev = L.reverse_dictionary(amod_dic)

    # Seq2Seq model
    mdlm_config = config['decoder_mdlm']
    diff_config = config['decoder_mdlm']['diffusion_config']
    config['decoder_diff']['diffusion_config']['pad_tok_id'] = amod_dic['X']
    config['decoder_diff']['diffusion_config']['resume_checkpoint'] = False
    config['decoder_diff']['diffusion_config']['sequence_len'] = config['pep_length'][1] + 1 # b/c of eos token
    config['decoder_diff']['model_config']['self_condition'] = diff_config['model']['self_condition']

    model = Seq2SeqMDLM(
        encoder_config     = config['encoder_dict'],
        decoder_config     = config['decoder_diff']['model_config'],
        diff_config        = diff_config,
        top_peaks          = config['top_peaks'],
        max_peptide_length = config['pep_length'][1],
        token_dict         = amod_dic,
        ensemble_config    = config['decoder_diff']['ensemble'],
        masses_path        = config['loader']['masses_path'],
    )
    model.reverse = config['loader']['reverse']

    search = glob(os.path.join(project_directory, regex_extension))
    assert len(search) > 0, f"No weights file found in {project_directory}"
    wts_path = search[0]
    model.load_state_dict(torch.load(wts_path, map_location=device, weights_only=True))
    if device:
        model.to(device)
    
    return model


if __name__ == '__main__':
    
    # Set this
    regex_extension = "weights/*high*wts"
    
    model = load_model("./experiment_directory", regex_extension=regex_extension, device=device)
    
    dummy = {
        'mz': torch.empty(10, 150, device=device).uniform_(100,2000).sort(dim=-1)[0],
        'ab': torch.empty(10, 150, device=device).uniform_(0,1),
        'charge': torch.empty(10, device=device).uniform_(2, 4).round().int(),
        'mass': torch.empty(10, device=device).uniform_(300, 1000),
        'length': torch.full((10,), 150, device=device).int()
    }
    out_dict = model.forward_eval(dummy, progress=True)
    print(out_dict)
