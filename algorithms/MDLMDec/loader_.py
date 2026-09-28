from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import re
from glob import glob
import sys
import pandas as pd
import numpy as np
join = os.path.join

# Tokenizer function - Copied from enumerate_tokens.py in MassiveKB
def partition_modified_sequence(sequence):
    
    # Split apart letter+number from continuous letters
    #   [A-Z]{0,1}: 0 or 1 letters to start the sequence
    #   [+-]?: one or none of + or -
    #   [0-9]*: any amount of digits 0-9
    #   [.]?: any amount of periods
    #   [0-9]+: 1 to any amount of digits 0-9
    split = re.split("([A-Z]{0,1}[+-]?[0-9]*[.]?[0-9]+)", sequence)
    
    # Split (unmodified) strings into characters, and remove ['']
    list_of_lists = [[x] if re.search("[+-]", x) else list(x) for x in split if x != '']
    
    # Flatten
    tokenized_sequence = [m for n in list_of_lists for m in n]
    
    return tokenized_sequence

# Map function which will be applied to every loaded spectrum
def map_fn(example, tokenizer, dic=None, top=100, max_seq=50, reverse=False):
    if 'intensity_array' in example:
        ab = example['intensity_array']
        spectrum_length = len(ab)
        example['spectrum_length'] = spectrum_length
        ab_sort = (-ab).argsort()[:top]
        ab = ab[ab_sort]
        if spectrum_length > 0:
            ab /= (ab.max() + 1e-9)
        mz = example['mz_array'][ab_sort]
        mz_sort = mz.argsort()
        length = len(mz)
        mz_ = np.zeros(top)
        mz_[:len(mz_sort)] = mz[mz_sort]
        ab_ = np.zeros(top)
        ab_[:len(ab_sort)] = ab[mz_sort]
        example['mz_array'] = mz_
        example['intensity_array'] = ab_
    if 'precursor_charge' in example:
        example['precursor_charge'] = example['precursor_charge']
    if 'precursor_mass' in example:
        example['precursor_mass'] = example['precursor_mass']
    if 'modified_sequence' in example:
        tokenized_sequence = tokenizer(example['modified_sequence'])
        peptide_length = len(tokenized_sequence)
        if reverse:
            tokenized_sequence = tokenized_sequence[::-1]
        example['tokenized_sequence'] = np.array([dic.get(m, dic['X']) for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=np.int32)
        example['peptide_length'] = peptide_length
    if 'name' in example:
        example['experiment_name'] = example['name'] # compat
    
    return example

# Collate function to tie individual loaded spectra together
def collate_fn(batch_list, custom_columns=[]):
    out = {}
    #out['experiment_name'] = np.array([m['experiment_name'] for m in batch_list])
    if 'intensity_array' in batch_list[0]:
        out['length'] = th.tensor(np.stack([m['spectrum_length'] for m in batch_list]), dtype=th.int32)
        #maxlength = out['length'].max()
        #out['mz'] = th.tensor(np.stack([m['mz_array'][:maxlength] for m in batch_list]), dtype=th.float32)
        out['mz'] = th.tensor(np.stack([m['mz_array'] for m in batch_list]), dtype=th.float32)
        #out['ab'] = th.tensor(np.stack([m['intensity_array'][:maxlength] for m in batch_list]), dtype=th.float32)
        out['ab'] = th.tensor(np.stack([m['intensity_array'] for m in batch_list]), dtype=th.float32)
    if 'precursor_charge' in batch_list[0]:
        out['charge'] = th.tensor(np.stack([m['precursor_charge'] for m in batch_list]), dtype=th.int32)
    if 'precursor_mass' in batch_list[0]:
        out['mass'] = th.tensor(np.stack([m['precursor_mass'] for m in batch_list]), dtype=th.float32)
    if 'tokenized_sequence' in batch_list[0].keys():
        out['peplen'] = th.tensor(np.stack([m['peptide_length'] for m in batch_list]), dtype=th.int32)
        #out['intseq'] = th.tensor(np.stack([m['tokenized_sequence'][:out['peplen'].max()] for m in batch_list]), dtype=th.int32)
        out['intseq'] = th.tensor(np.stack([m['tokenized_sequence'] for m in batch_list]), dtype=th.int32)
        #out['intseq'] = th.tensor(np.stack([m['tokenized_sequence'][:41] for m in batch_list]), dtype=th.int32)
    if 'chimeric' in batch_list[0].keys():
        out['chimeric'] = th.tensor(np.stack([m['chimeric'] for m in batch_list]))
    if 'Hyperscore' in batch_list[0]:
        out['hyperscore'] = th.tensor(np.stack([m['Hyperscore'] for m in batch_list]))
    if 'title' in batch_list[0]:
        out['id'] = np.array([m['title'] for m in batch_list])
    if 'file' in batch_list[0]:
        out['file'] = np.array([m['file'] for m in batch_list])
    if 'index' in batch_list[0]:
        out['index'] = np.array([m['index'] for m in batch_list])
    for column in custom_columns:
        out[column] = np.array([m[column] for m in batch_list])

    return out

# Loader object with functions to assist in building dataset
class LoaderObj:
    # Creating a pytorch dataloader
    def build_dataloader(self, dataset, batch_size, num_workers, collate_fn, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
    
    # Create sequence dictionary from tsv file
    def create_sequence_dictionary(self, dictionary_path):
        amod_dic = {
            line.split()[0]:m for m, line in enumerate(open(dictionary_path).read().strip().split('\n'))
        }
        amod_dic['X'] = len(amod_dic)

        return amod_dic
    
    # Create a reverse dictionary
    def reverse_dictionary(self, amod_dic):
        return {b:a for a,b in amod_dic.items()}

    # Set synonym tokens to same value
    def synonym(self, token1, token2, amod_dic):
        low = int(np.minimum(amod_dic[token1], amod_dic[token2]))
        high = int(np.maximum(amod_dic[token1], amod_dic[token2]))
        amod_dic[token1] = amod_dic[token2] = low
        amod_dic = {key:value if value < high else value-1 for key, value in amod_dic.items()}
        return amod_dic
    
    # Find mapping of sequences to files
    def create_label_dictionary(self, dataset_path, filename="species_list.txt"):
        filepath = join(dataset_path, f"parquet/labeled_sequences/{filename}")
        List = open(filepath).read().strip().split("\n")
        #tsv = pd.read_csv(filepath, sep='\t', header=None, names=['species_name', 'count'])
        label_dict = {j:i for i,j  in enumerate(List)}
        label_dictr = {j:i for i,j in label_dict.items()}

        return label_dict, label_dictr
    
    # Function to create tokenizer if a path is specified to enumerate_tokens.py file
    def create_tokenizer(self, tokenizer_path):
        # Tokenizer
        # - RULES
        #   1. There is a file named enumerate_tokens.py with a subroutine named
        #      partition_modified_sequence
        sys.path.insert(0, tokenizer_path)
        from enumerate_tokens import partition_modified_sequence
        tokenizer = partition_modified_sequence

        return tokenizer

    # Function to load token masses from a tsv file
    def load_token_masses(self, masses_path, regex='*masses.tsv'):
        try:
            masses_path = glob(join(masses_path, regex))[0]
            mass_frame = pd.read_csv(masses_path, delimiter="\t", header=None)
            massdic = {m:n for m,n in zip(mass_frame[0], mass_frame[1])}
        except:
            massdic = None

        return massdic

    # Function to read *size.tsv file and figure out the size of data splits
    def find_set_size_for_tqdm(self, dataset_path, include_name=None, exclude_name=None, regex='*sizes.tsv'):
        ss_path = join(dataset_path, regex)
        ss_path = glob(ss_path)[0]
        if os.path.exists(ss_path):
            split_sizes = pd.read_csv(ss_path, sep="\t", header=None, names=["name", "count"], index_col="name")
            
            # if None, then read everything except for val_name
            # search split_sizes based on val_name to accomodate 9 species 
            #  cross validation where val and train are in the same directory
            if include_name == None:
                size = int(split_sizes.query(f"name.str.contains('{exclude_name}')==False")['count'].sum())
            # else affirmatively find files that contain train_name
            else:
                size = int(split_sizes.query(f"name.str.contains('{include_name}')")['count'].sum())
        else:
            # This should still work with tqdm progress bar
            size = float('inf')

        return size

    # Function that creates hugging face dataset object
    def _load_dataset(self, dataset_path, include_name=None, exclude_name=None, ext=None):
        regex = "*" if include_name == None else f"*{include_name}*"
        if ext is not None:
            regex += ext
        dataset_path_ = join(dataset_path, regex)
        include_files = glob(dataset_path_)
        
        # If the train files cannot be specified without including the val file
        # --> affirmatively exclude it
        if exclude_name is not None:
            exclude_files = glob(join(dataset_path, f"*{exclude_name}*"))
            for file in exclude_files:
                if file in include_files:
                    include_files.remove(file)
                    print(f"<LOADCOMMENT> Removed {file.split('/')[-1]} from training")
        
        dataset = load_dataset(
            'parquet',
            data_files={'train': include_files},
            streaming=True
        ).with_format('numpy')

        return dataset, include_files

# Loader object for huggingface datasets
class LoaderHF(LoaderObj):
    def __init__(self, 
        dataset_path: str,
        dictionary_path: str='./dictionary.tsv',
        synonyms: list=None,
        masses_path: str='./',
        test_split_method: str='full_val',
        top_pks: int=100,
        pep_length: list=[0,40],
        reverse: bool=False,
        batch_size: int=100,
        num_workers: int=0,
        custom_columns: list=[],
        datapath_extension="parquet/processed",
        **kwargs
    ):

        self.custom_columns = []
        # Set the max_sequence for the map function
        max_seq = pep_length[1] if pep_length is not None else None
        assert os.path.exists(dataset_path), "Train dataset path doesn't exist"
        # If piping data from scratch directory
        if 'scratch' in kwargs and kwargs['scratch']['use']:
            train_dataset_path = kwargs['scratch']['train_path']
            val_dataset_path = kwargs['scratch']['val_path']

        ##############
        # Dictionary #
        ##############
        self.amod_dic = self.create_sequence_dictionary(dictionary_path)
        if synonyms is not None:
            for pair in synonyms:
                letter_a, letter_b = pair
                self.amod_dic = self.synonym(letter_a, letter_b, self.amod_dic)
        self.amod_dic_rev = self.reverse_dictionary(self.amod_dic)
        
        #####################
        # Dictionary masses #
        #####################
        # - RULES
        #   1. There is a file that matches the regex *masses.tsv in the masses_path
        self.massdic = self.load_token_masses(masses_path)

        ###############
        # Split sizes #
        ###############
        # - RULES
        #   1. There is a file that matches the regex *sizes.tsv in the train_dataset_path and val_dataset_path
        #   2. val_name will pick out 1 file's size from the val_dataset_path
        #self.train_size = self.find_set_size_for_tqdm(join(train_dataset_path, dpe), train_name, val_name, "*species*size*tsv")
        #self.val_size = self.find_set_size_for_tqdm(join(val_dataset_path, dpe), val_name, regex="*species*size*tsv")
        
        #############
        # Tokenizer #
        #############
        # - RULES
        #   1. There is a file named enumerate_tokens.py with a subroutine named
        #      partition_modified_sequence
        self.tokenizer = partition_modified_sequence # self.create_tokenizer('./')

        ###########
        # Dataset #
        ###########
        # - RULES
        #   1. The *_directory_path will contain its data in a directory named "parquet/processed"
        #   2. val_name only has to be somewhere in the filename -> *val_name*
        dataset, train_files = self._load_dataset(dataset_path, None, None, ext='parquet')
        self.size = int(open(join(dataset_path, "size.tsv")).read().split('\t')[1])
        #dataset_val, val_files = self._load_dataset(join(val_dataset_path, dpe), val_name)

        print(f"<LOADCOMMENT> Found {len(train_files)} file(s) for inference")
        
        #########################
        # Map to format outputs #
        #########################
        lambda_function = lambda example: map_fn(
            example,
            tokenizer=self.tokenizer,
            dic=self.amod_dic,
            top=top_pks,
            max_seq=max_seq,
            reverse=reverse,
        )
        if 'remove_columns' in kwargs:
            remove_train_columns = [column for column in kwargs['remove_columns'] if column in dataset['train'].features]
            remove_val_columns = [column for column in kwargs['remove_columns'] if column in dataset['val'].features]
        else:
            remove_train_columns = []
            remove_val_columns = []
        dataset['train'] = dataset['train'].map(
            lambda_function, 
            remove_columns=remove_train_columns,
        )
        
        #############
        # Filtering #
        #############
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['precursor_charge'] >= kwargs['charge'][0]) &
                (example['precursor_charge'] <= kwargs['charge'][1])
            )

        # Filter for length
        if pep_length is not None:
            dataset = dataset.filter(
                lambda example: 
                (len(example['tokenized_sequence']) >= pep_length[0]) &
                (len(example['tokenized_sequence']) <= pep_length[1]) 
                if 'tokenized_sequence' in example else True
            )
        
        self.dataset = dataset

        ###############
        # Dataloaders #
        ###############
        num_workers = min(self.dataset['train'].n_shards, num_workers)
        eval_collate_function = lambda x: collate_fn(x, custom_columns=custom_columns)
        self.dataloader = {
            'test': self.build_dataloader(dataset['train'], batch_size, num_workers, collate_fn),
        }

if __name__ == '__main__':
    loader = LoaderHF(
        "parquet", 
        synonyms=[['I','L']], 
        top_pks=300, 
        reverse=True, 
        batch_size=100, 
        num_workers=int(sys.argv[1])
    )

    for batch in loader.dataloader['test']:
        print(batch)
