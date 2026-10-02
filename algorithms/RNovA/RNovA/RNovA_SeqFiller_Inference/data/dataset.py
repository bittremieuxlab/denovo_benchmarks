import os
import torch
import numpy as np
from torch.utils.data import Dataset
from itertools import combinations_with_replacement
from utils.BasicClass import ResidueOnlyPeptide, Ion, Candidate_Residue_AA
from utils.spectra_pb2 import Spectra

class RnovaDataset(Dataset):
    def __init__(self, cfg, spectra):
        super().__init__()
        self.cfg = cfg
        self.spectra = spectra
        
    def __getitem__(self, idx):
        title = self.spectra[idx]['title']
        precursor_moverz = self.spectra[idx]['precursor_moverz']
        precursor_charge = self.spectra[idx]['precursor_charge']
        precursor_mass = Ion.precursorIonMoverz2ResidueOnlyPeptideMass(precursor_ion_moverz=precursor_moverz, precursor_ion_charge=precursor_charge)
        product_ion_moverz = self.spectra[idx]['product_ion_moverz']
        product_ion_intensity_log = self.spectra[idx]['product_ion_intensity_log']
        
        product_ion_order = np.argsort(-product_ion_intensity_log)
        product_ion_rank = product_ion_order.argsort()
        product_ion_moverz = product_ion_moverz[product_ion_rank<self.cfg.data.ms2_max_peak_count]
        product_ion_intensity_log = product_ion_intensity_log[product_ion_rank<self.cfg.data.ms2_max_peak_count]
        product_ion_rank = product_ion_rank[product_ion_rank<self.cfg.data.ms2_max_peak_count]

        node_mass, node_intensity_log, peak_intensity_rank, peak_moverz, node_class = self.spectrum_preprocess(product_ion_moverz, product_ion_intensity_log, product_ion_rank, precursor_mass, precursor_charge, precursor_moverz)
        return {'node_mass': torch.tensor(node_mass,dtype=torch.float32),
                'node_intensity': torch.tensor(node_intensity_log,dtype=torch.float32),
                'node_class': torch.tensor(node_class,dtype=torch.int64),
                'peak_intensity_rank': torch.tensor(peak_intensity_rank,dtype=torch.int64),
                'peak_moverz': torch.tensor(peak_moverz,dtype=torch.float32),
                'charge': precursor_charge,
                'precursor_mass': precursor_mass,
                'title': title}
        
    def __len__(self):
        return len(self.spectra)

    def spectrum_preprocess(self, product_ion_moverz, product_ion_intensity, product_ion_rank, precursor_mass, precursor_charge, precursor_moverz):
        # node_class:
        # "node_class" refers to the ion type used to predict the nodes.
        # 0 for <pad>
        # 1 for <presursor moverz>
        # 2 for <bos> node(0 Da)
        # 3 for <eos> node(precursor mass)
        # 4 for 1b
        # 5 for 1y
        
        node_class = []
        node_mass = []
        peak_intensity_rank = []
        peak_moverz = []
        node_intensity = []

        #1b ion
        node_class.append(np.ones(product_ion_moverz.size)*4)
        node_mass.append(Ion.peakMz2ResidueMassSum(product_ion_moverz, 'b', 1))
        peak_intensity_rank.append(product_ion_rank)
        node_intensity.append(product_ion_intensity)
        peak_moverz.append(product_ion_moverz)

        #1y ion
        node_class.append(np.ones(product_ion_moverz.size)*5)
        node_mass.append(precursor_mass - Ion.peakMz2ResidueMassSum(product_ion_moverz, 'y', 1))
        peak_intensity_rank.append(product_ion_rank)
        node_intensity.append(product_ion_intensity)
        peak_moverz.append(product_ion_moverz)
        
        node_class = np.concatenate(node_class)
        node_mass = np.concatenate(node_mass)
        peak_intensity_rank = np.concatenate(peak_intensity_rank)
        peak_moverz = np.concatenate(peak_moverz)
        node_intensity = np.concatenate(node_intensity)

        # If node mass greater or equal to precursor_mass or less or equal to 0,
        # that node mass will be deleted. It is optional.
        outbount_mask = np.logical_and(0<node_mass,node_mass<precursor_mass)
        node_mass = node_mass[outbount_mask]
        node_intensity = node_intensity[outbount_mask]
        peak_intensity_rank = peak_intensity_rank[outbount_mask]
        peak_moverz = peak_moverz[outbount_mask]
        node_class = node_class[outbount_mask]

        # Sort the nodes by node_mass
        sort_index = node_mass.argsort()
        node_mass = node_mass[sort_index]
        node_intensity = node_intensity[sort_index]
        peak_intensity_rank = peak_intensity_rank[sort_index]
        peak_moverz = peak_moverz[sort_index]
        node_class = node_class[sort_index]

        # Add <presursor moverz>, <bos> and <eos> node
        node_class = np.concatenate([np.array([1,2]),node_class,np.array([3])])
        node_mass = np.concatenate([np.array([precursor_moverz,0]),node_mass,np.array([precursor_mass])])
        peak_intensity_rank = np.concatenate([np.array([0,0]),peak_intensity_rank,np.array([0])])
        node_intensity = np.concatenate([np.array([node_intensity.max(),node_intensity.max()]),node_intensity,np.array([node_intensity.max()])])
        peak_moverz = np.concatenate([np.array([precursor_moverz,0]),peak_moverz,np.array([precursor_moverz])])

        return node_mass, node_intensity, peak_intensity_rank, peak_moverz, node_class
    
    def lossy_absolute_decompression(self, moverz, absolute_error):
        fixed_point = 0.5/absolute_error
        moverz = moverz.cumsum()
        moverz = moverz/fixed_point
        return moverz

    def lossy_relative_decompression(self, intensity, relative_error):
        fixed_point = 0.5/relative_error
        intensity = intensity.cumsum()
        intensity = intensity/fixed_point
        return intensity