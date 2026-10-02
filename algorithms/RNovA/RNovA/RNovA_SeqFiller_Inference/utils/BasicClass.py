import os
import re
import json
import numbers
import numpy as np
import polars as pl
from dataclasses import dataclass

class Composition():
    __element_mass = {
        # From NIST, "https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&all=all&isotype=some"
        'Neutron': 1.00866491595,
        'Proton': 1.007276466621,
        'Electron': 0.000548579909065,
        'H': 1.00782503223,
        'He':3.0160293201,
        'Li':6.0151228874,
        'Be':9.012183065,
        'B':10.01293695,
        'C': 12,
        'N': 14.00307400443,
        'O': 15.99491461957,
        'F':18.99840316273,
        'Ne':19.9924401762,
        'Na': 22.9897692820,
        'Mg':23.985041697,
        'Al':26.98153853,
        'Si':27.97692653465,
        'P': 30.97376199842,
        'S': 31.9720711744,
        'Cl':34.968852682,
        'Ar':35.967545105,
        'K':38.9637064864,
        'Ca':39.962590863,
        'Sc':44.95590828,
        'Ti':45.95262772,
        'V':49.94715601,
        'Cr':49.94604183,
        'Mn':54.93804391,
        'Fe':53.93960899,
        'Co':58.93319429,
        'Ni':57.93534241,
        'Cu':62.92959772,
        'Zn':63.92914201,
        'Ga':68.9255735,
        'Ge':69.92424875,
        'As':74.92159457,
        'Se':73.922475934,
        'Br':78.9183376,
        'Kr':77.92036494,
        'Rb':84.9117897379,
        'Sr':83.9134191,
        'Y':88.9058403,
        'Zr':89.9046977,
        'Nb':92.9063730,
        'Mo':91.90680796,
        'Tc':96.9063667,
        'Ru':95.90759025,
        'Rh':102.9054980,
        'Pd':101.9056022,
        'Ag':106.9050916,
        'Cd':105.9064599,
        'In':112.90406184,
        'Sn':111.90482387,
        'Sb':120.9038120,
        'Te':119.9040593,
        'I':126.9044719,
        'Xe':123.9058920,
        'Cs':132.9054519610,
        'Ba':129.9063207,
        'La':137.9071149,
        'Ce':135.90712921,
        'Pr':140.9076576,
        'Nd':141.9077290,
        'Pm':144.9127559,
        'Sm':143.9120065,
        'Eu':150.9198578,
        'Gd':151.9197995,
        'Tb':158.9253547,
        'Dy':155.9242847,
        'Ho':164.9303288,
        'Er':161.9287884,
        'Tm':168.9342179,
        'Yb':167.9338896,
        'Lu':174.9407752,
        'Hf':173.9400461,
        'Ta':179.9474648,
        'W':179.9467108,
        'Re':184.9529545,
        'Os':183.9524885,
        'Ir':190.9605893,
        'Pt':189.9599297,
        'Au':196.96656879,
        'Hg':195.9658326,
        'Tl':202.9723446,
        'Pb':203.9730440,
        'Bi':208.9803991,
        'Po':208.9824308,
        'At':209.9871479,
        'Rn':210.9906011,
        'Fr':223.0197360,
        'Ra':223.0185023,
        'Ac':227.0277523,
        'Th':230.0331341,
        'Pa':231.0358842,
        'U':233.0396355,
        'Np':236.046570,
        'Pu':238.0495601,
        'Am':241.0568293,
        'Cm':243.0613893,
        'Bk':247.0703073,
        'Cf':249.0748539,
        'Es':252.082980,
        'Fm':257.0951061,
        'Md':258.0984315,
        'No':259.10103,
        'Lr':262.10961,
        'Rf':267.12179,
        'Db':268.12567,
        'Sg':271.13393,
        'Bh':272.13826,
        'Hs':270.13429,
        'Mt':276.15159,
        'Ds':281.16451,
        'Rg':280.16514,
        'Cn':285.17712,
        'Nh':284.17873,
        'Fl':289.19042,
        'Mc':288.19274,
        'Lv':293.20449,
        'Ts':292.20746,
        'Og':294.21392,
        '[2H]':2.01410177812,
        '[3H]':3.0160492779,
        'D':2.01410177812,
        'T':3.0160492779,
        '[13C]': 13.00335483507,
        '[14C]': 14.0032419884,
        '[15N]': 15.00010889888,
        '[17O]': 16.99913175650,
        '[18O]': 17.99915961286,
    }

    def __init__(self, class_input):
        if type(class_input) == str:
            self.composition = self.parse_chemical_formula(class_input)
        elif type(class_input) == dict:
            self.composition = class_input
        else:
            raise TypeError
        self.mass = self.mass_calculater()
    
    def parse_chemical_formula(self, formula):
        if not formula: 
            raise ValueError("The chemical formula cannot be empty.")
        
        formula = formula.replace("+", "")

        if formula[0]=='-': 
            formula = formula[1:]
            all_minus = True
        else:
            all_minus = False

        pattern = re.compile(r'(\[?\d*[A-Z][a-z]*\]?)([+-]?\d*\.?\d+)?')
        elements = pattern.findall(formula)

        if not elements: 
            raise ValueError("No valid chemical elements found in the formula.")

        parsed_formula = ""
        element_dict = {}
        for element, str_number in elements:
            if element not in self.__element_mass:
                raise ValueError(f"Invalid element symbol: {element}")

            if str_number == '': 
                number = 1.0
            else: 
                number = float(str_number)
            
            if element not in element_dict:
                element_dict[element] = number
            else:
                element_dict[element] += number
            
            # 元素数量匹配到公式中
            if str_number.find('.') != -1: 
                len_str_number = len(str_number)
                decimal_length = len_str_number - str_number.find('.') - 1
                parsed_formula += element + f'{number:0>{len_str_number}.{decimal_length}f}'
            elif str_number!='': 
                parsed_formula += element + str(int(number))
            else: 
                parsed_formula += element
        
        if all_minus: 
            parsed_formula = '-'+parsed_formula
            formula = '-'+formula
            element_dict = {k: -element_dict[k] for k in element_dict}

        # 检查解析后的公式是否与原始公式相符
        if parsed_formula != formula: 
            raise ValueError(f"The chemical formula contains invalid elements or format.\nParsed formula: {parsed_formula}\nOriginal formula: {formula}")

        return element_dict

    def __add__(self, other):
        result = {}
        if isinstance(other, Composition):
            for k in self.composition:
                result.update({k: self.composition[k]})
            for k in other.composition:
                try:
                    result[k] += other.composition[k]
                    if result[k] == 0: result.pop(k)
                except KeyError:
                    result.update({k: other.composition[k]})
            return Composition(result)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        result = {}
        if isinstance(other, Composition):
            for k in self.composition:
                result.update({k: self.composition[k]})
            for k in other.composition:
                try:
                    result[k] -= other.composition[k]
                    if result[k] == 0: result.pop(k)
                except KeyError:
                    result.update({k: -other.composition[k]})
            return Composition(result)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            result = {}
            for k in self.composition:
                result.update({k: other * self.composition[k]})
            return Composition(result)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, Composition):
            return self.composition==other.composition
        else:
            raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, Composition):
            return self.mass>other.mass
        else:
            raise NotImplementedError

    def __ge__(self, other):
        if isinstance(other, Composition):
            return self.mass>=other.mass
        else:
            raise NotImplementedError

    def __lt__(self, other):
        if isinstance(other, Composition):
            return self.mass<other.mass
        else:
            raise NotImplementedError

    def __le__(self, other):
        if isinstance(other, Composition):
            return self.mass<=other.mass
        else:
            raise NotImplementedError
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __hash__(self):
        return hash(json.dumps(self.composition,sort_keys=True))

    def __repr__(self):
        return 'Composition('+str(self.composition)+')'

    def __str__(self):
        return 'Composition('+str(self.composition)+')'
    
    def __getitem__(self, idx):
        return self.composition[idx]

    def mass_calculater(self):
        result = 0
        for k in self.composition:
            if k in self.__element_mass: 
                result += self.composition[k] * self.__element_mass[k]
            else:
                raise ValueError(f"Invalid element symbol: {k}")
        return result

    def comp2formula(self):
        formula=''
        if all([element_num<0 for element_num in self.composition.values()]): 
            all_minus = True
        else: 
            all_minus = False
        
        if all_minus:
            formula+='-'
            for k in self.composition:
                if self.composition[k].is_integer(): 
                    if self.composition[k]==-1: formula+=k
                    else: formula+=k+str(int(-self.composition[k]))
                else: 
                    formula+=k+str(-self.composition[k])
        else:
            for k in self.composition:
                if self.composition[k].is_integer(): 
                    if self.composition[k]==1: formula+=k
                    else: formula+=k+str(int(self.composition[k]))
                else: 
                    formula+=k+str(self.composition[k])
        return formula

@dataclass
class Residual_AA():
    amino_acid_name: str
    n_terminal_PTM: str
    c_terminal_PTM: str
    r_group_PTM: str
    embedding_db_index: int
    composition: Composition
    mass: float
    full_name: str
    
    def __repr__(self):
        return (f"Residual_AA(amino_acid_name={self.amino_acid_name}, "
                f"n_terminal_PTM={self.n_terminal_PTM}, "
                f"c_terminal_PTM={self.c_terminal_PTM}, "
                f"r_group_PTM={self.r_group_PTM}, "
                f"composition={self.composition})")
    
    def __eq__(self, other):
        if not isinstance(other, Residual_AA):
            raise NotImplementedError

        return self.embedding_db_index == other.embedding_db_index
    
    def __ne__(self, other):
        if not isinstance(other, Residual_AA):
            raise NotImplementedError

        return self.embedding_db_index != other.embedding_db_index

    def __gt__(self, other):
        if not isinstance(other, Residual_AA):
            raise NotImplementedError
        return self.embedding_db_index > other.embedding_db_index

    def __lt__(self, other):
        if not isinstance(other, Residual_AA):
            raise NotImplementedError
        return self.embedding_db_index < other.embedding_db_index
   
    def __hash__(self):
        return hash(self.embedding_db_index)

class Candidate_Residue_AA():
    __candidate_residue_aa_with_ptm = {}
    __candidate_residue_aa_with_ptm_int_index = {}
    __instance = None
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(Candidate_Residue_AA, cls).__new__(cls)
            cls.__instance.read_aa_ptm_formula_database()
        return cls.__instance
    
    def __init__(self) -> None: pass
    
    def read_aa_ptm_formula_database(self):
        cur_path = os.path.abspath(os.path.dirname(__file__))
        ptm_aa_db = pl.read_csv(os.path.join(cur_path,'AA_PTM_Mol_Formula'))
        for i, (name, mol_formula) in enumerate(ptm_aa_db.iter_rows(),start=3):
            # 0 for <pad>
            # 1 for <bos_f>
            # 2 for <bos_r>
            amino_acid_name = name.split('|')[0]
            try: r_group_PTM = name.split('|')[1]
            except IndexError: r_group_PTM = ''
            try: n_terminal_PTM = name.split('|')[2]
            except IndexError: n_terminal_PTM = ''
            try: c_terminal_PTM = name.split('|')[3]
            except IndexError: c_terminal_PTM = ''
            self.__candidate_residue_aa_with_ptm[name] = \
                Residual_AA(amino_acid_name,
                            n_terminal_PTM=n_terminal_PTM,
                            c_terminal_PTM=c_terminal_PTM,
                            r_group_PTM=r_group_PTM,
                            embedding_db_index=i,
                            composition=Composition(mol_formula)-Composition('H2O'),
                            mass=(Composition(mol_formula)-Composition('H2O')).mass,
                            full_name=name)
            self.__candidate_residue_aa_with_ptm_int_index[i] = \
                Residual_AA(amino_acid_name,
                            n_terminal_PTM=n_terminal_PTM,
                            c_terminal_PTM=c_terminal_PTM,
                            r_group_PTM=r_group_PTM,
                            embedding_db_index=i,
                            composition=Composition(mol_formula)-Composition('H2O'),
                            mass=(Composition(mol_formula)-Composition('H2O')).mass,
                            full_name=name)
            # print(self.__candidate_residue_aa_with_ptm.keys())
    
    @classmethod
    def add_residue(cls, name: str, residue: Residual_AA):
        """添加新的氨基酸残基及其PTM信息到字典中"""
        cls.__candidate_residue_aa_with_ptm[name] = residue
        cls.__candidate_residue_aa_with_ptm_int_index[len(cls.__candidate_residue_aa_with_ptm_int_index)+3] = residue

    @classmethod
    def remove_residue(cls, name: str):
        """从字典中删除指定的氨基酸残基"""
        if name in cls.__candidate_residue_aa_with_ptm:
            del cls.__candidate_residue_aa_with_ptm[name]

    @classmethod
    def update_residue(cls, name: str, residue: Residual_AA):
        """更新字典中的氨基酸残基信息"""
        cls.__candidate_residue_aa_with_ptm[name] = residue
    
    def __repr__(self):
        """提供详细的类表示，适用于调试目的。"""
        residues_info = []
        for key, residue in self.__candidate_residue_aa_with_ptm.items():
            residue_info = (f"{key}: \n" 
                            f"Amino Acid Name: {residue.amino_acid_name}, \n"
                            f"N-Terminal PTM: {residue.n_terminal_PTM}, \n"
                            f"C-Terminal PTM: {residue.c_terminal_PTM}, \n"
                            f"R-Group PTM: {residue.r_group_PTM}, \n"
                            f"Composition: {residue.composition}"
                            f"Embedding DataBase Index: {residue.embedding_db_index}")
            residues_info.append(residue_info)
        residues_info = '\n'.join(residues_info)
        return f"Candidate_Residue_AA:\n{residues_info}"

    def __str__(self):
        """提供简单的类描述，适用于更一般的打印目的。"""
        residues_info = "\n".join([f"{key}: {value.composition}" for key, value in self.__candidate_residue_aa_with_ptm.items()])
        return f"Candidate_Residue_AA:\n{residues_info}"
    
    def __getitem__(self, key):
        """获取特定键的值"""
        if type(key)==int: return self.__candidate_residue_aa_with_ptm_int_index[key]
        else: return self.__candidate_residue_aa_with_ptm[key]
    
    def __iter__(self):
        """迭代字典"""
        return iter(self.__candidate_residue_aa_with_ptm)

    def __len__(self):
        """返回字典长度"""
        return len(self.__candidate_residue_aa_with_ptm)

    def __contains__(self, key):
        """检查键是否在字典中"""
        return key in self.__candidate_residue_aa_with_ptm
    
    def keys(self):
        return self.__candidate_residue_aa_with_ptm.keys()
    
    def values(self):
        return self.__candidate_residue_aa_with_ptm.values()

class ResidueOnlyPeptide():
    __aa_residue = Candidate_Residue_AA()
    def __init__(self, seq, mods=''):
        if '[' in seq or ']' in seq:
            self.sequence_residue_seq = []
            ptm_block_flag = False
            for char in seq:
                if char=='[':
                    ptm_block_flag = True
                    ptm_str = ''
                elif char==']':
                    if ptm_str == '':
                        raise ValueError('PTM cannot be empty.')
                    ptm_group = ptm_str.split('|')
                    ptm_group = ptm_group + ['']*(3-len(ptm_group))
                    self.sequence_residue_seq[-1][1:] = ptm_group
                    ptm_str = ''
                    ptm_block_flag = False
                elif not ptm_block_flag:
                    self.sequence_residue_seq.append([char, '', '', ''])
                else:
                    ptm_str += char

        else:
            self.sequence_residue_seq = [[aa,'','',''] for aa in seq]
            if mods:
                if isinstance(mods, str):
                    for mod in mods.split(';'):
                        mod_pos = mod[:mod.find('|')]
                        if mod_pos=='N':
                            mod_pos_int = 0
                            if self.sequence_residue_seq[mod_pos_int][2] != '':
                                raise ValueError('N-term can only have one PTM.')
                            self.sequence_residue_seq[mod_pos_int][2] = mod[mod.find('|')+1:]
                        elif mod_pos=='C':
                            mod_pos_int = -1
                            if self.sequence_residue_seq[mod_pos_int][3] != '':
                                raise ValueError('C-term can only have one PTM.')
                            self.sequence_residue_seq[mod_pos_int][3] = mod[mod.find('|')+1:]
                        else:
                            mod_pos_int = int(mod_pos)
                            if self.sequence_residue_seq[mod_pos_int][1] != '':
                                raise ValueError('R group can only have one PTM.')
                            self.sequence_residue_seq[mod_pos_int][1] = mod[mod.find('|')+1:]
                else:
                    raise NotImplementedError
        self.sequence_residue_seq = ['|'.join(aa_ptm_list).strip('|') for aa_ptm_list in self.sequence_residue_seq]
        for aa in self.sequence_residue_seq:
            if aa not in self.__aa_residue.keys():
                new_temp_aa = Residual_AA(aa[:aa.find('|')],
                                          n_terminal_PTM='',
                                          c_terminal_PTM='',
                                          r_group_PTM=aa[aa.find('|')+1:-1],
                                          embedding_db_index=len(self.__aa_residue)+3,
                                          composition=None,
                                          mass=float(aa[aa.find('|')+1:-1])+self.__aa_residue[aa[0]].mass,
                                          full_name=aa)
                self.__aa_residue.add_residue(aa,new_temp_aa)
        self.sequence_residue = [self.__aa_residue[aa_seq] for aa_seq in self.sequence_residue_seq]
        # self.sequence_residue_compositions = [aa_residual.composition for aa_residual in self.sequence_residue]
        self.sequence_residue_mass = np.array([aa_residual_composition.mass for aa_residual_composition in self.sequence_residue])
        # self.prefix_compositions = np.cumsum(self.sequence_residue_compositions).tolist()
        # self.suffix_compositions = np.cumsum(self.sequence_residue_compositions[::-1]).tolist()
        # self.prefix_mass = np.array([aa_residual_composition.mass for aa_residual_composition in self.prefix_compositions])
        # self.suffix_mass = np.array([aa_residual_composition.mass for aa_residual_composition in self.suffix_compositions])
        self.prefix_mass = np.cumsum(self.sequence_residue_mass)
        self.suffix_mass = np.cumsum(self.sequence_residue_mass[::-1])
        self.seq = self.sequence_residue_seq
        # self.total_residue_composition = sum(self.sequence_residue_compositions)
        self.total_residue_mass = sum(self.sequence_residue_mass)

    def __repr__(self):
        return str(self.seq)

    def __str__(self):
        return str(self.seq)

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, index):
        return self.sequence_residue[index]

    @classmethod
    def reset_aadict(cls,newAAdict):
        cls.__aa_residue_composition = newAAdict

    @classmethod
    def remove_from_aadict(cls, keys):
        for key in keys:
            cls.__aa_residue_composition.pop(key)

    @classmethod
    def add_to_aadict(cls, additional_AAcomps):
        for additional_AAcomp in additional_AAcomps:
            cls.__aa_residue_composition.update(additional_AAcomp)

    @classmethod
    def output_aalist(cls):
        return list(cls.__aa_residue_composition.keys())

    @classmethod
    def output_aadict(cls):
        return cls.__aa_residue_composition

    @classmethod
    def seqs2composition_list(cls,seq):
        return [cls.__aa_residue_composition[aa] for aa in seq]
    
    @classmethod
    def seqs2massmap(cls,seq):
        return [cls.__aa_residue_composition[aa].mass for aa in seq]

class Ion():
    # Ion offset design from http://www.matrixscience.com/help/fragmentation_help.html 
    # Part: Formulae to Calculate Fragment Ion m/z values
    __ion_offset = {
        'a': Composition('-CHO'),
        'a-NH3': Composition('-CHO') + Composition('-NH3'),
        'a-H2O': Composition('-CHO') + Composition('-H2O'),
        'b': Composition('-H'),
        'b-NH3': Composition('-H') + Composition('-NH3'),
        'b-H2O': Composition('-H') + Composition('-H2O'),
        'c': Composition('NH2'),
        'x': Composition('CO') + Composition('-H'),
        'y': Composition('H'),
        'y-NH3': Composition('H') + Composition('-NH3'),
        'y-H2O': Composition('H') + Composition('-H2O'),
        'z': Composition('-NH2')
    }

    __term_ion_offset = {
        'a': Composition('-CHO') + Composition('H'),
        'a-NH3': Composition('-CHO') + Composition('-NH3') + Composition('H'),
        'a-H2O': Composition('-CHO') + Composition('-H2O') + Composition('H'),
        'b': Composition('-H') + Composition('H'),
        'b-NH3': Composition('-H') + Composition('-NH3') + Composition('H'),
        'b-H2O': Composition('-H') + Composition('-H2O') + Composition('H'),
        'c': Composition('NH2') + Composition('H'),
        'x': Composition('CO') + Composition('-H') + Composition('OH'),
        'y': Composition('H') + Composition('OH'),
        'y-NH3': Composition('H') + Composition('-NH3') + Composition('OH'),
        'y-H2O': Composition('H') + Composition('-H2O') + Composition('OH'),
        'z': Composition('-NH2') + Composition('OH')
    }

    @classmethod
    def peakMz2ResidueMassSum(cls, peak_mz, ion, charge):
        return (peak_mz-Composition('Proton').mass)*charge-cls.__term_ion_offset[ion].mass

    @classmethod
    def peptide2PeakMz(cls, seq, ion, charge):
        if type(seq) != ResidueOnlyPeptide: raise TypeError('Seq must be object of ResidueOnlyPeptide class')
        ion_compsition = seq.total_residue_composition
        ion_compsition = ion_compsition+cls.__term_ion_offset[ion]
        ion_compsition = ion_compsition+Composition('Proton')*charge
        ion_mass = ion_compsition.mass/charge
        return ion_mass
    
    @classmethod
    def comp2PeakMz(cls, ion_compsition, ion, charge):
        if type(ion_compsition) != Composition: raise TypeError('Seq must be object of Composition class')
        ion_compsition = ion_compsition+cls.__term_ion_offset[ion]
        ion_compsition = ion_compsition+Composition('Proton')*charge
        ion_mass = ion_compsition.mass/charge
        return ion_mass
    
    @classmethod
    def peptideMass2PeakMz(cls, seqmass, ion, charge):
        ionmass_without_charge = seqmass + cls.__term_ion_offset[ion].mass
        seqmz = ionmass_without_charge/charge + Composition('Proton').mass
        return seqmz

    @classmethod
    def precursorIonMoverz2ResidueOnlyPeptideMass(cls, precursor_ion_moverz, precursor_ion_charge):
        return precursor_ion_moverz*precursor_ion_charge-Composition('Proton').mass*precursor_ion_charge-Composition('H2O').mass
    
    @classmethod
    def add_ion(cls,ion_comps):
        for ion_comp in ion_comps:
            cls.__ion_offset.update(ion_comp)
        cls.set_ionoffset_endterm()

    @classmethod
    def remove_ion(cls, keys):
        for key in keys:
            cls.__ion_offset.pop(key)
        cls.set_ionoffset_endterm()

    @classmethod
    def reset_ions(cls, ion_comps):
        cls.__ion_offset = ion_comps
        cls.set_ionoffset_endterm()

    @classmethod
    def output_ions(cls):
        return list(cls.__ion_offset.keys())

