"""
References:
- OpenFold: https://github.com/aqlaboratory/openfold/blob/main/openfold/np/residue_constants.py
"""


import torch
import gemmi
import warnings
import numpy as np
from pathlib import Path
from typing import Any, Tuple, Union, Optional, Union, Dict


__all__ = ['OpenfoldEntity', ]



# To facilitate usage, we cloned AF2.residue_constants to this file:
# arom-wise vocabulary
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.
atom_types2element = {k:k[0] for k in atom_types}

# residue-wise vocabulary
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
restype_num_with_x = len(restypes_with_x)  # := 21.

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
# NB: restype_3to1 differs from Bio.PDB.protein_letters_3to1 by being a simple
# 1-to-1 mapping of 3 letter names to one letter names. The latter contains
# many more, and less common, three letter names as keys and maps many of these
# to the same one letter name (including 'X' and 'U' which we don't use here).
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Note that .mmcif supports flexible chain-id while .pdb only supports A-Z, a-z, 0-9,
pdb_chain_ids = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]
pdb_chain_ids =  list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
pdb_chain_order = {chain_id: i for i, chain_id in enumerate(pdb_chain_ids)}

dtype_template: Dict[str, torch.dtype] = {
    'residue_atom37_coord': torch.float32,
    'residue_atom37_mask':  torch.float32,
    'residue_atom37_bfactor': torch.float32,
    'residue_mask':         torch.float32,
    'residue_aatype':       torch.int32,
    'residue_index':        torch.int32,
    'residue_chain_index':  torch.int32,
}

gemmi_default_protocol = {
    'drop_water':   True,
    'drop_ligand':  True,
    'drop_na':      True,
    'drop_nonstd':  False,
    'aggregate':    True,
}

gemmi_checker = {
    'is_aa': lambda residue: (
                (lambda r: r is not None and r.is_amino_acid())
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_na': lambda residue: (
                (lambda r: r is not None and r.is_nucleic_acid())
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_ligand': lambda residue: (
                (lambda r: r is not None and not (r.is_amino_acid() or r.is_nucleic_acid() or r.is_water()))
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_water': lambda residue: (
                (lambda r: r is not None and r.is_water())
                (gemmi.find_tabulated_residue(residue.name))
            ),
    'is_standard': lambda residue: (
                (lambda r: r is not None and r.is_standard())
                (gemmi.find_tabulated_residue(residue.name))
            ),
}


# TODO if necessary, implement biopython, biotite parser
def _gemmi_parser(
    p: Path,
    protocol: Dict[str, bool] = gemmi_default_protocol
) -> Any:
    
    structure = gemmi.read_structure(str(p))
    if len(structure) > 1:
        warnings.warn(f'Multiple conformation({len(structure)}) detected, taking the first conformation only ...')
    main_model = structure[0]
    
    # DEV: introducing symid, asymid feature & multi-chain cropping
    chain2feature: Dict[str, Dict[str, torch.Tensor]] = {}
    for chain in main_model:
        feature_template = {
            'residue_atom37_coord': [],         # [L, 37, 3]
            'residue_atom37_mask':  [],         # [L, 37]
            'residue_atom37_bfactor': [],       # [L, 37]
            'residue_mask':         [],         # [L]
            'residue_aatype':       [],         # [L]
            'residue_index':        [],         # [L],
            'residue_chain_index':  [],         # [L]
        }
        
        for residue in chain:
            # Default behavior: keep std & non-std aa
            if protocol['drop_water'] and gemmi_checker['is_water'](residue): continue
            if protocol['drop_ligand'] and gemmi_checker['is_ligand'](residue): continue
            if protocol['drop_na'] and gemmi_checker['is_na'](residue): continue
            if protocol['drop_nonstd'] and (not gemmi_checker['is_standard'](residue)): continue
                    
            atom37_coord = torch.zeros((atom_type_num, 3), dtype=dtype_template['residue_atom37_coord'])    # [37, 3]
            atom37_mask = torch.zeros((atom_type_num), dtype=dtype_template['residue_atom37_mask'])         # [37]
            atom37_bfactor = torch.zeros((atom_type_num), dtype=dtype_template['residue_atom37_bfactor'])   # [37]
            for atom in residue:
                if atom.name not in atom_types: continue
                atom37_coord[atom_order[atom.name]] = torch.tensor((atom.pos.x, atom.pos.y, atom.pos.z), dtype=dtype_template['residue_atom37_coord'])
                atom37_mask[atom_order[atom.name]] = 1.0
                atom37_bfactor[atom_order[atom.name]] = atom.b_iso
            
            # atom-wise
            feature_template['residue_atom37_coord'].append(atom37_coord)
            feature_template['residue_atom37_mask'].append(atom37_mask)
            feature_template['residue_atom37_bfactor'].append(atom37_bfactor)
            
            # residue-wise
            restype_idx = restype_order_with_x[restype_3to1.get(residue.name, 'X')]
            feature_template['residue_mask'].append(atom37_mask[1].to(dtype_template['residue_mask']))
            feature_template['residue_aatype'].append(
                torch.tensor(restype_idx, dtype=dtype_template['residue_aatype'])
            )
            feature_template['residue_index'].append(
                torch.tensor(residue.seqid.num, dtype=dtype_template['residue_index'])
            )
            feature_template['residue_chain_index'].append(
                torch.tensor(pdb_chain_order[chain.name], dtype=dtype_template['residue_chain_index'])
            )
        
        if feature_template['residue_atom37_coord'] != []:
            chain2feature[chain.name] = {k: torch.stack(v, dim=0) for k, v in feature_template.items()}
    
    # Concat different chains
    if protocol['aggregate']:
        feature_names = list(next(iter(chain2feature.values())).keys())
        return {
            feature_name: torch.cat([
                chain2feature[chain_id][feature_name] for chain_id in chain2feature
            ], dim=0)
            for feature_name in feature_names
        }
    else:
        return chain2feature
    
    


class OpenfoldEntity:
    """ A protein entity, organized as Dict[str, torch.Tensor] """
    
    def __init__(self):
        self.feature: Dict[str, torch.Tensor] = {}
    
    @classmethod
    def from_file(cls, file_path: Path):
        assert file_path.suffix.lower() in ['.cif', '.mmcif', '.pdb'], f'Unsupported file type: {file_path.suffix}'
        instance = cls()
        instance.feature.update(_gemmi_parser(file_path))
        return instance
    
    @classmethod
    def from_feature(cls, feature: Dict[str, torch.Tensor]):
        # `aatype` only decide the 3-char type field rather than `atom37_mask`
        atom37_coord = feature['residue_atom37_coord']
        atom37_mask = feature['residue_atom37_mask']
        L, device = atom37_coord.size(0), atom37_coord.device
        feature.setdefault('residue_mask', atom37_mask[:, 1]).to(dtype_template['residue_mask'])
        feature.setdefault('residue_atom37_bfactor', torch.zeros(L, atom_type_num, device=device, dtype=dtype_template['residue_atom37_bfactor']))
        feature.setdefault('residue_aatype', torch.zeros(L, device=device, dtype=dtype_template['residue_aatype']))
        feature.setdefault('residue_index', torch.arange(L, device=device, dtype=dtype_template['residue_index']))
        feature.setdefault('residue_chain_index', torch.zeros(L, device=device, dtype=dtype_template['residue_chain_index']))
        instance = cls()
        instance.feature.update(feature)
        return instance
        
    @property
    def shape(self) -> Dict[str, Any]:
        return {k:v.shape for k,v in self.feature.items()}
    
    @property
    def device(self) -> torch.device:
        return self.aatype.device
    
    @property
    def dtype(self) -> Dict[str, Any]:
        return {k:v.dtype for k,v in self.feature.items()}
    
    @property
    def aatype(self) -> torch.Tensor:
        return self.feature['residue_aatype']
    
    @property
    def bfactor(self) -> torch.Tensor:
        return self.feature['residue_bfactor']
    
    @property
    def residue_idx(self) -> torch.Tensor:
        return self.feature['residue_idx']
    
    @property
    def residue_mask(self) -> torch.Tensor:
        return self.feature['residue_mask']
    
    @property
    def ca_coord(self) -> torch.Tensor:
        ca_mask = self.feature['residue_atom37_mask'][:, atom_order['CA']]
        ca_coord = self.feature['residue_atom37_coord'][:, atom_order['CA'], :]
        impute_coord = ca_coord.new_zeros(ca_coord.shape)
        return ca_coord * ca_mask[:, None] + impute_coord * (1 - ca_mask[:, None])
    
    @property
    def cb_coord(self) -> torch.Tensor:
        cb_mask = self.feature['residue_atom37_mask'][:, atom_order['CB']]
        cb_coord = self.feature['residue_atom37_coord'][:, atom_order['CB'], :]
        impute_coord = self.ca_coord
        return cb_coord * cb_mask[:, None] + impute_coord * (1 - cb_mask[:, None])
    
    @property
    def backbone_coord(self) -> torch.Tensor:
        backbone_atom_order = [atom_order['N'], atom_order['CA'], atom_order['C'], atom_order['CB'], atom_order['O']]
        backbone_atom5_coord = self.feature['residue_atom37_coord'][:, backbone_atom_order]
        return backbone_atom5_coord.reshape(-1, 3)
    
    @property
    def ca_contact(self) -> Tuple[torch.Tensor, ...]:
        ca_coord = self.ca_coord
        dist_matrix = torch.cdist(ca_coord, ca_coord)
        contact_map = (dist_matrix < 8.0).float()
        return contact_map, dist_matrix
    
    @property
    def cb_contact(self) -> Tuple[torch.Tensor, ...]:
        cb_coord = self.cb_coord
        dist_matrix = torch.cdist(cb_coord, cb_coord)
        contact_map = (dist_matrix < 8.0).float()
        return contact_map, dist_matrix
    
    def __str__(self):
        res_shortname = np.fromiter(
            map(lambda i: restypes_with_x[i], self.feature['residue_aatype']),
            dtype='U1'
        )
        s = ''.join(res_shortname)
        return s
    
    def __len__(self):
        return self.feature['residue_atom37_coord'].size(0)
    
    def to(self, device: Union[str, torch.device]):
        for k,v in self.feature.items(): self.feature[k] = v.to(device)
        return self
    
    def to_pickle(self, pickle_path: Path):
        torch.save(self.feature, pickle_path)
    
    def to_pdb(self, pdb_path: Path):
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_string = self.to_pdb_string()
        pdb_path.write_text(pdb_string)

    def to_pdb_string(self, model: int = 1, add_end: bool = True) -> str:
        res_1to3 = lambda r: restype_1to3.get(restypes_with_x[r], 'UNK')
        atom37_types = atom_types
        # pdb is a column format
        pdb_lines: list[str] = []
        atom_positions = self.feature['residue_atom37_coord'].cpu().numpy()
        atom_mask = self.feature['residue_atom37_mask'].cpu().numpy()
        residue_index = self.feature['residue_index'].cpu().numpy()
        chain_index = self.feature['residue_chain_index'].cpu().numpy()
        aatype = self.feature['residue_aatype'].cpu().numpy()
        b_factors = self.feature['residue_atom37_bfactor'].cpu().numpy()
        
        # simple format check  
        if np.any(aatype > restype_num):
            raise ValueError('Invalid aatypes.')
        chain_ids = {}
        for i in np.unique(chain_index):  # np.unique gives sorted output
            if (pdb_chain_num := len(pdb_chain_ids)) <= i: raise ValueError(f'The PDB format supports at most {pdb_chain_num} chains.')
            chain_ids[i] = pdb_chain_ids[i]

        # start with MODEL
        pdb_lines.append(f'MODEL     {model}')
        atom_index = 1
        last_chain_index = chain_index[0]
        for i in range(len(self)): # for each residue
            # Close the previous chain if in a multichain pdb
            if last_chain_index != chain_index[i]:
                chain_end = 'TER'
                pdb_lines.append(
                    f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i - 1]):>3} '
                    f'{chain_ids[chain_index[i - 1]]:>1}{residue_index[i - 1]:>4}'
                )
                last_chain_index = chain_index[i]
                atom_index += 1  # Atom index increases at the TER symbol.

            res_name_3 = res_1to3(aatype[i])
            for atom37_name, pos, mask, b_factor in zip(atom37_types, atom_positions[i], atom_mask[i], b_factors[i]): # for atom37
                if mask < 0.5: continue
                record_type = 'ATOM'
                name = atom37_name if len(atom37_name) == 4 else f' {atom37_name}'
                alt_loc = ''
                insertion_code = ''
                occupancy = 1.00
                element = atom_types2element[atom37_name]
                charge = ''
                atom_line = (
                    f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                    f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                    f'{residue_index[i]:>4}{insertion_code:>1}   '
                    f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                    f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                    f'{element:>2}{charge:>2}'
                )
                pdb_lines.append(atom_line)
                atom_index += 1

        # close the final chain
        chain_end = 'TER'
        pdb_lines.append(
            f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[-1]):>3} '
            f'{chain_ids[chain_index[-1]]:>1}{residue_index[-1]:>4}'
        )
        pdb_lines.append('ENDMDL')
        if add_end: pdb_lines.append('END')
        # pad all lines to 80 characters.
        pdb_lines = [line.ljust(80) for line in pdb_lines]
        pdb_string = '\n'.join(pdb_lines) + '\n'  # add terminating newline
        return pdb_string
    