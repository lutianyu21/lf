from pathlib import Path
import warnings


__all__ = ['GlobalConstants']



class GlobalConstants:
    
    # TODO facilitate transfer between clusters
    working_dir = Path('/GenSIvePFS/users/lutianyu/lf')
    
    @classmethod
    def auto_numeric(cls, split_name: str) -> int:
        return {
            'p2s/dev':              0,
            'p2s/cameo2022':        1,
            'p2s/casp15':           2,
            'p2s/casp16':           3,
            'psps/dev':             10,
            'psps/cameo2022':       11,
            'psps/casp15':          12,
            'psps/casp16':          13,
        }[split_name]
    
    
    @classmethod
    def auto_string(cls, split_id: int) -> str:
        return {
            0:  'folding/dev',
            1:  'folding/cameo2022',
            2:  'folding/casp15',
            3:  'folding/casp16',
            10: 'cfolding/dev',
            11: 'cfolding/cameo2022',
            12: 'cfolding/casp15',
            13: 'cfolding/casp16',
        }[split_id]
    
    
    @classmethod
    def auto_pathing(cls, extended_uniref_accession: str) -> Path:
        # make sure there's no suffix:
        if '.' in extended_uniref_accession:
            warnings.warn(f'extended_uniref_accession {extended_uniref_accession} contains a dot (.) character. '
                          f'This may lead to unexpected behavior.')
            extended_uniref_accession = extended_uniref_accession.split('.')[0]
        
        # standardize the input format:
        if extended_uniref_accession.upper().startswith('AF-'):
            if extended_uniref_accession.startswith('af-'):
                warnings.warn(f'extended_uniref_accession {extended_uniref_accession} starts with lowercase "af-". '
                            f'Automatically converting to uppercase "AF-".')
                tmp_lower = extended_uniref_accession.lower().split('-')
                tmp_upper = extended_uniref_accession.upper().split('-')
                tmp_standard = tmp_upper[:-1] + [tmp_lower[-1]]
                extended_uniref_accession = '-'.join(tmp_standard)
        elif extended_uniref_accession.startswith('T'):
            pass
        else:
            if '_' in extended_uniref_accession:
                warnings.warn(f'extended_uniref_accession {extended_uniref_accession} contains unexpected characters ("_"). '
                              f'This may lead to unexpected behavior.')
                extended_uniref_accession = extended_uniref_accession.replace('_', '@')
            extended_uniref_accession = extended_uniref_accession[0:4].lower() + extended_uniref_accession[4:]   
        
        # valid formats:
        # - RCSB-style: 1ema, 1ema%1, 1ema@A
        # - AFDB-style: AF-A0A2U5YMB9-F1-model_v4
        # - casp-style: T1234
        if extended_uniref_accession.startswith('AF-'):
            # swissprot_v4
            database_swissprot_v4 = GlobalConstants.working_dir / 'data/swissprot_v4/raw'
            path = database_swissprot_v4 / f'{extended_uniref_accession}.cif.gz'
            if path.exists(): return path
            # unicluster40
            database_unicluster40 = GlobalConstants.working_dir / 'data/unicluster40/raw'
            path = database_unicluster40 / f'{extended_uniref_accession}.cif.gz'
            if path.exists(): return path
        elif extended_uniref_accession.startswith('T'):
            # casp15
            database_casp15 = GlobalConstants.working_dir / 'data/casp15/raw'
            path = database_casp15 / f'{extended_uniref_accession}.pdb'
            if path.exists(): return path
            # casp16
            database_casp16 = GlobalConstants.working_dir / 'data/casp16/raw'
            path = database_casp16 / f'{extended_uniref_accession}.pdb'
            if path.exists(): return path
        else:
            # rcsb
            database_rcsb = GlobalConstants.working_dir / 'data/rcsb/raw'
            path = database_rcsb / f'{extended_uniref_accession[0:4]}.cif'
            if path.exists(): return database_rcsb / f'{extended_uniref_accession}.cif'
        
        raise FileNotFoundError(f'Cannot find the file for extended_uniref_accession: {extended_uniref_accession}')
