"""
Chemical Motif Analysis Module - Extended Version

Uses RDKit to identify functional groups in SMILES, including 50+ categories and 200+ functional groups.

References:
- RDKit SMARTS documentation
- Daylight SMARTS theory
- PubChem substructure fingerprints
- MACCS Keys
- Functional Group Filters (RDKit)
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import os

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not installed, will use simplified regex matching")
    print("Install RDKit: conda install -c conda-forge rdkit")


class ChemicalMotifAnalyzer:
    """Chemical Motif Analyzer - Identifies 50+ categories and 200+ functional groups"""
    
    def __init__(self):
        self._init_target_patterns()
        if RDKIT_AVAILABLE:
            self._compile_smarts()
    
    def _init_target_patterns(self):
        """初始化目标官能团的SMARTS模式"""
        
        # 目标官能团SMARTS定义 - 扩展版
        self.target_functional_groups = {

            # ============================================================
            # 1. AMINES (胺类)
            # ============================================================
            'primary_amine': {
                'smarts': '[NX3;H2;!$(NC=O);!$(N-S)]',
                'name': 'Primary Amine (伯胺)',
                'description': '-NH2 group not in amide or sulfonamide',
                'category': 'Amines'
            },
            'secondary_amine': {
                'smarts': '[NX3;H1;!$(NC=O);!$(N-S)]([#6])[#6]',
                'name': 'Secondary Amine (仲胺)',
                'description': '-NH- group not in amide',
                'category': 'Amines'
            },
            'tertiary_amine': {
                'smarts': '[NX3;H0;!$(NC=O);!$(N-S);!$([N](-*)=*)]([#6])([#6])[#6]',
                'name': 'Tertiary Amine (叔胺)',
                'description': '-N< group (trisubstituted)',
                'category': 'Amines'
            },
            'quaternary_ammonium': {
                'smarts': '[NX4+]([#6])([#6])([#6])[#6]',
                'name': 'Quaternary Ammonium (季铵)',
                'description': 'N+ with four carbon substituents',
                'category': 'Amines'
            },
            'aromatic_amine': {
                'smarts': '[NX3;H2]c',
                'name': 'Aromatic Amine (芳香胺)',
                'description': 'Amine attached to aromatic ring',
                'category': 'Amines'
            },
            'aliphatic_amine': {
                'smarts': '[NX3;H2][CX4]',
                'name': 'Aliphatic Amine (脂肪胺)',
                'description': 'Amine attached to sp3 carbon',
                'category': 'Amines'
            },
            'enamine': {
                'smarts': '[NX3][CX3]=[CX3]',
                'name': 'Enamine (烯胺)',
                'description': 'N-C=C structure',
                'category': 'Amines'
            },
            'imine': {
                'smarts': '[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]',
                'name': 'Imine (亚胺)',
                'description': 'C=N-R structure',
                'category': 'Amines'
            },
            'oxime': {
                'smarts': '[CX3]=[NX2][OX2H1]',
                'name': 'Oxime (肟)',
                'description': 'C=N-OH structure',
                'category': 'Amines'
            },
            'hydroxylamine': {
                'smarts': '[NX3][OX2H1]',
                'name': 'Hydroxylamine (羟胺)',
                'description': 'N-OH structure',
                'category': 'Amines'
            },

            # ============================================================
            # 2. ALCOHOLS (醇类)
            # ============================================================
            'primary_alcohol': {
                'smarts': '[CX4;H2][OX2H1]',
                'name': 'Primary Alcohol (伯醇)',
                'description': 'R-CH2-OH',
                'category': 'Alcohols'
            },
            'secondary_alcohol': {
                'smarts': '[CX4;H1]([#6])([#6])[OX2H1]',
                'name': 'Secondary Alcohol (仲醇)',
                'description': 'R2CH-OH',
                'category': 'Alcohols'
            },
            'tertiary_alcohol': {
                'smarts': '[CX4;H0]([#6])([#6])([#6])[OX2H1]',
                'name': 'Tertiary Alcohol (叔醇)',
                'description': 'R3C-OH',
                'category': 'Alcohols'
            },
            'allylic_alcohol': {
                'smarts': '[CX3]=[CX3][CX4][OX2H1]',
                'name': 'Allylic Alcohol (烯丙醇)',
                'description': 'C=C-C-OH',
                'category': 'Alcohols'
            },
            'benzylic_alcohol': {
                'smarts': 'c[CX4][OX2H1]',
                'name': 'Benzylic Alcohol (苄醇)',
                'description': 'Ar-CH2-OH',
                'category': 'Alcohols'
            },
            'propargylic_alcohol': {
                'smarts': '[CX2]#[CX2][CX4][OX2H1]',
                'name': 'Propargylic Alcohol (炔丙醇)',
                'description': 'C≡C-C-OH',
                'category': 'Alcohols'
            },
            'diol_1_2': {
                'smarts': '[OX2H1][CX4][CX4][OX2H1]',
                'name': '1,2-Diol (1,2-二醇)',
                'description': 'Vicinal diol',
                'category': 'Alcohols'
            },
            'diol_1_3': {
                'smarts': '[OX2H1][CX4][CX4][CX4][OX2H1]',
                'name': '1,3-Diol (1,3-二醇)',
                'description': '1,3-diol',
                'category': 'Alcohols'
            },
            'hemiacetal': {
                'smarts': '[OX2H1][CX4]([#6])[OX2][#6]',
                'name': 'Hemiacetal (半缩醛)',
                'description': 'R-CH(OH)(OR)',
                'category': 'Alcohols'
            },

            # ============================================================
            # 3. ETHERS (醚类)
            # ============================================================
            'ether': {
                'smarts': '[OD2]([#6])[#6]',
                'name': 'Ether (醚)',
                'description': 'R-O-R structure',
                'category': 'Ethers'
            },
            'aliphatic_ether': {
                'smarts': '[CX4][OX2][CX4]',
                'name': 'Aliphatic Ether (脂肪醚)',
                'description': 'C-O-C (sp3)',
                'category': 'Ethers'
            },
            'aromatic_ether': {
                'smarts': 'c[OX2][#6]',
                'name': 'Aromatic Ether (芳香醚)',
                'description': 'Ar-O-R',
                'category': 'Ethers'
            },
            'vinyl_ether': {
                'smarts': '[CX3]=[CX3][OX2][#6]',
                'name': 'Vinyl Ether (乙烯基醚)',
                'description': 'C=C-O-R',
                'category': 'Ethers'
            },
            'acetal': {
                'smarts': '[CX4]([OX2][#6])([OX2][#6])[#6,H]',
                'name': 'Acetal (缩醛)',
                'description': 'R-CH(OR)2',
                'category': 'Ethers'
            },
            'ketal': {
                'smarts': '[CX4]([OX2][#6])([OX2][#6])([#6])[#6]',
                'name': 'Ketal (缩酮)',
                'description': 'R2C(OR)2',
                'category': 'Ethers'
            },
            'crown_ether_unit': {
                'smarts': '[OX2][CX4][CX4][OX2]',
                'name': 'Crown Ether Unit (冠醚单元)',
                'description': 'O-C-C-O pattern',
                'category': 'Ethers'
            },

            # ============================================================
            # 4. ALDEHYDES (醛类)
            # ============================================================
            'aldehyde': {
                'smarts': '[CX3H1](=O)[#6]',
                'name': 'Aldehyde (醛)',
                'description': 'R-CHO',
                'category': 'Aldehydes'
            },
            'aromatic_aldehyde': {
                'smarts': 'c[CX3H1](=O)',
                'name': 'Aromatic Aldehyde (芳香醛)',
                'description': 'Ar-CHO',
                'category': 'Aldehydes'
            },
            'alpha_beta_unsaturated_aldehyde': {
                'smarts': '[CX3]=[CX3][CX3H1](=O)',
                'name': 'α,β-Unsaturated Aldehyde (α,β-不饱和醛)',
                'description': 'C=C-CHO',
                'category': 'Aldehydes'
            },
            'formyl': {
                'smarts': '[CH1](=O)',
                'name': 'Formyl (甲酰基)',
                'description': '-CHO group',
                'category': 'Aldehydes'
            },

            # ============================================================
            # 5. KETONES (酮类)
            # ============================================================
            'ketone': {
                'smarts': '[#6][CX3](=O)[#6]',
                'name': 'Ketone (酮)',
                'description': 'R-CO-R',
                'category': 'Ketones'
            },
            'aromatic_ketone': {
                'smarts': 'c[CX3](=O)[#6]',
                'name': 'Aromatic Ketone (芳香酮)',
                'description': 'Ar-CO-R',
                'category': 'Ketones'
            },
            'alpha_beta_unsaturated_ketone': {
                'smarts': '[CX3]=[CX3][CX3](=O)[#6]',
                'name': 'α,β-Unsaturated Ketone (α,β-不饱和酮)',
                'description': 'C=C-CO-R (enone)',
                'category': 'Ketones'
            },
            'diketone_1_2': {
                'smarts': '[CX3](=O)[CX3](=O)',
                'name': '1,2-Diketone (1,2-二酮)',
                'description': 'R-CO-CO-R',
                'category': 'Ketones'
            },
            'diketone_1_3': {
                'smarts': '[CX3](=O)[CX4][CX3](=O)',
                'name': '1,3-Diketone (1,3-二酮)',
                'description': 'R-CO-CH2-CO-R',
                'category': 'Ketones'
            },
            'cyclic_ketone': {
                'smarts': '[R][CX3](=O)[R]',
                'name': 'Cyclic Ketone (环酮)',
                'description': 'Ketone in ring',
                'category': 'Ketones'
            },
            'quinone': {
                'smarts': '[#6]1[#6][#6](=O)[#6][#6][#6]1=O',
                'name': 'Quinone (醌)',
                'description': 'Cyclic diketone',
                'category': 'Ketones'
            },

            # ============================================================
            # 6. CARBOXYLIC ACIDS (羧酸)
            # ============================================================
            'carboxylic_acid': {
                'smarts': '[CX3](=O)[OX2H1]',
                'name': 'Carboxylic Acid (羧酸)',
                'description': 'R-COOH',
                'category': 'Carboxylic_Acids'
            },
            'aromatic_carboxylic_acid': {
                'smarts': 'c[CX3](=O)[OX2H1]',
                'name': 'Aromatic Carboxylic Acid (芳香羧酸)',
                'description': 'Ar-COOH',
                'category': 'Carboxylic_Acids'
            },
            'alpha_amino_acid': {
                'smarts': '[NX3][CX4][CX3](=O)[OX2H1]',
                'name': 'α-Amino Acid (α-氨基酸)',
                'description': 'H2N-CHR-COOH',
                'category': 'Carboxylic_Acids'
            },
            'dicarboxylic_acid': {
                'smarts': '[CX3](=O)[OX2H1].[CX3](=O)[OX2H1]',
                'name': 'Dicarboxylic Acid (二元羧酸)',
                'description': 'HOOC-R-COOH',
                'category': 'Carboxylic_Acids'
            },
            'carboxylate': {
                'smarts': '[CX3](=O)[O-]',
                'name': 'Carboxylate (羧酸根)',
                'description': 'R-COO-',
                'category': 'Carboxylic_Acids'
            },

            # ============================================================
            # 7. ESTERS (酯类)
            # ============================================================
            'ester': {
                'smarts': '[#6][CX3](=O)[OX2][#6]',
                'name': 'Ester (酯)',
                'description': 'R-COO-R',
                'category': 'Esters'
            },
            'aromatic_ester': {
                'smarts': 'c[CX3](=O)[OX2][#6]',
                'name': 'Aromatic Ester (芳香酯)',
                'description': 'Ar-COO-R',
                'category': 'Esters'
            },
            'phenyl_ester': {
                'smarts': '[#6][CX3](=O)[OX2]c',
                'name': 'Phenyl Ester (苯酯)',
                'description': 'R-COO-Ar',
                'category': 'Esters'
            },
            'formate_ester': {
                'smarts': '[CH1](=O)[OX2][#6]',
                'name': 'Formate Ester (甲酸酯)',
                'description': 'H-COO-R',
                'category': 'Esters'
            },
            'acetate_ester': {
                'smarts': '[CH3][CX3](=O)[OX2][#6]',
                'name': 'Acetate Ester (乙酸酯)',
                'description': 'CH3-COO-R',
                'category': 'Esters'
            },
            'carbonate': {
                'smarts': '[OX2][CX3](=O)[OX2]',
                'name': 'Carbonate (碳酸酯)',
                'description': 'R-O-CO-O-R',
                'category': 'Esters'
            },
            'orthoester': {
                'smarts': '[CX4]([OX2])([OX2])[OX2]',
                'name': 'Orthoester (原酸酯)',
                'description': 'RC(OR)3',
                'category': 'Esters'
            },

            # ============================================================
            # 8. AMIDES (酰胺)
            # ============================================================
            'amide': {
                'smarts': '[CX3](=O)[NX3]',
                'name': 'Amide (酰胺)',
                'description': 'R-CO-NR2',
                'category': 'Amides'
            },
            'primary_amide': {
                'smarts': '[CX3](=O)[NX3;H2]',
                'name': 'Primary Amide (伯酰胺)',
                'description': 'R-CO-NH2',
                'category': 'Amides'
            },
            'secondary_amide': {
                'smarts': '[CX3](=O)[NX3;H1][#6]',
                'name': 'Secondary Amide (仲酰胺)',
                'description': 'R-CO-NH-R',
                'category': 'Amides'
            },
            'tertiary_amide': {
                'smarts': '[CX3](=O)[NX3;H0]([#6])[#6]',
                'name': 'Tertiary Amide (叔酰胺)',
                'description': 'R-CO-NR2',
                'category': 'Amides'
            },
            'formamide': {
                'smarts': '[CH1](=O)[NX3]',
                'name': 'Formamide (甲酰胺)',
                'description': 'H-CO-NR2',
                'category': 'Amides'
            },
            'acetamide': {
                'smarts': '[CH3][CX3](=O)[NX3]',
                'name': 'Acetamide (乙酰胺)',
                'description': 'CH3-CO-NR2',
                'category': 'Amides'
            },
            'peptide_bond': {
                'smarts': '[NX3][CX4][CX3](=O)[NX3]',
                'name': 'Peptide Bond (肽键)',
                'description': 'N-C-CO-N',
                'category': 'Amides'
            },
            'imide': {
                'smarts': '[CX3](=O)[NX3][CX3](=O)',
                'name': 'Imide (酰亚胺)',
                'description': 'R-CO-N-CO-R',
                'category': 'Amides'
            },
            'cyclic_imide': {
                'smarts': '[R][CX3](=O)[NX3][CX3](=O)[R]',
                'name': 'Cyclic Imide (环酰亚胺)',
                'description': 'Cyclic imide (succinimide, phthalimide)',
                'category': 'Amides'
            },
            'hydroxamic_acid': {
                'smarts': '[CX3](=O)[NX3][OX2H1]',
                'name': 'Hydroxamic Acid (异羟肟酸)',
                'description': 'R-CO-NH-OH',
                'category': 'Amides'
            },

            # ============================================================
            # 9. NITRILES (腈类)
            # ============================================================
            'nitrile': {
                'smarts': '[CX2]#[NX1]',
                'name': 'Nitrile (腈)',
                'description': 'R-C≡N',
                'category': 'Nitriles'
            },
            'aromatic_nitrile': {
                'smarts': 'c[CX2]#[NX1]',
                'name': 'Aromatic Nitrile (芳香腈)',
                'description': 'Ar-C≡N',
                'category': 'Nitriles'
            },
            'isonitrile': {
                'smarts': '[NX2]#[CX2]',
                'name': 'Isonitrile (异腈)',
                'description': 'R-N≡C',
                'category': 'Nitriles'
            },
            'cyanate': {
                'smarts': '[OX2][CX2]#[NX1]',
                'name': 'Cyanate (氰酸酯)',
                'description': 'R-O-C≡N',
                'category': 'Nitriles'
            },
            'thiocyanate': {
                'smarts': '[SX2][CX2]#[NX1]',
                'name': 'Thiocyanate (硫氰酸酯)',
                'description': 'R-S-C≡N',
                'category': 'Nitriles'
            },
            'isothiocyanate': {
                'smarts': '[NX2]=[CX2]=[SX1]',
                'name': 'Isothiocyanate (异硫氰酸酯)',
                'description': 'R-N=C=S',
                'category': 'Nitriles'
            },

            # ============================================================
            # 10. NITRO (硝基)
            # ============================================================
            'nitro': {
                'smarts': '[NX3+](=O)[O-]',
                'name': 'Nitro (硝基)',
                'description': 'R-NO2',
                'category': 'Nitro'
            },
            'aromatic_nitro': {
                'smarts': 'c[NX3+](=O)[O-]',
                'name': 'Aromatic Nitro (芳香硝基)',
                'description': 'Ar-NO2',
                'category': 'Nitro'
            },
            'aliphatic_nitro': {
                'smarts': '[CX4][NX3+](=O)[O-]',
                'name': 'Aliphatic Nitro (脂肪硝基)',
                'description': 'R-CH2-NO2',
                'category': 'Nitro'
            },
            'nitroso': {
                'smarts': '[NX2]=[OX1]',
                'name': 'Nitroso (亚硝基)',
                'description': 'R-N=O',
                'category': 'Nitro'
            },
            'nitrate_ester': {
                'smarts': '[OX2][NX3+](=O)[O-]',
                'name': 'Nitrate Ester (硝酸酯)',
                'description': 'R-O-NO2',
                'category': 'Nitro'
            },
            'nitrite_ester': {
                'smarts': '[OX2][NX2]=[OX1]',
                'name': 'Nitrite Ester (亚硝酸酯)',
                'description': 'R-O-N=O',
                'category': 'Nitro'
            },

            # ============================================================
            # 11. TRIAZOLE (三唑)
            # ============================================================
            '1_2_3_triazole': {
                'smarts': 'c1nn[nH]c1',
                'name': '1,2,3-Triazole (1,2,3-三唑)',
                'description': '5-membered ring with 3 adjacent N atoms',
                'category': 'Triazole'
            },
            '1_2_4_triazole': {
                'smarts': 'c1ncnn1',
                'name': '1,2,4-Triazole (1,2,4-三唑)',
                'description': '5-membered ring with N at 1,2,4 positions',
                'category': 'Triazole'
            },
            '1_2_3_triazole_substituted': {
                'smarts': 'c1nnn(*)c1',
                'name': '1,2,3-Triazole N-substituted',
                'description': 'N-substituted 1,2,3-triazole',
                'category': 'Triazole'
            },
            'benzotriazole': {
                'smarts': 'c1ccc2[nH]nnc2c1',
                'name': 'Benzotriazole (苯并三唑)',
                'description': 'Fused benzene-triazole',
                'category': 'Triazole'
            },

            # ============================================================
            # 12. IMIDAZOLE (咪唑)
            # ============================================================
            'imidazole': {
                'smarts': 'c1c[nH]cn1',
                'name': 'Imidazole (咪唑)',
                'description': '5-membered ring with N at 1,3 positions',
                'category': 'Imidazole'
            },
            'imidazole_substituted': {
                'smarts': 'c1cn(*)cn1',
                'name': 'N-substituted Imidazole',
                'description': 'N1-substituted imidazole',
                'category': 'Imidazole'
            },
            'benzimidazole': {
                'smarts': 'c1ccc2[nH]cnc2c1',
                'name': 'Benzimidazole (苯并咪唑)',
                'description': 'Fused benzene-imidazole',
                'category': 'Imidazole'
            },
            'imidazoline': {
                'smarts': 'C1CN=CN1',
                'name': 'Imidazoline (咪唑啉)',
                'description': 'Partially saturated imidazole',
                'category': 'Imidazole'
            },
            'alkylimidazole': {
                'smarts': '[CH3,CH2]n1ccnc1',
                'name': 'N-Alkylimidazole (N-烷基咪唑)',
                'description': 'Imidazole with N-alkyl substituent',
                'category': 'Imidazole'
            },
            'methylimidazole': {
                'smarts': 'Cn1ccnc1',
                'name': 'N-Methylimidazole (N-甲基咪唑)',
                'description': 'Imidazole with N-methyl',
                'category': 'Imidazole'
            },

            # ============================================================
            # 13. PYRIDINE (吡啶)
            # ============================================================
            'pyridine': {
                'smarts': 'c1ccncc1',
                'name': 'Pyridine (吡啶)',
                'description': '6-membered aromatic ring with N',
                'category': 'Pyridine'
            },
            'pyridine_2_substituted': {
                'smarts': 'c1ccnc(*)c1',
                'name': '2-substituted Pyridine',
                'description': 'Pyridine with substituent at position 2',
                'category': 'Pyridine'
            },
            'pyridine_3_substituted': {
                'smarts': 'c1cc(*)cnc1',
                'name': '3-substituted Pyridine',
                'description': 'Pyridine with substituent at position 3',
                'category': 'Pyridine'
            },
            'pyridine_4_substituted': {
                'smarts': 'c1cc(*)ccn1',
                'name': '4-substituted Pyridine',
                'description': 'Pyridine with substituent at position 4',
                'category': 'Pyridine'
            },
            'pyridine_n_oxide': {
                'smarts': '[n+]1([O-])ccccc1',
                'name': 'Pyridine N-oxide (吡啶N-氧化物)',
                'description': 'Pyridine with N-oxide',
                'category': 'Pyridine'
            },
            'bipyridine': {
                'smarts': 'c1ccncc1-c2ccccn2',
                'name': 'Bipyridine (联吡啶)',
                'description': 'Two connected pyridine rings',
                'category': 'Pyridine'
            },

            # ============================================================
            # 14. PYRIMIDINE (嘧啶)
            # ============================================================
            'pyrimidine': {
                'smarts': 'c1cncnc1',
                'name': 'Pyrimidine (嘧啶)',
                'description': '6-membered ring with N at 1,3 positions',
                'category': 'Pyrimidine'
            },
            'pyrimidine_2_amino': {
                'smarts': 'Nc1ncccn1',
                'name': '2-Aminopyrimidine (2-氨基嘧啶)',
                'description': 'Pyrimidine with 2-amino group',
                'category': 'Pyrimidine'
            },
            'pyrimidine_4_amino': {
                'smarts': 'Nc1ccncn1',
                'name': '4-Aminopyrimidine (4-氨基嘧啶)',
                'description': 'Pyrimidine with 4-amino group',
                'category': 'Pyrimidine'
            },
            'uracil': {
                'smarts': 'O=c1cc[nH]c(=O)[nH]1',
                'name': 'Uracil (尿嘧啶)',
                'description': 'Pyrimidine-2,4-dione',
                'category': 'Pyrimidine'
            },
            'thymine': {
                'smarts': 'Cc1c[nH]c(=O)[nH]c1=O',
                'name': 'Thymine (胸腺嘧啶)',
                'description': '5-methyluracil',
                'category': 'Pyrimidine'
            },
            'cytosine': {
                'smarts': 'Nc1cc[nH]c(=O)n1',
                'name': 'Cytosine (胞嘧啶)',
                'description': '4-aminopyrimidin-2-one',
                'category': 'Pyrimidine'
            },

            # ============================================================
            # 15. PYRAZINE & PYRIDAZINE (吡嗪 & 哒嗪)
            # ============================================================
            'pyrazine': {
                'smarts': 'c1cnccn1',
                'name': 'Pyrazine (吡嗪)',
                'description': '6-membered ring with N at 1,4 positions',
                'category': 'Pyrazine'
            },
            'pyridazine': {
                'smarts': 'c1ccnnc1',
                'name': 'Pyridazine (哒嗪)',
                'description': '6-membered ring with N at 1,2 positions',
                'category': 'Pyrazine'
            },
            'quinoxaline': {
                'smarts': 'c1ccc2nccnc2c1',
                'name': 'Quinoxaline (喹喔啉)',
                'description': 'Fused benzene-pyrazine',
                'category': 'Pyrazine'
            },
            'phthalazine': {
                'smarts': 'c1ccc2cnncc2c1',
                'name': 'Phthalazine (酞嗪)',
                'description': 'Fused benzene-pyridazine',
                'category': 'Pyrazine'
            },

            # ============================================================
            # 16. PYRROLE (吡咯)
            # ============================================================
            'pyrrole': {
                'smarts': 'c1cc[nH]c1',
                'name': 'Pyrrole (吡咯)',
                'description': '5-membered aromatic ring with NH',
                'category': 'Pyrrole'
            },
            'n_substituted_pyrrole': {
                'smarts': 'c1ccn(*)c1',
                'name': 'N-substituted Pyrrole (N-取代吡咯)',
                'description': 'Pyrrole with N-substituent',
                'category': 'Pyrrole'
            },
            'indole': {
                'smarts': 'c1ccc2[nH]ccc2c1',
                'name': 'Indole (吲哚)',
                'description': 'Fused benzene-pyrrole',
                'category': 'Pyrrole'
            },
            'indole_3_substituted': {
                'smarts': 'c1ccc2[nH]cc(*)c2c1',
                'name': '3-substituted Indole (3-取代吲哚)',
                'description': 'Indole with 3-substituent',
                'category': 'Pyrrole'
            },
            'carbazole': {
                'smarts': 'c1ccc2c(c1)[nH]c3ccccc23',
                'name': 'Carbazole (咔唑)',
                'description': 'Dibenzopyrrole',
                'category': 'Pyrrole'
            },
            'pyrrolidine': {
                'smarts': 'C1CCNC1',
                'name': 'Pyrrolidine (吡咯烷)',
                'description': 'Saturated 5-membered ring with N',
                'category': 'Pyrrole'
            },
            'proline': {
                'smarts': 'C1C[C@H](NC1)C(=O)O',
                'name': 'Proline (脯氨酸)',
                'description': 'Pyrrolidine amino acid',
                'category': 'Pyrrole'
            },

            # ============================================================
            # 17. PYRAZOLE (吡唑)
            # ============================================================
            'pyrazole': {
                'smarts': 'c1cc[nH]n1',
                'name': 'Pyrazole (吡唑)',
                'description': '5-membered ring with adjacent N atoms',
                'category': 'Pyrazole'
            },
            'n_substituted_pyrazole': {
                'smarts': 'c1ccn(*)n1',
                'name': 'N-substituted Pyrazole (N-取代吡唑)',
                'description': 'Pyrazole with N-substituent',
                'category': 'Pyrazole'
            },
            'indazole': {
                'smarts': 'c1ccc2[nH]ncc2c1',
                'name': 'Indazole (吲唑)',
                'description': 'Fused benzene-pyrazole',
                'category': 'Pyrazole'
            },
            'pyrazoline': {
                'smarts': 'C1CC=NN1',
                'name': 'Pyrazoline (吡唑啉)',
                'description': 'Partially saturated pyrazole',
                'category': 'Pyrazole'
            },

            # ============================================================
            # 18. OXAZOLE & ISOXAZOLE (噁唑 & 异噁唑)
            # ============================================================
            'oxazole': {
                'smarts': 'c1cocn1',
                'name': 'Oxazole (噁唑)',
                'description': '5-membered ring with O and N',
                'category': 'Oxazole'
            },
            'isoxazole': {
                'smarts': 'c1conc1',
                'name': 'Isoxazole (异噁唑)',
                'description': '5-membered ring with O-N bond',
                'category': 'Oxazole'
            },
            'benzoxazole': {
                'smarts': 'c1ccc2ocnc2c1',
                'name': 'Benzoxazole (苯并噁唑)',
                'description': 'Fused benzene-oxazole',
                'category': 'Oxazole'
            },
            'benzisoxazole': {
                'smarts': 'c1ccc2oncc2c1',
                'name': 'Benzisoxazole (苯并异噁唑)',
                'description': 'Fused benzene-isoxazole',
                'category': 'Oxazole'
            },
            'oxazoline': {
                'smarts': 'C1COC=N1',
                'name': 'Oxazoline (噁唑啉)',
                'description': 'Partially saturated oxazole',
                'category': 'Oxazole'
            },
            'oxadiazole_1_2_4': {
                'smarts': 'c1nnco1',
                'name': '1,2,4-Oxadiazole (1,2,4-噁二唑)',
                'description': '5-membered ring with O and 2N',
                'category': 'Oxazole'
            },
            'oxadiazole_1_3_4': {
                'smarts': 'c1nnoc1',
                'name': '1,3,4-Oxadiazole (1,3,4-噁二唑)',
                'description': '5-membered ring with O and 2N',
                'category': 'Oxazole'
            },

            # ============================================================
            # 19. THIAZOLE (噻唑)
            # ============================================================
            'thiazole': {
                'smarts': 'c1cscn1',
                'name': 'Thiazole (噻唑)',
                'description': '5-membered ring with S and N',
                'category': 'Thiazole'
            },
            'isothiazole': {
                'smarts': 'c1csnc1',
                'name': 'Isothiazole (异噻唑)',
                'description': '5-membered ring with S-N bond',
                'category': 'Thiazole'
            },
            'benzothiazole': {
                'smarts': 'c1ccc2scnc2c1',
                'name': 'Benzothiazole (苯并噻唑)',
                'description': 'Fused benzene-thiazole',
                'category': 'Thiazole'
            },
            'thiazoline': {
                'smarts': 'C1CSC=N1',
                'name': 'Thiazoline (噻唑啉)',
                'description': 'Partially saturated thiazole',
                'category': 'Thiazole'
            },
            'thiadiazole_1_2_3': {
                'smarts': 'c1nnsn1',
                'name': '1,2,3-Thiadiazole (1,2,3-噻二唑)',
                'description': '5-membered ring with S and 2N',
                'category': 'Thiazole'
            },
            'thiadiazole_1_2_4': {
                'smarts': 'c1nncs1',
                'name': '1,2,4-Thiadiazole (1,2,4-噻二唑)',
                'description': '5-membered ring with S and 2N',
                'category': 'Thiazole'
            },
            'thiadiazole_1_3_4': {
                'smarts': 'c1nnsc1',
                'name': '1,3,4-Thiadiazole (1,3,4-噻二唑)',
                'description': '5-membered ring with S and 2N',
                'category': 'Thiazole'
            },

            # ============================================================
            # 20. QUINOLINE & ISOQUINOLINE (喹啉 & 异喹啉)
            # ============================================================
            'quinoline': {
                'smarts': 'c1ccc2ncccc2c1',
                'name': 'Quinoline (喹啉)',
                'description': 'Fused benzene-pyridine',
                'category': 'Quinoline'
            },
            'isoquinoline': {
                'smarts': 'c1ccc2cnccc2c1',
                'name': 'Isoquinoline (异喹啉)',
                'description': 'Fused benzene-pyridine isomer',
                'category': 'Quinoline'
            },
            'quinazoline': {
                'smarts': 'c1ccc2ncncc2c1',
                'name': 'Quinazoline (喹唑啉)',
                'description': 'Fused benzene-pyrimidine',
                'category': 'Quinoline'
            },
            'cinnoline': {
                'smarts': 'c1ccc2nncc2c1',
                'name': 'Cinnoline (噌啉)',
                'description': 'Fused benzene-pyridazine',
                'category': 'Quinoline'
            },
            'acridine': {
                'smarts': 'c1ccc2nc3ccccc3cc2c1',
                'name': 'Acridine (吖啶)',
                'description': 'Tricyclic N-heterocycle',
                'category': 'Quinoline'
            },
            'phenanthridine': {
                'smarts': 'c1ccc2c(c1)ccc3ncccc23',
                'name': 'Phenanthridine (菲啶)',
                'description': 'Tricyclic N-heterocycle',
                'category': 'Quinoline'
            },

            # ============================================================
            # 21. PURINE (嘌呤)
            # ============================================================
            'purine': {
                'smarts': 'c1ncc2[nH]cnc2n1',
                'name': 'Purine (嘌呤)',
                'description': 'Fused pyrimidine-imidazole',
                'category': 'Purine'
            },
            'adenine': {
                'smarts': 'Nc1ncnc2[nH]cnc12',
                'name': 'Adenine (腺嘌呤)',
                'description': '6-aminopurine',
                'category': 'Purine'
            },
            'guanine': {
                'smarts': 'Nc1nc2[nH]cnc2c(=O)[nH]1',
                'name': 'Guanine (鸟嘌呤)',
                'description': '2-amino-6-oxopurine',
                'category': 'Purine'
            },
            'hypoxanthine': {
                'smarts': 'O=c1[nH]cnc2[nH]cnc12',
                'name': 'Hypoxanthine (次黄嘌呤)',
                'description': '6-oxopurine',
                'category': 'Purine'
            },
            'xanthine': {
                'smarts': 'O=c1[nH]c(=O)c2[nH]cnc2[nH]1',
                'name': 'Xanthine (黄嘌呤)',
                'description': '2,6-dioxopurine',
                'category': 'Purine'
            },
            'caffeine': {
                'smarts': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
                'name': 'Caffeine (咖啡因)',
                'description': '1,3,7-trimethylxanthine',
                'category': 'Purine'
            },

            # ============================================================
            # 22. PIPERIDINE & PIPERAZINE (哌啶 & 哌嗪)
            # ============================================================
            'piperidine': {
                'smarts': 'C1CCNCC1',
                'name': 'Piperidine (哌啶)',
                'description': 'Saturated 6-membered ring with N',
                'category': 'Piperidine'
            },
            'n_substituted_piperidine': {
                'smarts': 'C1CCN(*)CC1',
                'name': 'N-substituted Piperidine (N-取代哌啶)',
                'description': 'Piperidine with N-substituent',
                'category': 'Piperidine'
            },
            'piperazine': {
                'smarts': 'C1CNCCN1',
                'name': 'Piperazine (哌嗪)',
                'description': '6-membered ring with 2 N atoms',
                'category': 'Piperidine'
            },
            'n_substituted_piperazine': {
                'smarts': 'C1CN(*)CCN1',
                'name': 'N-substituted Piperazine (N-取代哌嗪)',
                'description': 'Piperazine with N-substituent',
                'category': 'Piperidine'
            },
            'morpholine': {
                'smarts': 'C1COCCN1',
                'name': 'Morpholine (吗啉)',
                'description': '6-membered ring with O and N',
                'category': 'Piperidine'
            },
            'thiomorpholine': {
                'smarts': 'C1CSCCN1',
                'name': 'Thiomorpholine (硫代吗啉)',
                'description': '6-membered ring with S and N',
                'category': 'Piperidine'
            },
            'tetrahydropyran': {
                'smarts': 'C1CCOCC1',
                'name': 'Tetrahydropyran (四氢吡喃)',
                'description': 'Saturated 6-membered ring with O',
                'category': 'Piperidine'
            },

            # ============================================================
            # 23. THIOPHENE (噻吩)
            # ============================================================
            'thiophene': {
                'smarts': 'c1ccsc1',
                'name': 'Thiophene (噻吩)',
                'description': '5-membered aromatic ring with S',
                'category': 'Thiophene'
            },
            'thiophene_substituted': {
                'smarts': 's1cccc1',
                'name': 'Thiophene (substituted)',
                'description': 'Substituted thiophene ring',
                'category': 'Thiophene'
            },
            'benzothiophene': {
                'smarts': 'c1ccc2sccc2c1',
                'name': 'Benzothiophene (苯并噻吩)',
                'description': 'Fused benzene-thiophene',
                'category': 'Thiophene'
            },
            'dibenzothiophene': {
                'smarts': 'c1ccc2c(c1)sc3ccccc23',
                'name': 'Dibenzothiophene (二苯并噻吩)',
                'description': 'Tricyclic S-heterocycle',
                'category': 'Thiophene'
            },
            'thienyl': {
                'smarts': '[cH1]1[cH1][cH1]sc1',
                'name': 'Thienyl (噻吩基)',
                'description': 'Thiophene substituent',
                'category': 'Thiophene'
            },

            # ============================================================
            # 24. THIOL & THIOETHER (硫醇 & 硫醚)
            # ============================================================
            'thiol': {
                'smarts': '[SX2H1]',
                'name': 'Thiol (硫醇)',
                'description': 'R-SH',
                'category': 'Sulfur'
            },
            'aliphatic_thiol': {
                'smarts': '[CX4][SX2H1]',
                'name': 'Aliphatic Thiol (脂肪硫醇)',
                'description': 'R-CH2-SH',
                'category': 'Sulfur'
            },
            'aromatic_thiol': {
                'smarts': 'c[SX2H1]',
                'name': 'Aromatic Thiol (芳香硫醇)',
                'description': 'Ar-SH',
                'category': 'Sulfur'
            },
            'thioether': {
                'smarts': '[#6][SX2][#6]',
                'name': 'Thioether (硫醚)',
                'description': 'R-S-R',
                'category': 'Sulfur'
            },
            'disulfide': {
                'smarts': '[SX2][SX2]',
                'name': 'Disulfide (二硫键)',
                'description': 'R-S-S-R',
                'category': 'Sulfur'
            },
            'thioester': {
                'smarts': '[CX3](=O)[SX2][#6]',
                'name': 'Thioester (硫酯)',
                'description': 'R-CO-S-R',
                'category': 'Sulfur'
            },
            'dithiocarbamate': {
                'smarts': '[NX3][CX3](=S)[SX2]',
                'name': 'Dithiocarbamate (二硫代氨基甲酸酯)',
                'description': 'R2N-CS-S-R',
                'category': 'Sulfur'
            },
            'thioamide': {
                'smarts': '[CX3](=S)[NX3]',
                'name': 'Thioamide (硫代酰胺)',
                'description': 'R-CS-NR2',
                'category': 'Sulfur'
            },
            'thiourea': {
                'smarts': '[NX3][CX3](=S)[NX3]',
                'name': 'Thiourea (硫脲)',
                'description': 'R2N-CS-NR2',
                'category': 'Sulfur'
            },

            # ============================================================
            # 25. SULFOXIDE & SULFONE (亚砜 & 砜)
            # ============================================================
            'sulfoxide': {
                'smarts': '[#6][SX3](=O)[#6]',
                'name': 'Sulfoxide (亚砜)',
                'description': 'R-SO-R',
                'category': 'Sulfur'
            },
            'sulfone': {
                'smarts': '[#6][SX4](=O)(=O)[#6]',
                'name': 'Sulfone (砜)',
                'description': 'R-SO2-R',
                'category': 'Sulfur'
            },
            'sulfonamide': {
                'smarts': '[SX4](=O)(=O)[NX3]',
                'name': 'Sulfonamide (磺酰胺)',
                'description': 'R-SO2-NR2',
                'category': 'Sulfur'
            },
            'primary_sulfonamide': {
                'smarts': '[SX4](=O)(=O)[NX3;H2]',
                'name': 'Primary Sulfonamide (伯磺酰胺)',
                'description': 'R-SO2-NH2',
                'category': 'Sulfur'
            },
            'sulfonic_acid': {
                'smarts': '[SX4](=O)(=O)[OX2H1]',
                'name': 'Sulfonic Acid (磺酸)',
                'description': 'R-SO3H',
                'category': 'Sulfur'
            },
            'sulfonate_ester': {
                'smarts': '[SX4](=O)(=O)[OX2][#6]',
                'name': 'Sulfonate Ester (磺酸酯)',
                'description': 'R-SO2-O-R',
                'category': 'Sulfur'
            },
            'sulfonyl_chloride': {
                'smarts': '[SX4](=O)(=O)[Cl]',
                'name': 'Sulfonyl Chloride (磺酰氯)',
                'description': 'R-SO2-Cl',
                'category': 'Sulfur'
            },
            'sulfate_ester': {
                'smarts': '[OX2][SX4](=O)(=O)[OX2]',
                'name': 'Sulfate Ester (硫酸酯)',
                'description': 'R-O-SO2-O-R',
                'category': 'Sulfur'
            },
            'sulfinic_acid': {
                'smarts': '[SX3](=O)[OX2H1]',
                'name': 'Sulfinic Acid (亚磺酸)',
                'description': 'R-SO-OH',
                'category': 'Sulfur'
            },
            'sulfenic_acid': {
                'smarts': '[SX2][OX2H1]',
                'name': 'Sulfenic Acid (次磺酸)',
                'description': 'R-S-OH',
                'category': 'Sulfur'
            },

            # ============================================================
            # 26. FURAN & OXYGEN HETEROCYCLES (呋喃 & 含氧杂环)
            # ============================================================
            'furan': {
                'smarts': 'c1ccoc1',
                'name': 'Furan (呋喃)',
                'description': '5-membered aromatic ring with O',
                'category': 'Furan'
            },
            'furan_substituted': {
                'smarts': 'o1cccc1',
                'name': 'Furan (substituted)',
                'description': 'Substituted furan ring',
                'category': 'Furan'
            },
            'benzofuran': {
                'smarts': 'c1ccc2occc2c1',
                'name': 'Benzofuran (苯并呋喃)',
                'description': 'Fused benzene-furan',
                'category': 'Furan'
            },
            'dibenzofuran': {
                'smarts': 'c1ccc2c(c1)oc3ccccc23',
                'name': 'Dibenzofuran (二苯并呋喃)',
                'description': 'Tricyclic O-heterocycle',
                'category': 'Furan'
            },
            'tetrahydrofuran': {
                'smarts': 'C1CCOC1',
                'name': 'Tetrahydrofuran (四氢呋喃)',
                'description': 'Saturated 5-membered ring with O',
                'category': 'Furan'
            },
            'dioxolane': {
                'smarts': 'C1OCOC1',
                'name': '1,3-Dioxolane (1,3-二氧戊环)',
                'description': '5-membered ring with 2 O atoms',
                'category': 'Furan'
            },
            'dioxane': {
                'smarts': 'C1COCOC1',
                'name': '1,4-Dioxane (1,4-二噁烷)',
                'description': '6-membered ring with 2 O atoms',
                'category': 'Furan'
            },
            'benzodioxole': {
                'smarts': 'c1ccc2OCOc2c1',
                'name': '1,3-Benzodioxole (苯并二氧杂环戊烯)',
                'description': 'Methylenedioxy benzene',
                'category': 'Furan'
            },
            'benzodioxane': {
                'smarts': 'c1ccc2OCCOc2c1',
                'name': '1,4-Benzodioxane (苯并二噁烷)',
                'description': 'Fused benzene-dioxane',
                'category': 'Furan'
            },
            'chromene': {
                'smarts': 'c1ccc2c(c1)C=COC2',
                'name': 'Chromene (色烯)',
                'description': 'Benzopyran',
                'category': 'Furan'
            },
            'coumarin': {
                'smarts': 'c1ccc2c(c1)ccc(=O)o2',
                'name': 'Coumarin (香豆素)',
                'description': 'Benzopyran-2-one',
                'category': 'Furan'
            },
            'chromone': {
                'smarts': 'c1ccc2c(c1)oc(=O)cc2',
                'name': 'Chromone (色酮)',
                'description': 'Benzopyran-4-one',
                'category': 'Furan'
            },
            'flavone': {
                'smarts': 'c1ccc(-c2cc(=O)c3ccccc3o2)cc1',
                'name': 'Flavone (黄酮)',
                'description': '2-phenylchromone',
                'category': 'Furan'
            },

            # ============================================================
            # 27. EPOXIDE & LACTONE & LACTAM (环氧 & 内酯 & 内酰胺)
            # ============================================================
            'epoxide': {
                'smarts': 'C1OC1',
                'name': 'Epoxide (环氧)',
                'description': '3-membered ring with O',
                'category': 'Epoxide'
            },
            'aziridine': {
                'smarts': 'C1NC1',
                'name': 'Aziridine (氮丙啶)',
                'description': '3-membered ring with N',
                'category': 'Epoxide'
            },
            'thiirane': {
                'smarts': 'C1SC1',
                'name': 'Thiirane (硫丙环)',
                'description': '3-membered ring with S',
                'category': 'Epoxide'
            },
            'beta_lactone': {
                'smarts': 'C1CC(=O)O1',
                'name': 'β-Lactone (β-内酯)',
                'description': '4-membered lactone',
                'category': 'Epoxide'
            },
            'gamma_lactone': {
                'smarts': 'C1CCC(=O)O1',
                'name': 'γ-Lactone (γ-内酯)',
                'description': '5-membered lactone',
                'category': 'Epoxide'
            },
            'delta_lactone': {
                'smarts': 'C1CCCC(=O)O1',
                'name': 'δ-Lactone (δ-内酯)',
                'description': '6-membered lactone',
                'category': 'Epoxide'
            },
            'beta_lactam': {
                'smarts': 'C1CC(=O)N1',
                'name': 'β-Lactam (β-内酰胺)',
                'description': '4-membered lactam',
                'category': 'Epoxide'
            },
            'gamma_lactam': {
                'smarts': 'C1CCC(=O)N1',
                'name': 'γ-Lactam (γ-内酰胺)',
                'description': '5-membered lactam (pyrrolidinone)',
                'category': 'Epoxide'
            },
            'delta_lactam': {
                'smarts': 'C1CCCC(=O)N1',
                'name': 'δ-Lactam (δ-内酰胺)',
                'description': '6-membered lactam (piperidinone)',
                'category': 'Epoxide'
            },
            'caprolactam': {
                'smarts': 'C1CCCCC(=O)N1',
                'name': 'Caprolactam (己内酰胺)',
                'description': '7-membered lactam',
                'category': 'Epoxide'
            },

            # ============================================================
            # 28. HALOGENS (卤素)
            # ============================================================
            'fluorine': {
                'smarts': '[F]',
                'name': 'Fluorine (氟)',
                'description': 'F atom',
                'category': 'Halogens'
            },
            'chlorine': {
                'smarts': '[Cl]',
                'name': 'Chlorine (氯)',
                'description': 'Cl atom',
                'category': 'Halogens'
            },
            'bromine': {
                'smarts': '[Br]',
                'name': 'Bromine (溴)',
                'description': 'Br atom',
                'category': 'Halogens'
            },
            'iodine': {
                'smarts': '[I]',
                'name': 'Iodine (碘)',
                'description': 'I atom',
                'category': 'Halogens'
            },
            'aryl_fluoride': {
                'smarts': 'c[F]',
                'name': 'Aryl Fluoride (芳基氟)',
                'description': 'Ar-F',
                'category': 'Halogens'
            },
            'aryl_chloride': {
                'smarts': 'c[Cl]',
                'name': 'Aryl Chloride (芳基氯)',
                'description': 'Ar-Cl',
                'category': 'Halogens'
            },
            'aryl_bromide': {
                'smarts': 'c[Br]',
                'name': 'Aryl Bromide (芳基溴)',
                'description': 'Ar-Br',
                'category': 'Halogens'
            },
            'aryl_iodide': {
                'smarts': 'c[I]',
                'name': 'Aryl Iodide (芳基碘)',
                'description': 'Ar-I',
                'category': 'Halogens'
            },
            'alkyl_fluoride': {
                'smarts': '[CX4][F]',
                'name': 'Alkyl Fluoride (烷基氟)',
                'description': 'R-F',
                'category': 'Halogens'
            },
            'alkyl_chloride': {
                'smarts': '[CX4][Cl]',
                'name': 'Alkyl Chloride (烷基氯)',
                'description': 'R-Cl',
                'category': 'Halogens'
            },
            'alkyl_bromide': {
                'smarts': '[CX4][Br]',
                'name': 'Alkyl Bromide (烷基溴)',
                'description': 'R-Br',
                'category': 'Halogens'
            },
            'alkyl_iodide': {
                'smarts': '[CX4][I]',
                'name': 'Alkyl Iodide (烷基碘)',
                'description': 'R-I',
                'category': 'Halogens'
            },
            'trifluoromethyl': {
                'smarts': '[CX4](F)(F)F',
                'name': 'Trifluoromethyl (三氟甲基)',
                'description': '-CF3',
                'category': 'Halogens'
            },
            'difluoromethyl': {
                'smarts': '[CX4](F)F',
                'name': 'Difluoromethyl (二氟甲基)',
                'description': '-CHF2',
                'category': 'Halogens'
            },
            'trifluoromethoxy': {
                'smarts': '[OX2][CX4](F)(F)F',
                'name': 'Trifluoromethoxy (三氟甲氧基)',
                'description': '-O-CF3',
                'category': 'Halogens'
            },
            'acyl_chloride': {
                'smarts': '[CX3](=O)[Cl]',
                'name': 'Acyl Chloride (酰氯)',
                'description': 'R-CO-Cl',
                'category': 'Halogens'
            },
            'acyl_fluoride': {
                'smarts': '[CX3](=O)[F]',
                'name': 'Acyl Fluoride (酰氟)',
                'description': 'R-CO-F',
                'category': 'Halogens'
            },
            'gem_dihalide': {
                'smarts': '[CX4]([F,Cl,Br,I])([F,Cl,Br,I])',
                'name': 'Geminal Dihalide (偕二卤)',
                'description': 'R-CX2-R',
                'category': 'Halogens'
            },
            'vinyl_halide': {
                'smarts': '[CX3]=[CX3][F,Cl,Br,I]',
                'name': 'Vinyl Halide (乙烯基卤)',
                'description': 'C=C-X',
                'category': 'Halogens'
            },

            # ============================================================
            # 29. ALKENE & ALKYNE (烯烃 & 炔烃)
            # ============================================================
            'alkene': {
                'smarts': '[CX3]=[CX3]',
                'name': 'Alkene (烯烃)',
                'description': 'C=C double bond',
                'category': 'Unsaturated'
            },
            'terminal_alkene': {
                'smarts': '[CX3]=[CH2]',
                'name': 'Terminal Alkene (末端烯)',
                'description': 'R-CH=CH2',
                'category': 'Unsaturated'
            },
            'cis_alkene': {
                'smarts': '[#6]/[CX3]=[CX3]/[#6]',
                'name': 'Cis Alkene (顺式烯)',
                'description': 'Z-alkene',
                'category': 'Unsaturated'
            },
            'trans_alkene': {
                'smarts': '[#6]/[CX3]=[CX3]\\[#6]',
                'name': 'Trans Alkene (反式烯)',
                'description': 'E-alkene',
                'category': 'Unsaturated'
            },
            'conjugated_diene': {
                'smarts': '[CX3]=[CX3][CX3]=[CX3]',
                'name': 'Conjugated Diene (共轭二烯)',
                'description': 'C=C-C=C',
                'category': 'Unsaturated'
            },
            'allene': {
                'smarts': '[CX3]=[CX2]=[CX3]',
                'name': 'Allene (丙二烯)',
                'description': 'C=C=C',
                'category': 'Unsaturated'
            },
            'alkyne': {
                'smarts': '[CX2]#[CX2]',
                'name': 'Alkyne (炔烃)',
                'description': 'C≡C triple bond',
                'category': 'Unsaturated'
            },
            'terminal_alkyne': {
                'smarts': '[CX2]#[CH1]',
                'name': 'Terminal Alkyne (末端炔)',
                'description': 'R-C≡C-H',
                'category': 'Unsaturated'
            },
            'internal_alkyne': {
                'smarts': '[#6][CX2]#[CX2][#6]',
                'name': 'Internal Alkyne (内炔)',
                'description': 'R-C≡C-R',
                'category': 'Unsaturated'
            },
            'enyne': {
                'smarts': '[CX3]=[CX3][CX2]#[CX2]',
                'name': 'Enyne (烯炔)',
                'description': 'C=C-C≡C',
                'category': 'Unsaturated'
            },
            'vinyl': {
                'smarts': '[CH2]=[CH1]',
                'name': 'Vinyl (乙烯基)',
                'description': 'CH2=CH-',
                'category': 'Unsaturated'
            },
            'allyl': {
                'smarts': '[CX3]=[CX3][CX4]',
                'name': 'Allyl (烯丙基)',
                'description': 'C=C-C',
                'category': 'Unsaturated'
            },
            'propargyl': {
                'smarts': '[CX2]#[CX2][CX4]',
                'name': 'Propargyl (炔丙基)',
                'description': 'C≡C-C',
                'category': 'Unsaturated'
            },

            # ============================================================
            # 30. AROMATIC SYSTEMS (芳香体系)
            # ============================================================
            'benzene': {
                'smarts': 'c1ccccc1',
                'name': 'Benzene (苯)',
                'description': '6-membered aromatic ring',
                'category': 'Aromatics'
            },
            'naphthalene': {
                'smarts': 'c1ccc2ccccc2c1',
                'name': 'Naphthalene (萘)',
                'description': 'Fused bicyclic aromatic',
                'category': 'Aromatics'
            },
            'anthracene': {
                'smarts': 'c1ccc2cc3ccccc3cc2c1',
                'name': 'Anthracene (蒽)',
                'description': 'Linear tricyclic aromatic',
                'category': 'Aromatics'
            },
            'phenanthrene': {
                'smarts': 'c1ccc2c(c1)ccc3ccccc23',
                'name': 'Phenanthrene (菲)',
                'description': 'Angular tricyclic aromatic',
                'category': 'Aromatics'
            },
            'pyrene': {
                'smarts': 'c1cc2ccc3cccc4ccc(c1)c2c34',
                'name': 'Pyrene (芘)',
                'description': 'Tetracyclic aromatic',
                'category': 'Aromatics'
            },
            'biphenyl': {
                'smarts': 'c1ccccc1-c2ccccc2',
                'name': 'Biphenyl (联苯)',
                'description': 'Two connected benzene rings',
                'category': 'Aromatics'
            },
            'terphenyl': {
                'smarts': 'c1ccccc1-c2ccc(-c3ccccc3)cc2',
                'name': 'Terphenyl (三联苯)',
                'description': 'Three connected benzene rings',
                'category': 'Aromatics'
            },
            'phenol': {
                'smarts': 'c[OX2H1]',
                'name': 'Phenol (苯酚)',
                'description': 'Ar-OH',
                'category': 'Aromatics'
            },
            'catechol': {
                'smarts': 'c1cc(O)c(O)cc1',
                'name': 'Catechol (邻苯二酚)',
                'description': '1,2-dihydroxybenzene',
                'category': 'Aromatics'
            },
            'resorcinol': {
                'smarts': 'c1cc(O)cc(O)c1',
                'name': 'Resorcinol (间苯二酚)',
                'description': '1,3-dihydroxybenzene',
                'category': 'Aromatics'
            },
            'hydroquinone': {
                'smarts': 'c1cc(O)ccc1O',
                'name': 'Hydroquinone (对苯二酚)',
                'description': '1,4-dihydroxybenzene',
                'category': 'Aromatics'
            },
            'aniline': {
                'smarts': 'c[NX3;H2]',
                'name': 'Aniline (苯胺)',
                'description': 'Ar-NH2',
                'category': 'Aromatics'
            },
            'anisole': {
                'smarts': 'c[OX2][CH3]',
                'name': 'Anisole (苯甲醚)',
                'description': 'Ar-O-CH3',
                'category': 'Aromatics'
            },
            'toluene': {
                'smarts': 'c[CH3]',
                'name': 'Toluene (甲苯)',
                'description': 'Ar-CH3',
                'category': 'Aromatics'
            },
            'xylene': {
                'smarts': 'c([CH3])c[CH3]',
                'name': 'Xylene (二甲苯)',
                'description': 'Ar-(CH3)2',
                'category': 'Aromatics'
            },
            'styrene': {
                'smarts': 'c[CX3]=[CH2]',
                'name': 'Styrene (苯乙烯)',
                'description': 'Ar-CH=CH2',
                'category': 'Aromatics'
            },
            'benzyl': {
                'smarts': 'c[CH2]',
                'name': 'Benzyl (苄基)',
                'description': 'Ar-CH2-',
                'category': 'Aromatics'
            },
            'phenyl': {
                'smarts': '[cH1]1[cH1][cH1][cH1][cH1]c1',
                'name': 'Phenyl (苯基)',
                'description': 'C6H5-',
                'category': 'Aromatics'
            },

            # ============================================================
            # 31. HYDRAZINE & AZO & AZIDE (肼 & 偶氮 & 叠氮)
            # ============================================================
            'hydrazine': {
                'smarts': '[NX3;H2][NX3;H2]',
                'name': 'Hydrazine (肼)',
                'description': 'H2N-NH2 group',
                'category': 'Hydrazine'
            },
            'hydrazine_mono_substituted': {
                'smarts': '[NX3;H1]([#6])[NX3;H2]',
                'name': 'Monosubstituted Hydrazine',
                'description': 'R-NH-NH2',
                'category': 'Hydrazine'
            },
            'hydrazine_di_substituted': {
                'smarts': '[NX3]([#6])([#6])[NX3]',
                'name': 'Disubstituted Hydrazine',
                'description': 'R2N-NR2 or R-NH-NH-R',
                'category': 'Hydrazine'
            },
            'hydrazide': {
                'smarts': '[CX3](=O)[NX3][NX3]',
                'name': 'Hydrazide (酰肼)',
                'description': 'C(=O)-NH-NH2',
                'category': 'Hydrazine'
            },
            'hydrazone': {
                'smarts': '[CX3]=[NX2][NX3]',
                'name': 'Hydrazone (腙)',
                'description': 'C=N-NH2',
                'category': 'Hydrazine'
            },
            'semicarbazone': {
                'smarts': '[CX3]=[NX2][NX3][CX3](=O)[NX3]',
                'name': 'Semicarbazone (缩氨脲)',
                'description': 'C=N-NH-CO-NH2',
                'category': 'Hydrazine'
            },
            'azo': {
                'smarts': '[NX2]=[NX2]',
                'name': 'Azo (偶氮)',
                'description': 'R-N=N-R',
                'category': 'Hydrazine'
            },
            'azoxy': {
                'smarts': '[NX2]=[NX3+]([O-])',
                'name': 'Azoxy (氧化偶氮)',
                'description': 'R-N=N(O)-R',
                'category': 'Hydrazine'
            },
            'diazo': {
                'smarts': '[CX3]=[NX2+]=[NX1-]',
                'name': 'Diazo (重氮)',
                'description': 'R2C=N+=N-',
                'category': 'Hydrazine'
            },
            'diazonium': {
                'smarts': '[NX2+]#[NX1]',
                'name': 'Diazonium (重氮盐)',
                'description': 'R-N≡N+',
                'category': 'Hydrazine'
            },
            'azide': {
                'smarts': '[NX1]=[NX2]=[NX2]',
                'name': 'Azide (叠氮)',
                'description': 'R-N3',
                'category': 'Hydrazine'
            },
            'organic_azide': {
                'smarts': '[#6][NX2]=[NX2+]=[NX1-]',
                'name': 'Organic Azide (有机叠氮)',
                'description': 'R-N=N+=N-',
                'category': 'Hydrazine'
            },

            # ============================================================
            # 32. ISOCYANATE & CARBAMATE & UREA (异氰酸酯 & 氨基甲酸酯 & 脲)
            # ============================================================
            'isocyanate': {
                'smarts': '[NX2]=[CX2]=[OX1]',
                'name': 'Isocyanate (异氰酸酯)',
                'description': 'R-N=C=O',
                'category': 'Urea'
            },
            'carbamate': {
                'smarts': '[NX3][CX3](=O)[OX2]',
                'name': 'Carbamate (氨基甲酸酯)',
                'description': 'R2N-CO-O-R',
                'category': 'Urea'
            },
            'urethane': {
                'smarts': '[OX2][CX3](=O)[NX3]',
                'name': 'Urethane (氨基甲酸酯)',
                'description': 'R-O-CO-NR2',
                'category': 'Urea'
            },
            'urea': {
                'smarts': '[NX3][CX3](=O)[NX3]',
                'name': 'Urea (脲)',
                'description': 'R2N-CO-NR2',
                'category': 'Urea'
            },
            'cyclic_urea': {
                'smarts': '[NX3]1[CX3](=O)[NX3]C1',
                'name': 'Cyclic Urea (环脲)',
                'description': 'Cyclic urea',
                'category': 'Urea'
            },
            'guanidine': {
                'smarts': '[NX3][CX3](=[NX2])[NX3]',
                'name': 'Guanidine (胍)',
                'description': 'R2N-C(=NR)-NR2',
                'category': 'Urea'
            },
            'guanidinium': {
                'smarts': '[NX3][CX3](=[NX2+])[NX3]',
                'name': 'Guanidinium (胍盐)',
                'description': 'Protonated guanidine',
                'category': 'Urea'
            },
            'amidine': {
                'smarts': '[CX3](=[NX2])[NX3]',
                'name': 'Amidine (脒)',
                'description': 'R-C(=NR)-NR2',
                'category': 'Urea'
            },
            'carbodiimide': {
                'smarts': '[NX2]=[CX2]=[NX2]',
                'name': 'Carbodiimide (碳二亚胺)',
                'description': 'R-N=C=N-R',
                'category': 'Urea'
            },
            'cyanamide': {
                'smarts': '[NX3][CX2]#[NX1]',
                'name': 'Cyanamide (氰胺)',
                'description': 'R2N-C≡N',
                'category': 'Urea'
            },

            # ============================================================
            # 33. PHOSPHORUS (磷)
            # ============================================================
            'phosphate': {
                'smarts': '[PX4](=O)([OX2])([OX2])[OX2]',
                'name': 'Phosphate (磷酸酯)',
                'description': 'R-O-PO3',
                'category': 'Phosphorus'
            },
            'phosphonate': {
                'smarts': '[PX4](=O)([OX2])([OX2])[#6]',
                'name': 'Phosphonate (膦酸酯)',
                'description': 'R-PO3',
                'category': 'Phosphorus'
            },
            'phosphine': {
                'smarts': '[PX3]([#6])([#6])[#6]',
                'name': 'Phosphine (膦)',
                'description': 'R3P',
                'category': 'Phosphorus'
            },
            'phosphine_oxide': {
                'smarts': '[PX4](=O)([#6])([#6])[#6]',
                'name': 'Phosphine Oxide (氧化膦)',
                'description': 'R3P=O',
                'category': 'Phosphorus'
            },
            'phosphoramide': {
                'smarts': '[PX4](=O)([NX3])([NX3])[NX3]',
                'name': 'Phosphoramide (磷酰胺)',
                'description': '(R2N)3P=O',
                'category': 'Phosphorus'
            },
            'phosphoric_acid': {
                'smarts': '[PX4](=O)([OX2H1])([OX2H1])[OX2H1]',
                'name': 'Phosphoric Acid (磷酸)',
                'description': 'H3PO4',
                'category': 'Phosphorus'
            },
            'phosphonic_acid': {
                'smarts': '[PX4](=O)([OX2H1])([OX2H1])[#6]',
                'name': 'Phosphonic Acid (膦酸)',
                'description': 'R-PO(OH)2',
                'category': 'Phosphorus'
            },
            'phosphodiester': {
                'smarts': '[OX2][PX4](=O)([OX2])[OX2]',
                'name': 'Phosphodiester (磷酸二酯)',
                'description': 'R-O-PO2-O-R',
                'category': 'Phosphorus'
            },

            # ============================================================
            # 34. BORON (硼)
            # ============================================================
            'boronic_acid': {
                'smarts': '[BX3]([OX2H1])([OX2H1])',
                'name': 'Boronic Acid (硼酸)',
                'description': 'R-B(OH)2',
                'category': 'Boron'
            },
            'boronate_ester': {
                'smarts': '[BX3]([OX2])([OX2])',
                'name': 'Boronate Ester (硼酸酯)',
                'description': 'R-B(OR)2',
                'category': 'Boron'
            },
            'borate': {
                'smarts': '[BX4-]([OX2])([OX2])([OX2])[OX2]',
                'name': 'Borate (硼酸盐)',
                'description': 'B(OR)4-',
                'category': 'Boron'
            },
            'borane': {
                'smarts': '[BX3]([#6])([#6])[#6]',
                'name': 'Borane (硼烷)',
                'description': 'R3B',
                'category': 'Boron'
            },

            # ============================================================
            # 35. CYCLOALKANES (环烷烃)
            # ============================================================
            'cyclopropane': {
                'smarts': 'C1CC1',
                'name': 'Cyclopropane (环丙烷)',
                'description': '3-membered carbon ring',
                'category': 'Cycloalkanes'
            },
            'cyclopropylamine': {
                'smarts': 'C1CC1[NX3]',
                'name': 'Cyclopropylamine (环丙胺)',
                'description': 'Cyclopropane with amine',
                'category': 'Cycloalkanes'
            },
            'cyclobutane': {
                'smarts': 'C1CCC1',
                'name': 'Cyclobutane (环丁烷)',
                'description': '4-membered carbon ring',
                'category': 'Cycloalkanes'
            },
            'cyclopentane': {
                'smarts': 'C1CCCC1',
                'name': 'Cyclopentane (环戊烷)',
                'description': '5-membered carbon ring',
                'category': 'Cycloalkanes'
            },
            'cyclohexane': {
                'smarts': 'C1CCCCC1',
                'name': 'Cyclohexane (环己烷)',
                'description': '6-membered carbon ring',
                'category': 'Cycloalkanes'
            },
            'cycloheptane': {
                'smarts': 'C1CCCCCC1',
                'name': 'Cycloheptane (环庚烷)',
                'description': '7-membered carbon ring',
                'category': 'Cycloalkanes'
            },
            'cyclooctane': {
                'smarts': 'C1CCCCCCC1',
                'name': 'Cyclooctane (环辛烷)',
                'description': '8-membered carbon ring',
                'category': 'Cycloalkanes'
            },
            'adamantane': {
                'smarts': 'C1C2CC3CC1CC(C2)C3',
                'name': 'Adamantane (金刚烷)',
                'description': 'Tricyclic cage structure',
                'category': 'Cycloalkanes'
            },
            'norbornane': {
                'smarts': 'C1CC2CCC1C2',
                'name': 'Norbornane (降冰片烷)',
                'description': 'Bicyclic bridged structure',
                'category': 'Cycloalkanes'
            },
            'decalin': {
                'smarts': 'C1CCC2CCCCC2C1',
                'name': 'Decalin (十氢萘)',
                'description': 'Fused bicyclic',
                'category': 'Cycloalkanes'
            },
            'spirocycle': {
                'smarts': '[R]1[R][R]([R]2[R][R][R]2)[R]1',
                'name': 'Spirocycle (螺环)',
                'description': 'Spiro ring system',
                'category': 'Cycloalkanes'
            },
            'bicyclic': {
                'smarts': '[R2]',
                'name': 'Bicyclic (双环)',
                'description': 'Atom in two rings',
                'category': 'Cycloalkanes'
            },

            # ============================================================
            # 36. SILICON (硅)
            # ============================================================
            'silane': {
                'smarts': '[SiX4]([#6])([#6])([#6])[#6]',
                'name': 'Silane (硅烷)',
                'description': 'R4Si',
                'category': 'Silicon'
            },
            'silyl_ether': {
                'smarts': '[SiX4]([#6])([#6])([#6])[OX2]',
                'name': 'Silyl Ether (硅醚)',
                'description': 'R3Si-O-R',
                'category': 'Silicon'
            },
            'silanol': {
                'smarts': '[SiX4][OX2H1]',
                'name': 'Silanol (硅醇)',
                'description': 'R3Si-OH',
                'category': 'Silicon'
            },
            'siloxane': {
                'smarts': '[SiX4][OX2][SiX4]',
                'name': 'Siloxane (硅氧烷)',
                'description': 'Si-O-Si',
                'category': 'Silicon'
            },

            # ============================================================
            # 37. METAL COORDINATION (金属配位)
            # ============================================================
            'metal_carboxylate': {
                'smarts': '[CX3](=O)[O-]',
                'name': 'Metal Carboxylate (金属羧酸盐)',
                'description': 'R-COO- (metal salt)',
                'category': 'Metal'
            },
            'metal_alkoxide': {
                'smarts': '[OX1-]',
                'name': 'Metal Alkoxide (金属醇盐)',
                'description': 'R-O- (metal salt)',
                'category': 'Metal'
            },
            'metal_amide': {
                'smarts': '[NX2-]',
                'name': 'Metal Amide (金属酰胺)',
                'description': 'R2N- (metal salt)',
                'category': 'Metal'
            },

            # ============================================================
            # 38. ANHYDRIDE (酸酐)
            # ============================================================
            'anhydride': {
                'smarts': '[CX3](=O)[OX2][CX3](=O)',
                'name': 'Anhydride (酸酐)',
                'description': 'R-CO-O-CO-R',
                'category': 'Anhydride'
            },
            'cyclic_anhydride': {
                'smarts': '[R][CX3](=O)[OX2][CX3](=O)[R]',
                'name': 'Cyclic Anhydride (环酸酐)',
                'description': 'Cyclic anhydride',
                'category': 'Anhydride'
            },
            'mixed_anhydride': {
                'smarts': '[CX3](=O)[OX2][PX4,SX4]',
                'name': 'Mixed Anhydride (混合酸酐)',
                'description': 'Carboxylic-phosphoric/sulfonic anhydride',
                'category': 'Anhydride'
            },

            # ============================================================
            # 39. PEROXIDE (过氧化物)
            # ============================================================
            'peroxide': {
                'smarts': '[OX2][OX2]',
                'name': 'Peroxide (过氧化物)',
                'description': 'R-O-O-R',
                'category': 'Peroxide'
            },
            'hydroperoxide': {
                'smarts': '[OX2][OX2H1]',
                'name': 'Hydroperoxide (氢过氧化物)',
                'description': 'R-O-O-H',
                'category': 'Peroxide'
            },
            'peracid': {
                'smarts': '[CX3](=O)[OX2][OX2H1]',
                'name': 'Peracid (过酸)',
                'description': 'R-CO-O-O-H',
                'category': 'Peroxide'
            },

            # ============================================================
            # 40. TETRAZOLE (四唑)
            # ============================================================
            'tetrazole': {
                'smarts': 'c1nnn[nH]1',
                'name': 'Tetrazole (四唑)',
                'description': '5-membered ring with 4 N atoms',
                'category': 'Tetrazole'
            },
            'tetrazole_substituted': {
                'smarts': 'c1nnnn1*',
                'name': 'Substituted Tetrazole (取代四唑)',
                'description': 'N-substituted tetrazole',
                'category': 'Tetrazole'
            },
            'tetrazine': {
                'smarts': 'c1nncnn1',
                'name': 'Tetrazine (四嗪)',
                'description': '6-membered ring with 4 N atoms',
                'category': 'Tetrazole'
            },

            # ============================================================
            # 41. SPECIAL DRUG MOTIFS (特殊药物基序)
            # ============================================================
            'sulfa_drug': {
                'smarts': 'c1ccc(N)cc1[SX4](=O)(=O)[NX3]',
                'name': 'Sulfa Drug Core (磺胺药核心)',
                'description': 'p-aminobenzenesulfonamide',
                'category': 'Drug_Motifs'
            },
            'beta_lactam_core': {
                'smarts': '[NX3]1[CX4][CX4][CX3]1=O',
                'name': 'β-Lactam Core (β-内酰胺核心)',
                'description': 'Penicillin/cephalosporin core',
                'category': 'Drug_Motifs'
            },
            'benzodiazepine': {
                'smarts': 'c1ccc2NC(=O)CN=C(c3ccccc3)c2c1',
                'name': 'Benzodiazepine (苯二氮卓)',
                'description': 'Benzodiazepine core',
                'category': 'Drug_Motifs'
            },
            'barbiturate': {
                'smarts': 'O=C1NC(=O)NC(=O)C1',
                'name': 'Barbiturate (巴比妥)',
                'description': 'Barbituric acid core',
                'category': 'Drug_Motifs'
            },
            'steroid_core': {
                'smarts': 'C1CCC2C(C1)CCC3C2CCC4CCCCC34',
                'name': 'Steroid Core (甾体核心)',
                'description': 'Steroid ring system',
                'category': 'Drug_Motifs'
            },
            'nucleoside_base': {
                'smarts': '[nR1]1c[nR1]c2c1nc[nR1]c2',
                'name': 'Nucleoside Base (核苷碱基)',
                'description': 'Purine/pyrimidine base',
                'category': 'Drug_Motifs'
            },

            # ============================================================
            # 42. AMINO ACID SIDE CHAINS (氨基酸侧链)
            # ============================================================
            'indole_tryptophan': {
                'smarts': 'c1ccc2[nH]cc(CC([NX3])C(=O)O)c2c1',
                'name': 'Tryptophan Side Chain (色氨酸侧链)',
                'description': 'Indole-containing amino acid',
                'category': 'Amino_Acids'
            },
            'imidazole_histidine': {
                'smarts': 'c1c[nH]cn1CC([NX3])C(=O)O',
                'name': 'Histidine Side Chain (组氨酸侧链)',
                'description': 'Imidazole-containing amino acid',
                'category': 'Amino_Acids'
            },
            'phenol_tyrosine': {
                'smarts': 'c1cc(O)ccc1CC([NX3])C(=O)O',
                'name': 'Tyrosine Side Chain (酪氨酸侧链)',
                'description': 'Phenol-containing amino acid',
                'category': 'Amino_Acids'
            },
            'thiol_cysteine': {
                'smarts': '[SX2H1]CC([NX3])C(=O)O',
                'name': 'Cysteine Side Chain (半胱氨酸侧链)',
                'description': 'Thiol-containing amino acid',
                'category': 'Amino_Acids'
            },
            'guanidine_arginine': {
                'smarts': '[NX3][CX3](=[NX2])[NX3]CCCC([NX3])C(=O)O',
                'name': 'Arginine Side Chain (精氨酸侧链)',
                'description': 'Guanidine-containing amino acid',
                'category': 'Amino_Acids'
            },

            # ============================================================
            # 43. SUGAR MOTIFS (糖类基序)
            # ============================================================
            'pyranose': {
                'smarts': 'C1OC(O)C(O)C(O)C1O',
                'name': 'Pyranose (吡喃糖)',
                'description': '6-membered sugar ring',
                'category': 'Sugars'
            },
            'furanose': {
                'smarts': 'C1OC(O)C(O)C1O',
                'name': 'Furanose (呋喃糖)',
                'description': '5-membered sugar ring',
                'category': 'Sugars'
            },
            'glycosidic_bond': {
                'smarts': '[OX2][CX4]1[OX2][CX4][CX4][CX4][CX4]1',
                'name': 'Glycosidic Bond (糖苷键)',
                'description': 'O-glycosidic linkage',
                'category': 'Sugars'
            },
            'n_glycoside': {
                'smarts': '[NX3][CX4]1[OX2][CX4][CX4][CX4][CX4]1',
                'name': 'N-Glycoside (N-糖苷)',
                'description': 'N-glycosidic linkage',
                'category': 'Sugars'
            },

            # ============================================================
            # 44. LIPID MOTIFS (脂质基序)
            # ============================================================
            'fatty_acid': {
                'smarts': 'CCCCCCCC[CX3](=O)[OX2H1]',
                'name': 'Fatty Acid (脂肪酸)',
                'description': 'Long chain carboxylic acid',
                'category': 'Lipids'
            },
            'glycerol_ester': {
                'smarts': '[CX4]([OX2][CX3]=O)([CX4][OX2][CX3]=O)[CX4][OX2]',
                'name': 'Glycerol Ester (甘油酯)',
                'description': 'Triglyceride backbone',
                'category': 'Lipids'
            },
            'phospholipid_head': {
                'smarts': '[OX2][PX4](=O)([OX2])[OX2][CX4][CX4][NX4+]',
                'name': 'Phospholipid Head (磷脂头基)',
                'description': 'Phosphatidylcholine head',
                'category': 'Lipids'
            },
            'sphingosine': {
                'smarts': '[CX3]=[CX3][CX4]([OX2H1])[CX4]([NX3])[CX4][OX2H1]',
                'name': 'Sphingosine (鞘氨醇)',
                'description': 'Sphingolipid backbone',
                'category': 'Lipids'
            },
        }

        
        # 按类别组织 - 扩展版
        self.categories = [
            # 基础官能团
            'Amines', 'Alcohols', 'Ethers', 'Aldehydes', 'Ketones',
            'Carboxylic_Acids', 'Esters', 'Amides', 'Nitriles', 'Nitro',
            # 含氮杂环
            'Triazole', 'Imidazole', 'Pyridine', 'Pyrimidine', 'Pyrazine',
            'Pyrrole', 'Pyrazole', 'Oxazole', 'Thiazole', 'Quinoline', 'Purine',
            'Piperidine',
            # 含硫基团
            'Thiophene', 'Sulfur',
            # 含氧杂环
            'Furan', 'Epoxide',
            # 卤素
            'Halogens',
            # 不饱和键
            'Unsaturated',
            # 芳香体系
            'Aromatics',
            # 特殊基团
            'Hydrazine', 'Urea', 'Phosphorus', 'Boron',
            # 环系
            'Cycloalkanes', 'Silicon', 'Metal', 'Anhydride', 'Peroxide',
            'Tetrazole',
            # 生物相关
            'Drug_Motifs', 'Amino_Acids', 'Sugars', 'Lipids'
        ]
    
    def _compile_smarts(self):
        """预编译所有SMARTS模式"""
        self.compiled_patterns = {}
        failed_count = 0
        
        for name, info in self.target_functional_groups.items():
            try:
                pattern = Chem.MolFromSmarts(info['smarts'])
                if pattern is not None:
                    self.compiled_patterns[name] = {
                        'pattern': pattern,
                        'info': info
                    }
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
        
        print(f"Successfully compiled {len(self.compiled_patterns)}/{len(self.target_functional_groups)} SMARTS patterns")
        if failed_count > 0:
            print(f"Warning: {failed_count} SMARTS patterns failed to compile")

    def extract_motifs_from_smiles(self, smiles: str) -> Dict:
        """Extract all target functional groups from SMILES"""
        if not smiles:
            return {'motifs': [], 'smiles': smiles, 'mol': None}
        
        if not RDKIT_AVAILABLE:
            return {'motifs': [], 'smiles': smiles, 'mol': None, 
                    'warning': 'RDKit not available'}
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'motifs': [], 'smiles': smiles, 'mol': None}
        
        all_motifs = []
        
        for name, data in self.compiled_patterns.items():
            pattern = data['pattern']
            info = data['info']
            
            matches = mol.GetSubstructMatches(pattern)
            
            for match_idx, atom_indices in enumerate(matches):
                motif_info = {
                    'name': name,
                    'display_name': info['name'],
                    'description': info.get('description', ''),
                    'category': info.get('category', 'Unknown'),
                    'smarts': info['smarts'],
                    'atom_indices': list(atom_indices),
                    'num_atoms': len(atom_indices),
                    'match_index': match_idx,
                    'atoms': [mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices]
                }
                all_motifs.append(motif_info)
        
        all_motifs = self._deduplicate_motifs(all_motifs)
        
        return {
            'smiles': smiles,
            'mol': mol,
            'num_atoms': mol.GetNumAtoms(),
            'motifs': all_motifs,
            'motif_count': len(all_motifs),
            'by_category': self._group_by_category(all_motifs),
            'summary': self._summarize_motifs(all_motifs)
        }
    
    def _deduplicate_motifs(self, motifs: List[Dict]) -> List[Dict]:
        """去除完全相同原子集合的重复基序"""
        seen = set()
        unique = []
        for m in motifs:
            key = (m['name'], tuple(sorted(m['atom_indices'])))
            if key not in seen:
                seen.add(key)
                unique.append(m)
        return unique
    
    def _group_by_category(self, motifs: List[Dict]) -> Dict[str, List[Dict]]:
        """按类别分组"""
        grouped = defaultdict(list)
        for m in motifs:
            grouped[m['category']].append(m)
        return dict(grouped)
    
    def _summarize_motifs(self, motifs: List[Dict]) -> Dict:
        """生成基序摘要"""
        summary = {
            'total_count': len(motifs),
            'by_category': defaultdict(int),
            'by_name': defaultdict(int),
        }
        for m in motifs:
            summary['by_category'][m['category']] += 1
            summary['by_name'][m['display_name']] += 1
        return dict(summary)


    def visualize_smarts_patterns(self, save_path: str = None):
        """Visualize all SMARTS pattern structures"""
        if not RDKIT_AVAILABLE:
            print("RDKit required for SMARTS visualization")
            return
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        patterns_by_category = defaultdict(list)
        for name, info in self.target_functional_groups.items():
            patterns_by_category[info['category']].append((name, info))
        
        for category in self.categories:
            if category not in patterns_by_category:
                continue
            
            patterns = patterns_by_category[category]
            n_patterns = len(patterns)
            
            n_cols = min(4, n_patterns)
            n_rows = (n_patterns + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            fig.suptitle(f'{category} SMARTS Patterns ({n_patterns})', fontsize=12, fontweight='bold')
            
            if n_patterns == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
            for idx, (name, info) in enumerate(patterns):
                ax = axes[idx]
                smarts = info['smarts']
                
                mol = Chem.MolFromSmarts(smarts)
                
                if mol is not None:
                    try:
                        AllChem.Compute2DCoords(mol)
                        img = Draw.MolToImage(mol, size=(250, 200))
                        ax.imshow(img)
                    except:
                        ax.text(0.5, 0.5, f"SMARTS:\n{smarts}", 
                               ha='center', va='center', fontsize=8,
                               transform=ax.transAxes, wrap=True)
                else:
                    ax.text(0.5, 0.5, f"SMARTS:\n{smarts}", 
                           ha='center', va='center', fontsize=8,
                           transform=ax.transAxes, wrap=True)
                
                ax.set_title(f"{info['name'][:30]}", fontsize=8)
                ax.axis('off')
            
            for idx in range(n_patterns, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                fig_path = os.path.join(save_path, f'{category}_patterns.png')
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            
            plt.close()

    def visualize_motifs_on_molecule(self, smiles: str, save_path: str = None,
                                      highlight_categories: List[str] = None) -> Dict:
        """Highlight identified functional groups on molecule diagram"""
        if not RDKIT_AVAILABLE:
            print("RDKit required for molecule visualization")
            return {}
        
        result = self.extract_motifs_from_smiles(smiles)
        mol = result.get('mol')
        
        if mol is None:
            return {'error': 'Cannot parse SMILES'}
        
        motifs = result['motifs']
        
        if highlight_categories:
            motifs = [m for m in motifs if m['category'] in highlight_categories]
        
        if not motifs:
            return result
        
        # 为每个类别分配颜色
        import colorsys
        n_categories = len(self.categories)
        category_colors = {}
        for i, cat in enumerate(self.categories):
            hue = i / n_categories
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            category_colors[cat] = rgb
        
        highlight_atoms = []
        atom_colors = {}
        
        for motif in motifs:
            color = category_colors.get(motif['category'], (0.8, 0.8, 0.8))
            for atom_idx in motif['atom_indices']:
                if atom_idx not in highlight_atoms:
                    highlight_atoms.append(atom_idx)
                    atom_colors[atom_idx] = color
        
        AllChem.Compute2DCoords(mol)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                           highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        import io
        from PIL import Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        axes[0].imshow(img)
        axes[0].set_title(f'Molecule with Highlighted Functional Groups\n{smiles[:50]}...', fontsize=10)
        axes[0].axis('off')
        
        axes[1].axis('off')
        axes[1].set_title('Detected Functional Groups', fontsize=11, fontweight='bold')
        
        y_pos = 0.98
        for category in self.categories:
            category_motifs = [m for m in motifs if m['category'] == category]
            if category_motifs:
                color = category_colors.get(category, (0.8, 0.8, 0.8))
                
                axes[1].add_patch(plt.Rectangle((0.02, y_pos-0.015), 0.03, 0.02, 
                                               facecolor=color, transform=axes[1].transAxes))
                axes[1].text(0.06, y_pos, f'{category}:', fontsize=9, fontweight='bold',
                           transform=axes[1].transAxes, va='top')
                y_pos -= 0.035
                
                for m in category_motifs[:5]:  # 限制显示数量
                    axes[1].text(0.08, y_pos, 
                               f"• {m['display_name'][:35]}", 
                               fontsize=7, transform=axes[1].transAxes, va='top')
                    y_pos -= 0.025
                
                if len(category_motifs) > 5:
                    axes[1].text(0.08, y_pos, f"  ... and {len(category_motifs)-5} more",
                               fontsize=7, transform=axes[1].transAxes, va='top', style='italic')
                    y_pos -= 0.025
                
                y_pos -= 0.01
                
                if y_pos < 0.05:
                    break
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        
        return result

    def print_all_patterns(self):
        """Print all SMARTS patterns"""
        print("\n" + "="*80)
        print(f"Target Functional Group SMARTS Patterns (Total: {len(self.target_functional_groups)})")
        print("="*80)
        
        for category in self.categories:
            patterns = [(n, i) for n, i in self.target_functional_groups.items() 
                       if i['category'] == category]
            if patterns:
                print(f"\n[{category}] ({len(patterns)} types)")
                print("-"*60)
                for name, info in patterns:
                    print(f"  {info['name']}")
                    print(f"    SMARTS: {info['smarts']}")
                    print()
    
    def get_statistics(self) -> Dict:
        """Get functional group statistics"""
        stats = {
            'total_patterns': len(self.target_functional_groups),
            'total_categories': len(self.categories),
            'patterns_by_category': {}
        }
        
        for category in self.categories:
            count = sum(1 for info in self.target_functional_groups.values() 
                       if info['category'] == category)
            if count > 0:
                stats['patterns_by_category'][category] = count
        
        return stats


class MotifAttentionAnalyzer:
    """基序级别的注意力分析器"""
    
    def __init__(self, motif_analyzer: ChemicalMotifAnalyzer = None):
        self.motif_analyzer = motif_analyzer or ChemicalMotifAnalyzer()
    
    def compute_motif_attention(self, smiles: str, 
                                 atom_attention: np.ndarray = None,
                                 embedding: np.ndarray = None) -> Dict:
        """计算每个化学基序的注意力权重"""
        result = self.motif_analyzer.extract_motifs_from_smiles(smiles)
        motifs = result['motifs']
        num_atoms = result.get('num_atoms', 0)
        
        if not motifs:
            return {'smiles': smiles, 'motifs_with_attention': [], 'top_motifs': []}
        
        if atom_attention is None and embedding is not None:
            if num_atoms > 0:
                atom_attention = np.interp(
                    np.linspace(0, len(embedding)-1, num_atoms),
                    np.arange(len(embedding)),
                    np.abs(embedding)
                )
            else:
                atom_attention = np.ones(max(1, num_atoms))
        
        if atom_attention is None:
            atom_attention = np.ones(max(1, num_atoms))
        
        motifs_with_attention = []
        for motif in motifs:
            atom_indices = motif['atom_indices']
            valid_indices = [i for i in atom_indices if i < len(atom_attention)]
            
            if valid_indices:
                motif_attn_values = atom_attention[valid_indices]
                motif_attention = {
                    'mean': float(np.mean(motif_attn_values)),
                    'max': float(np.max(motif_attn_values)),
                    'sum': float(np.sum(motif_attn_values)),
                }
            else:
                motif_attention = {'mean': 0, 'max': 0, 'sum': 0}
            
            motifs_with_attention.append({**motif, 'attention': motif_attention})
        
        motifs_with_attention.sort(key=lambda x: x['attention']['mean'], reverse=True)
        
        return {
            'smiles': smiles,
            'num_atoms': num_atoms,
            'motifs_with_attention': motifs_with_attention,
            'top_motifs': motifs_with_attention[:10],
        }

    def compute_enzyme_motif_interaction(self, enzymes: Dict, smiles: str,
                                          emb_enzyme: np.ndarray,
                                          emb_smiles: np.ndarray) -> Dict:
        """计算酶与化学基序之间的交互"""
        result = self.motif_analyzer.extract_motifs_from_smiles(smiles)
        motifs = result['motifs']
        
        if not enzymes or not motifs:
            return {'enzyme_motif_pairs': [], 'top_interactions': []}
        
        enzyme_ids = list(enzymes.keys())
        n_enzymes = len(enzyme_ids)
        n_motifs = len(motifs)
        
        emb_similarity = np.dot(emb_enzyme, emb_smiles) / (
            np.linalg.norm(emb_enzyme) * np.linalg.norm(emb_smiles) + 1e-8
        )
        
        # 类别权重 - 扩展版
        category_weights = {
            'Amines': 1.5, 'Alcohols': 1.2, 'Ethers': 1.1, 'Aldehydes': 1.3,
            'Ketones': 1.3, 'Carboxylic_Acids': 1.4, 'Esters': 1.2, 'Amides': 1.4,
            'Nitriles': 1.2, 'Nitro': 1.3, 'Triazole': 1.4, 'Imidazole': 1.4,
            'Pyridine': 1.4, 'Pyrimidine': 1.4, 'Pyrazine': 1.3, 'Pyrrole': 1.3,
            'Pyrazole': 1.3, 'Oxazole': 1.3, 'Thiazole': 1.3, 'Quinoline': 1.4,
            'Purine': 1.5, 'Piperidine': 1.3, 'Thiophene': 1.3, 'Sulfur': 1.4,
            'Furan': 1.2, 'Epoxide': 1.3, 'Halogens': 1.2, 'Unsaturated': 1.1,
            'Aromatics': 1.2, 'Hydrazine': 1.3, 'Urea': 1.3, 'Phosphorus': 1.4,
            'Boron': 1.3, 'Cycloalkanes': 1.1, 'Silicon': 1.2, 'Metal': 1.3,
            'Anhydride': 1.2, 'Peroxide': 1.3, 'Tetrazole': 1.4,
            'Drug_Motifs': 1.5, 'Amino_Acids': 1.4, 'Sugars': 1.3, 'Lipids': 1.2
        }
        
        enzyme_motif_pairs = []
        interaction_matrix = np.zeros((n_enzymes, n_motifs))
        
        for i, eid in enumerate(enzyme_ids):
            enzyme_name = enzymes[eid]['name']
            
            for j, motif in enumerate(motifs):
                category_weight = category_weights.get(motif['category'], 1.0)
                interaction_score = abs(emb_similarity) * category_weight
                
                interaction_matrix[i, j] = interaction_score
                
                enzyme_motif_pairs.append({
                    'enzyme_id': eid,
                    'enzyme_name': enzyme_name,
                    'motif_name': motif['name'],
                    'motif_display_name': motif['display_name'],
                    'motif_category': motif['category'],
                    'atom_indices': motif['atom_indices'],
                    'interaction_score': float(interaction_score),
                })
        
        if interaction_matrix.max() > 0:
            interaction_matrix = interaction_matrix / interaction_matrix.max()
        
        for pair in enzyme_motif_pairs:
            i = enzyme_ids.index(pair['enzyme_id'])
            j = next(idx for idx, m in enumerate(motifs) 
                    if m['name'] == pair['motif_name'] and m['atom_indices'] == pair['atom_indices'])
            pair['normalized_score'] = float(interaction_matrix[i, j])
        
        enzyme_motif_pairs.sort(key=lambda x: x['normalized_score'], reverse=True)
        
        return {
            'enzyme_motif_pairs': enzyme_motif_pairs,
            'top_interactions': enzyme_motif_pairs[:30],
            'interaction_matrix': interaction_matrix.tolist(),
            'enzyme_labels': [enzymes[eid]['name'] for eid in enzyme_ids],
            'motif_labels': [f"{m['display_name']}" for m in motifs],
        }


class MotifVisualizationHelper:
    """基序可视化辅助类"""
    
    @staticmethod
    def visualize_enzyme_motif_interaction(interaction_result: Dict,
                                            save_path: str,
                                            drug_name: str = "Drug",
                                            enzyme_source: str = "Partner Drug"):
        """可视化酶-基序交互矩阵"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        matrix = np.array(interaction_result.get('interaction_matrix', []))
        enzyme_labels = interaction_result.get('enzyme_labels', [])
        motif_labels = interaction_result.get('motif_labels', [])
        top_interactions = interaction_result.get('top_interactions', [])
        
        if matrix.size == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle(f'{enzyme_source} Enzymes ↔ {drug_name} Functional Groups', 
                    fontsize=14, fontweight='bold')
        
        ax1 = axes[0]
        max_enzymes = min(20, len(enzyme_labels))
        max_motifs = min(25, len(motif_labels))
        
        matrix_display = matrix[:max_enzymes, :max_motifs]
        enzyme_display = enzyme_labels[:max_enzymes]
        motif_display = [l[:25] for l in motif_labels[:max_motifs]]
        
        im = ax1.imshow(matrix_display, cmap='YlOrRd', aspect='auto')
        ax1.set_yticks(range(len(enzyme_display)))
        ax1.set_yticklabels(enzyme_display, fontsize=8)
        ax1.set_xticks(range(len(motif_display)))
        ax1.set_xticklabels(motif_display, rotation=45, ha='right', fontsize=7)
        ax1.set_xlabel('Functional Groups')
        ax1.set_ylabel('Enzymes')
        ax1.set_title('Enzyme-Motif Interaction Heatmap')
        plt.colorbar(im, ax=ax1, label='Interaction Score')
        
        ax2 = axes[1]
        if top_interactions:
            top_n = min(20, len(top_interactions))
            labels = [f"{t['enzyme_name'][:15]} ↔ {t['motif_display_name'][:15]}" 
                     for t in top_interactions[:top_n]]
            scores = [t['normalized_score'] for t in top_interactions[:top_n]]
            
            colors = plt.cm.YlOrRd([s for s in scores])
            ax2.barh(range(len(labels)), scores, color=colors)
            ax2.set_yticks(range(len(labels)))
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel('Normalized Interaction Score')
            ax2.set_title('Top Enzyme-Motif Interactions')
            ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def test_motif_analyzer():
    """Test motif analyzer"""
    print("\n" + "="*80)
    print("Testing Chemical Motif Analyzer - Extended Version")
    print("="*80)
    
    analyzer = ChemicalMotifAnalyzer()
    
    # Print statistics
    stats = analyzer.get_statistics()
    print(f"\nTotal functional groups: {stats['total_patterns']}")
    print(f"Total categories: {stats['total_categories']}")
    print("\nFunctional groups by category:")
    for cat, count in sorted(stats['patterns_by_category'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    # Test molecules
    test_molecules = {
        'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
        'Caffeine': 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
        'Fluconazole': 'OC(Cn1cncn1)(Cn2cncn2)c3ccc(F)cc3F',
        'Ibuprofen': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'Penicillin_G': 'CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O',
        'Metformin': 'CN(C)C(=N)NC(=N)N',
        'Atorvastatin': 'CC(C)c1c(C(=O)Nc2ccccc2)c(c3ccc(F)cc3)n(c1CCC(O)CC(O)CC(=O)O)c4ccccc4',
    }
    
    for name, smiles in test_molecules.items():
        print(f"\n{'='*60}")
        print(f"Molecule: {name}")
        print(f"SMILES: {smiles}")
        
        result = analyzer.extract_motifs_from_smiles(smiles)
        
        if result['mol'] is not None:
            print(f"Atom count: {result['num_atoms']}")
            print(f"Identified {result['motif_count']} functional groups:")
            
            for category, motifs in sorted(result['by_category'].items()):
                print(f"\n  [{category}] ({len(motifs)})")
                for m in motifs[:3]:
                    print(f"    - {m['display_name']}")
                if len(motifs) > 3:
                    print(f"    ... and {len(motifs)-3} more")


if __name__ == '__main__':
    test_motif_analyzer()

