"""
为 dataset_mecddi/drug_smiles.csv 生成新的 motif.txt
使用 RDKit 的 BRICS (Breakage of Retro-synthetically Interesting Chemical Substructures) 算法
按化学键类型将分子切割成有意义的结构片段（保留化学子结构）

输出格式: drug_id,motif_smiles,include
"""

import re
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS


def _clean_brics_fragment(frag_smiles: str) -> str:
    """
    清理 BRICS 片段中的虚拟连接原子标记。
    BRICS 输出如 '[1*]C(=O)[3*]' -> 替换为 '*C(=O)*'
    同时做 SMILES 规范化（去除手性标记等噪音）。
    """
    # 替换所有 [数字*] 形式的虚拟原子为 *
    cleaned = re.sub(r'\[\d+\*\]', '*', frag_smiles)
    return cleaned


def _canonical_smiles(smiles: str) -> str:
    """将 SMILES 规范化为 canonical form（去除手性等冗余信息）。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def main():
    smiles_path = '/home/jy/桌面/xhc/SCE-FENN/dataset_mecddi/drug_smiles.csv'
    output_path = '/home/jy/桌面/xhc/SCE-FENN/dataset_mecddi/motif.txt'

    print("Loading SMILES data...")
    df = pd.read_csv(smiles_path)
    print(f"Loaded {len(df)} drugs from {smiles_path}")

    # 先扫描一遍，统计唯一 fragment 以便排序
    all_unique_frags = set()
    frag_count = 0
    drugs_with_frags = 0
    drugs_without_smiles = 0

    print("\nFirst pass: collecting unique fragments...")
    for idx, row in df.iterrows():
        drug_id = row['drug_id']
        smiles = row.get('smiles', '')
        if pd.isna(smiles) or str(smiles).strip() == '':
            drugs_without_smiles += 1
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        frag_set = BRICS.BRICSDecompose(mol)
        if frag_set:
            drugs_with_frags += 1
            for f in frag_set:
                cleaned = _clean_brics_fragment(f)
                if cleaned:
                    all_unique_frags.add(cleaned)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} drugs, {len(all_unique_frags)} unique fragments so far...")

    # 按 SMILES 字母序排序，统一分配 ID
    sorted_frags = sorted(all_unique_frags)
    frag_to_id = {f: f"BRICS_{i+1:04d}" for i, f in enumerate(sorted_frags)}

    print(f"\nTotal unique BRICS fragments: {len(sorted_frags)}")
    print("Sample fragments (first 20):")
    for f in sorted_frags[:20]:
        print(f"  {frag_to_id[f]}: {f}")

    # 第二遍：生成 motif.txt
    print("\nSecond pass: generating motif.txt...")
    lines = []
    frag_count = 0

    for idx, row in df.iterrows():
        drug_id = row['drug_id']
        smiles = row.get('smiles', '')
        if pd.isna(smiles) or str(smiles).strip() == '':
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        frag_set = BRICS.BRICSDecompose(mol)
        for f in sorted(frag_set):  # 排序保证顺序一致
            cleaned = _clean_brics_fragment(f)
            if cleaned and cleaned in frag_to_id:
                lines.append(f"{drug_id},{cleaned},include")
                frag_count += 1

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} drugs, {frag_count} entries so far...")

    # 写入
    print(f"\nWriting {len(lines)} lines to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

    print("\n=== Summary ===")
    print(f"Total drugs processed: {len(df)}")
    print(f"Drugs with at least one BRICS fragment: {drugs_with_frags}")
    print(f"Drugs without SMILES: {drugs_without_smiles}")
    print(f"Total drug-fragment pairs written: {frag_count}")
    print(f"Unique BRICS fragment types: {len(sorted_frags)}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == '__main__':
    main()
