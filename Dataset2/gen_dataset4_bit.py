"""
MACCS Fingerprint 生成脚本（isSmarts=False）
输出：drug_id//MACCS{bit}//include
使用 RDKit MACCSkeys.GenMACCSKeys 生成位向量指纹（默认方式）
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys

input_file = "/home/jy/桌面/MSE-ENN/dataset_mecddi/drug_smiles.csv"
output_file = "/home/jy/桌面/MSE-ENN/dataset_mecddi/dataset4.txt"

df = pd.read_csv(input_file)
results = []

for _, row in df.iterrows():
    drug_id = row["drug_id"]
    smiles = row["smiles"]
    if pd.isna(smiles) or str(smiles).strip() == "":
        continue
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        continue
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    for i in range(1, 168):  # MACCS key 编号从 1 开始，共 167 个（key 1 固定为 0）
        if maccs_fp[i - 1]:
            results.append(f"{drug_id}//MACCS{i}//include")

with open(output_file, "w") as f:
    f.write("\n".join(results) + "\n")

print(f"Done: {len(results)} lines written to {output_file}")
