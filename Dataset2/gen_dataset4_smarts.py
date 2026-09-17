"""
MACCS Fingerprint 生成脚本（isSmarts=True）
输出：drug_id//MACCS{idx}_{smarts}//include
通过 RDKit MACCSkeys.smartsPatts 获取真实的 MACCS SMARTS 定义，
模拟 isSmarts=True 的行为（返回激活 key 对应的 SMARTS 模式）
"""
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MACCSkeys import smartsPatts

# 从 RDKit 获取真实的 MACCS SMARTS 定义（167 个 keys，key 1 和 key 166 为 '?'，无法解析）
MACCS_SMARTS = {idx: smarts for idx, (smarts, _) in smartsPatts.items()}

# 预编译所有 SMARTS 模式，跳过 '?' 和编译失败的
maccs_patterns = {}
for idx, smarts in MACCS_SMARTS.items():
    if smarts == '?':
        continue  # '?' 无法解析
    try:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None:
            maccs_patterns[idx] = pat
    except Exception:
        pass

input_file = "/home/jy/桌面/MSE-ENN/dataset_mecddi/drug_smiles.csv"
output_file = "/home/jy/桌面/MSE-ENN/dataset_mecddi/dataset4_smarts.txt"

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
    for idx, pattern in maccs_patterns.items():
        if mol.HasSubstructMatch(pattern):
            results.append(f"{drug_id}//MACCS{idx}//include")

with open(output_file, "w") as f:
    f.write("\n".join(results) + "\n")

print(f"Done: {len(results)} lines written to {output_file}")
print(f"Total parsed MACCS keys: {len(maccs_patterns)}")
