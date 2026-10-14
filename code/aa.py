try:
    import random
    import Levenshtein
    from Bio import SeqIO
    from collections import defaultdict
    import os
except ImportError as e:
    missing_lib = str(e).split("No module named ")[-1].strip("'")
    print(f"错误：缺少必要库 '{missing_lib}'，请先执行命令安装：")
    install_cmd = {
        "Levenshtein": "pip install python-Levenshtein",
        "Bio": "pip install biopython"
    }.get(missing_lib, f"pip install {missing_lib}")
    print(install_cmd)
    exit()

# -------------------------- 1. 配置文件路径（适配你的本地路径） --------------------------
# 输入：你的非AMP负样本文件
non_amp_input = "data/negative.csv"
# 输出：与AMP同目录（daima文件夹），文件名带non_amp前缀区分
non_amp_train = r"c:/Users/ahdezneytiry-boop1/Desktop/daima/non_amp_train.fasta"  # 非AMP训练集（70%）
non_amp_valid = r"c:/Users/ahdezneytiry-boop1/Desktop/daima/non_amp_valid.fasta"  # 非AMP验证集（20%）
non_amp_test = r"c:/Users/ahdezneytiry-boop1/Desktop/daima/non_amp_test.fasta"    # 非AMP测试集（10%）

print("="*70)
print("开始处理非AMP负样本（negative.csv），标准与AMP正样本一致...")
print(f"非AMP输入文件路径：{non_amp_input}")
print("="*70)

# -------------------------- 2. 解析文件（优先FASTA，兼容表格） --------------------------
standard_aa = set('ACDEFGHIKLMNPQRSTVWY')  # 20种标准氨基酸（与AMP一致）
valid_non_amp = []

try:
    # 尝试按FASTA解析（大概率成功，因之前AMP的CSV实为FASTA）
    print("尝试按FASTA格式解析negative.csv...")
    total_parsed = sum(1 for _ in SeqIO.parse(non_amp_input, "fasta"))
    for record in SeqIO.parse(non_amp_input, "fasta"):
        seq_str = str(record.seq).upper()  # 统一大写，消除大小写干扰
        # 双重筛选：13-100aa + 仅含标准氨基酸（与AMP筛选逻辑完全一致）
        if 13 <= len(seq_str) <= 100 and all(aa in standard_aa for aa in seq_str):
            valid_non_amp.append(record)
    print(f"✅ FASTA解析成功！共读取到 {total_parsed} 条原始非AMP序列")

except ValueError:
    # 若FASTA解析失败，尝试按表格CSV解析（假设列名为"ID"和"Sequence"）
    print("⚠️ FASTA解析失败，尝试按表格CSV解析...")
    try:
        import pandas as pd
        df = pd.read_csv(non_amp_input)
        # 检查是否有必要列（ID和Sequence）
        if not all(col in df.columns for col in ["ID", "Sequence"]):
            raise ValueError("CSV缺少'ID'或'Sequence'列，无法解析")
        
        total_parsed = len(df)
        for idx, row in df.iterrows():
            seq_str = str(row["Sequence"]).upper().strip()
            seq_id = str(row["ID"]).strip()
            # 双重筛选：13-100aa + 仅含标准氨基酸
            if 13 <= len(seq_str) <= 100 and all(aa in standard_aa for aa in seq_str):
                # 转为SeqRecord格式（与FASTA解析结果统一）
                from Bio.SeqRecord import SeqRecord
                from Bio.Seq import Seq
                record = SeqRecord(Seq(seq_str), id=seq_id, description="Non-AMP from negative.csv")
                valid_non_amp.append(record)
        print(f"✅ CSV表格解析成功！共读取到 {total_parsed} 条原始非AMP序列")
    
    except Exception as e:
        print(f"❌ 解析失败：{str(e)}")
        exit()

# -------------------------- 3. 筛选结果日志（与AMP格式对齐） --------------------------
invalid_count = total_parsed - len(valid_non_amp)
print(f"\n1. 非AMP序列筛选结果：")
print(f"   - 原始非AMP序列总数：{total_parsed} 条")
print(f"   - 有效序列数（13-100aa + 仅含标准氨基酸）：{len(valid_non_amp)} 条")
print(f"   - 过滤无效序列数：{invalid_count} 条（长度异常/含非标准氨基酸）")

if len(valid_non_amp) == 0:
    print("❌ 错误：无有效非AMP序列，程序终止！")
    exit()

# -------------------------- 4. 去冗余（≥90%相似度，与AMP逻辑一致） --------------------------
print(f"\n2. 开始去冗余（相似度≥90%，优先保留长序列作为代表）...")
# 按长度分组优化效率（长度差>10%的序列相似度不可能≥90%，跳过比对）
len_groups = defaultdict(list)
for record in valid_non_amp:
    len_groups[len(record.seq)].append(record)

unique_non_amp = []
# 按长度降序处理（模拟CD-HIT逻辑，长序列优先为聚类代表）
sorted_lengths = sorted(len_groups.keys(), reverse=True)
for length in sorted_lengths:
    group_records = len_groups[length]
    for record in group_records:
        seq_str = str(record.seq)
        is_redundant = False
        # 仅与已保存的、长度相近的序列比对
        for rep in unique_non_amp:
            rep_len = len(rep.seq)
            if abs(length - rep_len) / max(length, rep_len) > 0.1:
                continue  # 长度差超10%，跳过
            if Levenshtein.ratio(seq_str, str(rep.seq)) >= 0.90:
                is_redundant = True
                break
        if not is_redundant:
            unique_non_amp.append(record)

# 去冗余结果日志
redundant_count = len(valid_non_amp) - len(unique_non_amp)
print(f"   - 去冗余前非AMP序列数：{len(valid_non_amp)} 条")
print(f"   - 去冗余后非AMP序列数：{len(unique_non_amp)} 条（非冗余）")
print(f"   - 过滤冗余序列数：{redundant_count} 条（相似度≥90%）")

if len(unique_non_amp) == 0:
    print("❌ 错误：去冗余后无剩余非AMP序列，程序终止！")
    exit()

# -------------------------- 5. 7:2:1划分数据集（与AMP比例完全一致） --------------------------
print(f"\n3. 按7:2:1比例划分非AMP数据集...")
random.seed(42)  # 固定随机种子，确保结果可复现（与AMP相同种子）
random.shuffle(unique_non_amp)  # 打乱序列，避免偏差

# 计算各集大小（用“总序列数-训练集-验证集”算测试集，避免小数累计误差）
total_unique = len(unique_non_amp)
train_size = int(total_unique * 0.7)  # 70%训练集
valid_size = int(total_unique * 0.2)  # 20%验证集
test_size = total_unique - train_size - valid_size  # 10%测试集

# 边界处理：若总序列≤10条，确保三集均至少1条（避免空集）
if total_unique <= 10:
    print(f"⚠️  警告：非AMP序列仅{total_unique}条，调整为最小划分单位...")
    train_size = max(1, train_size)
    valid_size = max(1, valid_size)
    test_size = max(1, total_unique - train_size - valid_size)

# 执行划分
non_amp_train_set = unique_non_amp[:train_size]
non_amp_valid_set = unique_non_amp[train_size:train_size + valid_size]
non_amp_test_set = unique_non_amp[train_size + valid_size:]

# 划分结果日志（含实际占比，验证7:2:1）
print(f"   - 非AMP非冗余总序列数：{total_unique} 条")
print(f"   - 非AMP训练集（70%）：{len(non_amp_train_set)} 条（实际占比：{round(len(non_amp_train_set)/total_unique*100, 1)}%）")
print(f"   - 非AMP验证集（20%）：{len(non_amp_valid_set)} 条（实际占比：{round(len(non_amp_valid_set)/total_unique*100, 1)}%）")
print(f"   - 非AMP测试集（10%）：{len(non_amp_test_set)} 条（实际占比：{round(len(non_amp_test_set)/total_unique*100, 1)}%）")

# -------------------------- 6. 导出文件并验证（确保可正常使用） --------------------------
print(f"\n4. 导出非AMP训练集/验证集/测试集...")
try:
    SeqIO.write(non_amp_train_set, non_amp_train, "fasta")
    SeqIO.write(non_amp_valid_set, non_amp_valid, "fasta")
    SeqIO.write(non_amp_test_set, non_amp_test, "fasta")
except PermissionError:
    print(f"❌ 错误：无权限写入文件！请关闭已打开的daima文件夹或文件，重试。")
    exit()

# 验证文件有效性（存在且非空）
train_ok = os.path.exists(non_amp_train) and os.path.getsize(non_amp_train) > 0
valid_ok = os.path.exists(non_amp_valid) and os.path.getsize(non_amp_valid) > 0
test_ok = os.path.exists(non_amp_test) and os.path.getsize(non_amp_test) > 0

# -------------------------- 7. 最终结果汇总 --------------------------
print("="*70)
print("非AMP负样本（negative.csv）处理完成！最终结果：")
print(f"非AMP训练集：{non_amp_train}")
print(f"   - 序列数：{len(non_amp_train_set)} 条 | 用途：模型训练（负例）")
print(f"   - 状态：{'生成成功' if train_ok else '❌ 生成失败'}")
print(f"非AMP验证集：{non_amp_valid}")
print(f"   - 序列数：{len(non_amp_valid_set)} 条 | 用途：模型调优/过拟合监测（负例）")
print(f"   - 状态：{'生成成功' if valid_ok else '❌ 生成失败'}")
print(f"非AMP测试集：{non_amp_test}")
print(f"   - 序列数：{len(non_amp_test_set)} 条 | 用途：模型泛化能力评估（负例）")
print(f"   - 状态：{' 生成成功' if test_ok else '❌ 生成失败'}")
print("="*70)

# 额外：与AMP正样本规模对比（确保均衡）
print(f"\n📊 正负样本规模对比（非冗余后）：")
amp_unique_count = 2402  # 之前AMP去冗余后的序列数（固定值）
print(f"   - AMP正样本非冗余数：{amp_unique_count} 条")
print(f"   - 非AMP负样本非冗余数：{len(unique_non_amp)} 条")
print(f"   - 正负样本比例：1:{round(len(unique_non_amp)/amp_unique_count, 2)}（建议1:0.8~1:1.2，当前{'均衡' if 0.8<=len(unique_non_amp)/amp_unique_count<=1.2 else '需调整'}）")