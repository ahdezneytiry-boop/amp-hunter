try:
    import random
    import Levenshtein
    from Bio import SeqIO
    from collections import defaultdict
    import os
except ImportError as e:
    missing_lib = str(e).split("No module named ")[-1].strip("'")
    print(f"错误：缺少必要库 '{missing_lib}'，请先执行以下命令安装：")
    install_cmd = {
        "Levenshtein": "pip install python-Levenshtein",
        "Bio": "pip install biopython"
    }.get(missing_lib, f"pip install {missing_lib}")
    print(install_cmd)
    exit()

# 2. 配置文件路径（新增验证集路径，适配你的文件位置）
input_file = "data/amp.csv"
train_file = r"c:/Users/ahdezneytiry-boop1/Desktop/daima/train.fasta"    # 训练集（70%）
valid_file = r"c:/Users/ahdezneytiry-boop1/Desktop/daima/valid.fasta"    # 验证集（20%）- 新增
test_file = r"c:/Users/ahdezneytiry-boop1/Desktop/daima/test.fasta"      # 测试集（10%）

print("="*70)
print("开始处理AMP序列（按7:2:1比例划分训练集/验证集/测试集）...")
print(f"输入文件路径：{input_file}")
print("="*70)

# 3. 解析FASTA+过滤无效序列（长度13-100 + 标准氨基酸，逻辑不变）
standard_aa = set('ACDEFGHIKLMNPQRSTVWY')  # 20种标准氨基酸
valid_records = []
try:
    # 统计原始序列总数（避免重复解析）
    total_parsed = sum(1 for _ in SeqIO.parse(input_file, "fasta"))
    # 筛选有效序列（双重校验：长度+氨基酸有效性）
    for record in SeqIO.parse(input_file, "fasta"):
        seq_str = str(record.seq).upper()  # 统一大写，消除大小写干扰
        if 13 <= len(seq_str) <= 100 and all(aa in standard_aa for aa in seq_str):
            valid_records.append(record)
except FileNotFoundError:
    print(f"❌ 错误：输入文件 '{input_file}' 不存在，请检查路径是否正确！")
    exit()
except ValueError:
    print(f"❌ 错误：输入文件 '{input_file}' 不是标准FASTA格式（可能含乱码或非序列内容）！")
    exit()

# 输出筛选日志（保持原格式，清晰展示无效序列过滤情况）
invalid_count = total_parsed - len(valid_records)
print(f"1. 序列筛选结果：")
print(f"   - 原始序列总数：{total_parsed} 条")
print(f"   - 有效序列数（13-100aa + 仅含标准氨基酸）：{len(valid_records)} 条")
print(f"   - 过滤无效序列数：{invalid_count} 条（含长度异常、非标准氨基酸序列）")

if len(valid_records) == 0:
    print("❌ 错误：无有效序列可处理，程序终止！")
    exit()

# 4. 去冗余（≥90%相似度，按长度分组优化效率，逻辑不变）
print(f"\n2. 开始去冗余（相似度≥90%，优先保留长序列作为代表）...")
# 按长度分组：减少跨长度比对（长度差>10%的序列相似度不可能≥90%）
len_groups = defaultdict(list)
for record in valid_records:
    len_groups[len(record.seq)].append(record)

# 按长度降序处理，模拟CD-HIT聚类逻辑
unique_records = []
sorted_lengths = sorted(len_groups.keys(), reverse=True)  # 长度从大到小排序
for length in sorted_lengths:
    group_records = len_groups[length]
    for record in group_records:
        seq_str = str(record.seq)
        is_redundant = False
        # 仅与长度相近的已保存序列比对（减少计算量）
        for rep in unique_records:
            rep_len = len(rep.seq)
            if abs(length - rep_len) / max(length, rep_len) > 0.1:
                continue  # 长度差超10%，跳过比对
            if Levenshtein.ratio(seq_str, str(rep.seq)) >= 0.90:
                is_redundant = True
                break
        if not is_redundant:
            unique_records.append(record)

# 输出去冗余日志
redundant_count = len(valid_records) - len(unique_records)
print(f"   - 去冗余前序列数：{len(valid_records)} 条")
print(f"   - 去冗余后序列数：{len(unique_records)} 条（非冗余序列）")
print(f"   - 过滤冗余序列数：{redundant_count} 条（相似度≥90%的重复序列）")

if len(unique_records) == 0:
    print("❌ 错误：去冗余后无剩余序列，程序终止！")
    exit()

# 5. 核心修改：按7:2:1划分训练集、验证集、测试集
print(f"\n3. 按7:2:1比例划分数据集...")
random.seed(42)  # 固定随机种子，确保结果可复现（科研必备）
random.shuffle(unique_records)  # 打乱序列，避免按长度/ID排序导致的偏差

# 计算各集大小（按总非冗余序列数的比例取整，确保比例精准）
total_unique = len(unique_records)
train_size = int(total_unique * 0.7)  # 训练集：70%
valid_size = int(total_unique * 0.2)  # 验证集：20%
# 测试集：剩余10%（用总序列数减去训练集+验证集，避免小数累计误差）
test_size = total_unique - train_size - valid_size

# 边界处理：若总序列数少（≤10条），确保各集至少1条（避免空集）
if total_unique <= 10:
    print(f"⚠️  警告：非冗余序列仅{total_unique}条（不足10条），调整为最小划分单位：")
    train_size = max(1, train_size)  # 训练集至少1条
    valid_size = max(1, valid_size)  # 验证集至少1条
    test_size = max(1, total_unique - train_size - valid_size)  # 测试集至少1条

# 执行划分（切片索引：训练集→验证集→测试集）
train_set = unique_records[:train_size]
valid_set = unique_records[train_size:train_size + valid_size]
test_set = unique_records[train_size + valid_size:]

# 输出划分详情（新增各集占比计算，直观验证7:2:1比例）
print(f"   - 去冗余后总序列数：{total_unique} 条")
print(f"   - 训练集（70%）：{len(train_set)} 条（实际占比：{round(len(train_set)/total_unique*100, 1)}%）")
print(f"   - 验证集（20%）：{len(valid_set)} 条（实际占比：{round(len(valid_set)/total_unique*100, 1)}%）")
print(f"   - 测试集（10%）：{len(test_set)} 条（实际占比：{round(len(test_set)/total_unique*100, 1)}%）")

# 6. 导出三集文件并验证（新增验证集导出，覆盖所有文件的权限与存在性检查）
print(f"\n4. 导出训练集、验证集、测试集文件...")
try:
    SeqIO.write(train_set, train_file, "fasta")  # 导出训练集
    SeqIO.write(valid_set, valid_file, "fasta")  # 导出验证集（新增）
    SeqIO.write(test_set, test_file, "fasta")    # 导出测试集
except PermissionError:
    print(f"❌ 错误：无权限写入文件！请检查目标文件夹是否可编辑（如关闭已打开的文件）。")
    exit()

# 验证所有文件的生成状态（检查是否存在且非空）
train_ok = os.path.exists(train_file) and os.path.getsize(train_file) > 0
valid_ok = os.path.exists(valid_file) and os.path.getsize(valid_file) > 0  # 新增验证集检查
test_ok = os.path.exists(test_file) and os.path.getsize(test_file) > 0

# 最终结果汇总（突出7:2:1比例，明确各集用途）
print("="*70)
print("AMP序列处理完成！最终结果（7:2:1训练集/验证集/测试集）：")
print(f"训练集文件：{train_file}")
print(f"序列数量：{len(train_set)} 条 | 用途：模型训练")
print(f"文件状态：{'生成成功' if train_ok else '❌ 生成失败'}")
print(f"验证集文件：{valid_file}")  # 新增验证集信息
print(f"序列数量：{len(valid_set)} 条 | 用途：模型参数调优/过拟合监测")
print(f"文件状态：{'生成成功' if valid_ok else '❌ 生成失败'}")
print(f"测试集文件：{test_file}")
print(f"序列数量：{len(test_set)} 条 | 用途：模型泛化能力评估")
print(f"文件状态：{'生成成功' if test_ok else '❌ 生成失败'}")
print("="*70)