import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ===================== 1. 基础配置与样式 =====================
# 字体与基础样式 (参考给定风格)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"] 
plt.rcParams["axes.unicode_minus"] = False

# 配色方案与图例
COLORS = ["#c0627a", "#4a72bf", "#e0bc82"]
DATASET_NAMES = ["train", "valid", "test"]

# 路径配置
BASE_PATH = r"E:\26大三\kcsj\daima"
FASTA_FILES = {name: os.path.join(BASE_PATH, f"{name}.fasta") for name in DATASET_NAMES}

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# ===================== 2. 特征提取模块 (优化版) =====================
def calculate_features(seq, max_len=50):
    """一次性计算 AAC、理化性质 和 One-Hot 编码"""
    length = len(seq)
    
    # 1. AAC
    aac = np.zeros(len(AMINO_ACIDS))
    for aa in seq:
        if aa in AA_TO_IDX:
            aac[AA_TO_IDX[aa]] += 1
    aac = aac / length if length > 0 else aac
    
    # 2. 理化性质
    hydropathy = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 
        'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 
        'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    charge = {'K': 1, 'R': 1, 'H': 1, 'D': -1, 'E': -1}
    
    hydro = np.mean([hydropathy.get(aa, 0) for aa in seq]) if length > 0 else 0
    net_charge = sum([charge.get(aa, 0) for aa in seq])
    
    # 3. One-Hot (展平)
    one_hot = np.zeros((max_len, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            one_hot[i, AA_TO_IDX[aa]] = 1
            
    return np.concatenate([aac, [hydro, net_charge, length], one_hot.flatten()])

def process_dataset(fasta_path):
    """读取数据并转换为 DataFrame"""
    if not os.path.exists(fasta_path):
        return pd.DataFrame()
        
    features = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = ''.join([c for c in str(record.seq).upper() if c in AMINO_ACIDS])
        if seq:
            features.append(calculate_features(seq))
            
    # 构建列名
    aac_cols = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    phys_cols = ["Hydrophobicity", "Net_Charge", "Length"]
    oh_cols = [f"OH_{i}_{aa}" for i in range(50) for aa in AMINO_ACIDS]
    
    return pd.DataFrame(features, columns=aac_cols + phys_cols + oh_cols)

# ===================== 3. 主执行与可视化 =====================
if __name__ == "__main__":
    print("=== 抗菌肽特征提取与分布可视化 ===")
    
    plot_data = [] # 收集用于作图的二维特征
    
    # --- 数据处理 ---
    for name in DATASET_NAMES:
        fasta_path = FASTA_FILES[name]
        df = process_dataset(fasta_path)
        
        # 兼容处理：如果没有找到文件，生成模拟数据以便演示可视化代码
        if df.empty:
            print(f"[警告] 未找到文件 {fasta_path}，正使用模拟数据演示可视化...")
            np.random.seed(len(name)) # 保证每次随机固定
            hydro_mock = np.random.randn(80) * 0.8 + (1 if name=="train" else -0.5)
            charge_mock = np.random.randn(80) * 2 + (0 if name=="valid" else 1.5)
            plot_data.append((name, hydro_mock, charge_mock))
        else:
            save_path = os.path.join(BASE_PATH, f"{name}_features.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ [{name}] 特征提取完成 | 保存至: {save_path}")
            plot_data.append((name, df["Hydrophobicity"].values, df["Net_Charge"].values))

    # --- 绘图 (完全参照给定风格) ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # 寻找全局最小极值，用于绘制边缘辅助标记（Rug plot）
    all_x = np.concatenate([x for _, x, _ in plot_data])
    all_y = np.concatenate([y for _, _, y in plot_data])
    min_x, min_y = np.min(all_x) - 0.5, np.min(all_y) - 1.5

    # 绘图循环
    for i, (name, x_data, y_data) in enumerate(plot_data):
        c = COLORS[i % len(COLORS)]
        lbl = f"数据集: {name}"
        
        # 绘制主散点
        ax.scatter(x_data, y_data, s=100, c=c, alpha=0.6, label=lbl)
        
        # 绘制边缘辅助标记 (贴合底部与左侧)
        ax.scatter(x_data, [min_y] * len(x_data), marker="|", color=c, alpha=0.5, s=300)
        ax.scatter([min_x] * len(y_data), y_data, marker="_", color=c, alpha=0.5, s=300)

    # 坐标轴与布局设置
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    # 移除顶部和右侧边框让画面更清爽
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("平均疏水性 (Hydrophobicity)", fontweight='bold')
    ax.set_ylabel("序列净电荷 (Net Charge)", fontweight='bold')
    ax.set_title("抗菌肽特征分布空间", pad=20)

    # 图例设置
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.05), frameon=False)
    
    plt.tight_layout()
    plt.show()