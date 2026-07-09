#!/usr/bin/env python
"""
supplement_micro_samples_enhanced.py - 增强版微观样本补充工具（20维特征）
--------------------------------------------------------
基于原有代码，修改为支持20维学术标准特征提取
主要改进：
1. 真实的20维微观特征提取（替代随机生成）
2. 真实的20维宏观特征处理
3. 时间窗口划分支持
4. 保持原有路径配置
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import logging
from collections import defaultdict, Counter
import pickle
import gc
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from scipy.stats import entropy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局函数用于并行处理
def process_flow_batch_worker(args):
    """并行处理工作函数 - 修改为20维特征"""
    flow_batch, pkt_data_dict, max_seq_len, strategy = args
    
    batch_results = {
        'extended_flows': [],
        'micro_sequences': [],
        'micro_masks': []
    }
    
    for flow_id, group_data in flow_batch:
        group_size = len(group_data)
        
        # 获取包数据
        flow_packets = pkt_data_dict.get(flow_id, [])
        
        # 生成变体 - 使用真实特征
        try:
            if len(flow_packets) == 0:
                variants = create_dummy_variants_20d(group_size, max_seq_len)
            else:
                variants = create_real_variants_20d(flow_packets, group_size, max_seq_len, strategy)
        except:
            variants = create_dummy_variants_20d(group_size, max_seq_len)
        
        # 处理组内每行数据
        for i, row_data in enumerate(group_data):
            # 添加变体ID
            extended_row = row_data.copy()
            extended_row['variant_id'] = i
            batch_results['extended_flows'].append(extended_row)
            
            # 添加对应的序列
            if i < len(variants):
                seq, mask = variants[i]
            else:
                seq, mask = variants[-1]
            
            batch_results['micro_sequences'].append(seq)
            batch_results['micro_masks'].append(mask)
    
    return batch_results

def create_dummy_variants_20d(num_variants, max_seq_len):
    """生成20维虚拟特征变体（仅作为后备）"""
    variants = []
    
    for i in range(num_variants):
        seq = np.zeros((max_seq_len, 20), dtype=np.float32)
        
        # 生成更合理的虚拟特征模式
        for j in range(min(10 + i, max_seq_len)):
            # 1. 对数包长度
            seq[j, 0] = np.random.uniform(0.1, 0.8)
            # 2. 方向
            seq[j, 1] = np.random.choice([0, 1])
            # 3. 带方向长度
            seq[j, 2] = seq[j, 0] * (1 if seq[j, 1] == 0 else -1)
            # 4. 对数IAT
            seq[j, 3] = np.random.uniform(0.0, 0.5)
            # 5-7. 突发特征
            seq[j, 4] = np.random.uniform(0.0, 0.3)
            seq[j, 5] = np.random.uniform(0.0, 0.2)
            seq[j, 6] = np.random.uniform(0.0, 0.4)
            # 8-11. K包摘要
            seq[j, 7] = np.random.uniform(0.1, 0.6)
            seq[j, 8] = np.random.uniform(0.3, 0.9)
            seq[j, 9] = np.random.uniform(0.0, 0.8)
            seq[j, 10] = np.random.uniform(0.1, 0.8)
            # 12-14. TCP标志
            if j < 3:
                seq[j, 11] = 1.0 if j == 0 else 0.0  # SYN
                seq[j, 12] = 1.0  # ACK
                seq[j, 13] = 0.0  # FIN/RST
            # 15-16. 位置特征
            seq[j, 14] = j / max_seq_len
            seq[j, 15] = 1.0 if j < 5 else 0.0
            # 17-20. 窗口特征
            seq[j, 16] = np.random.uniform(0.0, 0.5)
            seq[j, 17] = np.random.uniform(0.0, 0.3)
            seq[j, 18] = np.random.uniform(0.0, 0.2)
            seq[j, 19] = np.random.uniform(0.6, 0.9)
        
        mask = np.zeros(max_seq_len, dtype=np.float32)
        mask[:min(10 + i, max_seq_len)] = 1.0
        
        variants.append((seq, mask))
    
    return variants

def create_real_variants_20d(packets_data, num_variants, max_seq_len, strategy='time_split'):
    """从真实包数据创建20维特征变体"""
    # 首先提取真实的20维特征
    real_features = extract_real_micro_features_20d(packets_data, max_seq_len * 2)  # 提取更多包
    
    if len(real_features) == 0:
        return create_dummy_variants_20d(num_variants, max_seq_len)
    
    variants = []
    
    if strategy == 'time_split' and num_variants > 1:
        # 时间分割策略
        total_packets = len(real_features)
        
        for i in range(num_variants):
            if total_packets <= max_seq_len:
                # 包太少，直接使用
                variant_features = real_features.copy()
            else:
                # 分段采样
                segment_size = total_packets // num_variants
                start_idx = i * segment_size
                end_idx = min(start_idx + max_seq_len, total_packets)
                
                if start_idx >= total_packets:
                    # 随机采样
                    indices = np.random.choice(total_packets, max_seq_len, replace=True)
                    variant_features = real_features[indices]
                else:
                    variant_features = real_features[start_idx:end_idx]
            
            # 填充或截断到固定长度
            seq, mask = pad_or_truncate_sequence(variant_features, max_seq_len)
            variants.append((seq, mask))
    else:
        # 简单策略：所有变体使用相同特征
        seq, mask = pad_or_truncate_sequence(real_features, max_seq_len)
        for i in range(num_variants):
            variants.append((seq.copy(), mask.copy()))
    
    return variants

def extract_real_micro_features_20d(packets_data, max_packets=1000):
    """从包数据中提取真实的20维微观特征"""
    if not packets_data:
        return np.array([]).reshape(0, 20)
    
    # 限制包数量
    packets_data = packets_data[:max_packets]
    n_packets = len(packets_data)
    
    if n_packets == 0:
        return np.array([]).reshape(0, 20)
    
    # 提取基本信息
    times = []
    lengths = []
    directions = []
    flags_list = []
    
    for pkt in packets_data:
        times.append(pkt.get('packet_time', 0))
        lengths.append(pkt.get('packet_length', 0))
        # 从micro_feature_* 列读取方向，或使用默认值
        directions.append(pkt.get('direction', 0))
        flags_list.append(pkt.get('flags', ''))
    
    # 计算IAT
    iats = [0.0]  # 第一个包IAT为0
    for i in range(1, len(times)):
        iats.append(max(0, times[i] - times[i-1]))
    
    # 特征矩阵
    features = np.zeros((n_packets, 20), dtype=np.float32)
    
    for i in range(n_packets):
        # 如果包数据已经包含micro_feature_*列，直接使用
        feature_found = False
        for j in range(20):
            feature_key = f'micro_feature_{j}'
            if feature_key in packets_data[i]:
                features[i, j] = float(packets_data[i][feature_key])
                feature_found = True
        
        if feature_found:
            continue  # 已经有预计算的特征，跳过计算
        
        # 否则从原始数据计算特征
        pkt_len = lengths[i]
        direction = directions[i]
        iat = iats[i]
        flags_str = flags_list[i]
        
        # 1. 对数包长度
        if pkt_len > 0:
            features[i, 0] = np.log10(pkt_len + 1) / 4.0
        
        # 2. 方向
        features[i, 1] = float(direction)
        
        # 3. 带方向长度
        features[i, 2] = (pkt_len / 1500.0) * (1 if direction == 0 else -1)
        
        # 4. 对数IAT
        if iat > 0:
            features[i, 3] = np.log10(iat * 1000 + 1) / 3.0
        
        # 5-7. 突发特征（需要至少3个包）
        if i >= 2:
            # 5. 当前突发大小
            burst_size = 1
            for j in range(i-1, -1, -1):
                if directions[j] == direction:
                    burst_size += 1
                else:
                    break
            features[i, 4] = min(burst_size / 10.0, 1.0)
            
            # 6. 小间隔计数
            recent_iats = iats[max(0, i-4):i+1]
            small_iat_count = sum(1 for x in recent_iats if x < 0.01)
            features[i, 5] = small_iat_count / 5.0
            
            # 7. 方向变化数
            recent_dirs = directions[max(0, i-9):i+1]
            direction_changes = sum(1 for j in range(1, len(recent_dirs)) 
                                  if recent_dirs[j] != recent_dirs[j-1])
            features[i, 6] = direction_changes / 10.0
        
        # 8-11. 早期K包摘要（需要至少5个包）
        if i >= 4:
            recent_lengths = lengths[max(0, i-4):i+1]
            
            # 8-9. 长度分位数
            if len(recent_lengths) >= 3:
                features[i, 7] = np.percentile(recent_lengths, 25) / 1500.0
                features[i, 8] = np.percentile(recent_lengths, 75) / 1500.0
            
            # 10. 长度变化次数
            length_changes = sum(1 for j in range(1, len(recent_lengths)) 
                               if abs(recent_lengths[j] - recent_lengths[j-1]) > 100)
            features[i, 9] = length_changes / 4.0
            
            # 11. Run-length
            recent_dirs = directions[max(0, i-4):i+1]
            run_length = 1
            for j in range(len(recent_dirs)-1, 0, -1):
                if recent_dirs[j] == recent_dirs[-1]:
                    run_length += 1
                else:
                    break
            features[i, 10] = min(run_length / 5.0, 1.0)
        
        # 12-14. TCP标志
        features[i, 11] = 1.0 if 'SYN' in flags_str else 0.0
        features[i, 12] = 1.0 if 'ACK' in flags_str else 0.0
        features[i, 13] = 1.0 if ('FIN' in flags_str or 'RST' in flags_str) else 0.0
        
        # 15-16. 位置特征
        features[i, 14] = i / max(n_packets-1, 1)
        features[i, 15] = 1.0 if i < 5 else 0.0
        
        # 17-20. 窗口特征（需要至少3个包）
        if i >= 2:
            recent_lens = lengths[max(0, i-2):i+1]
            
            # 17. 长度熵
            if len(set(recent_lens)) > 1:
                len_counts = Counter(recent_lens)
                probs = [count/len(recent_lens) for count in len_counts.values()]
                features[i, 16] = entropy(probs) / 2.0
            
            # 18. 小包比例
            small_count = sum(1 for x in recent_lens if x < 100)
            features[i, 17] = small_count / len(recent_lens)
            
            # 19. 大包比例
            large_count = sum(1 for x in recent_lens if x > 1000)
            features[i, 18] = large_count / len(recent_lens)
            
            # 20. 载荷比例估计
            total_payload = sum(max(0, x-20) for x in recent_lens)
            total_length = sum(recent_lens)
            if total_length > 0:
                features[i, 19] = total_payload / total_length
    
    return features

def pad_or_truncate_sequence(features, max_seq_len):
    """填充或截断序列到固定长度"""
    n_packets, n_features = features.shape
    
    # 创建填充后的序列
    seq = np.zeros((max_seq_len, n_features), dtype=np.float32)
    mask = np.zeros(max_seq_len, dtype=np.float32)
    
    # 填充数据
    actual_len = min(n_packets, max_seq_len)
    seq[:actual_len] = features[:actual_len]
    mask[:actual_len] = 1.0
    
    return seq, mask

class OptimizedMicroSampleSupplementor:
    def __init__(self, root_csv_dir: str, max_seq_len: int = 64, n_workers: int = None):
        self.root_csv_dir = Path(root_csv_dir)
        self.max_seq_len = max_seq_len
        self.n_workers = n_workers or min(4, cpu_count())
        self.flow_df = None
        self.pkt_df = None
        self.label_encoders = {}
        
    def discover_csv_files(self):
        """发现所有CSV文件"""
        logger.info(f"扫描目录: {self.root_csv_dir}")
        
        flow_csv_files = []
        for csv_file in self.root_csv_dir.rglob("*.csv"):
            if not csv_file.name.endswith(".pkt.csv"):
                flow_csv_files.append(csv_file)
        
        logger.info(f"找到 {len(flow_csv_files)} 个流级CSV文件")
        
        flow_pkt_pairs = []
        for flow_csv in flow_csv_files:
            pkt_csv = flow_csv.with_suffix(".pkt.csv")
            flow_pkt_pairs.append((flow_csv, pkt_csv if pkt_csv.exists() else None))
        
        return flow_pkt_pairs
    
    def load_data(self):
        """优化的数据加载"""
        logger.info("开始批量加载数据...")
        
        file_pairs = self.discover_csv_files()
        if not file_pairs:
            raise FileNotFoundError(f"在目录 {self.root_csv_dir} 中未找到CSV文件")
        
        all_flow_dfs = []
        all_pkt_dfs = []
        
        # 批量加载，减少内存峰值
        for flow_csv, pkt_csv in tqdm(file_pairs, desc="加载文件"):
            try:
                # 加载流数据
                flow_df = pd.read_csv(flow_csv)
                if len(flow_df) == 0 or 'flow_id' not in flow_df.columns:
                    continue
                
                # 添加文件信息和推断标签
                flow_df['source_file'] = flow_csv.name
                if 'class_label' not in flow_df.columns:
                    flow_df['class_label'] = self._infer_class_label(flow_csv.name)
                if 'activity_label' not in flow_df.columns:
                    flow_df['activity_label'] = self._infer_activity_label(flow_csv.name)
                
                all_flow_dfs.append(flow_df)
                
                # 加载包数据
                if pkt_csv and pkt_csv.exists():
                    try:
                        pkt_df = pd.read_csv(pkt_csv)
                        if len(pkt_df) > 0 and 'flow_id' in pkt_df.columns:
                            valid_flow_ids = set(flow_df['flow_id'].values)
                            pkt_df = pkt_df[pkt_df['flow_id'].isin(valid_flow_ids)]
                            if len(pkt_df) > 0:
                                all_pkt_dfs.append(pkt_df)
                    except Exception as e:
                        logger.warning(f"包文件加载失败 {pkt_csv.name}: {e}")
                
                # 定期清理内存
                if len(all_flow_dfs) % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"文件加载失败 {flow_csv.name}: {e}")
                continue
        
        if not all_flow_dfs:
            raise ValueError("未成功加载任何数据")
        
        # 合并数据
        logger.info("合并数据...")
        self.flow_df = pd.concat(all_flow_dfs, ignore_index=True)
        
        if all_pkt_dfs:
            self.pkt_df = pd.concat(all_pkt_dfs, ignore_index=True)
        else:
            self.pkt_df = None
        
        logger.info(f"加载完成: 流数据 {len(self.flow_df)} 行")
        if self.pkt_df is not None:
            logger.info(f"包数据 {len(self.pkt_df)} 行")
    
    def _infer_class_label(self, filename: str) -> str:
        """从文件名推断类别标签"""
        filename_lower = filename.lower()
        if 'nonvpn' in filename_lower:
            return 'NONVPN'
        elif 'vpn' in filename_lower:
            return 'VPN'
        return 'NONVPN'
    
    def _infer_activity_label(self, filename: str) -> str:
        """从文件名推断活动标签"""
        filename_lower = filename.lower()
        activity_patterns = {
            'AUDIO': ['audio', 'music', 'voice'],
            'BROWSING': ['browsing', 'web', 'http'],
            'CHAT': ['chat', 'message', 'im'],
            'FILE': ['file', 'download', 'upload'],
            'VIDEO': ['video', 'stream', 'youtube'],
            'EMAIL': ['email', 'mail'],
            'P2P': ['p2p', 'torrent'],
            'VOIP': ['voip', 'sip', 'call'],
        }
        
        for activity, keywords in activity_patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return activity
        return 'UNKNOWN'
    
    def analyze_duplicates(self):
        """分析重复情况"""
        flow_id_counts = self.flow_df['flow_id'].value_counts()
        duplicated_flow_ids = flow_id_counts[flow_id_counts > 1]
        
        logger.info(f"重复分析:")
        logger.info(f"  总行数: {len(self.flow_df)}")
        logger.info(f"  唯一flow_id: {len(flow_id_counts)}")
        logger.info(f"  重复flow_id: {len(duplicated_flow_ids)}")
        
        return duplicated_flow_ids
    
    def supplement_micro_features(self, output_dir: str, strategy: str = 'time_split'):
        """增强版特征补充 - 支持20维特征"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始20维特征补充...")
        
        duplicated_flow_ids = self.analyze_duplicates()
        if len(duplicated_flow_ids) == 0:
            logger.info("无重复flow_id，无需补充")
            return
        
        # 添加唯一ID
        self.flow_df['unique_id'] = range(len(self.flow_df))
        
        # 预处理包数据为字典格式以提高查找效率
        logger.info("构建包数据索引...")
        pkt_data_dict = {}
        if self.pkt_df is not None:
            for flow_id, group in self.pkt_df.groupby('flow_id'):
                # 转换为简单字典列表
                pkt_data_dict[flow_id] = group.to_dict('records')
        
        # 准备并行处理数据
        logger.info("准备并行处理...")
        flow_groups = []
        for flow_id, group in self.flow_df.groupby('flow_id'):
            # 转换为简单数据结构
            group_data = []
            for _, row in group.iterrows():
                group_data.append(row.to_dict())
            flow_groups.append((flow_id, group_data))
        
        # 分批处理
        batch_size = max(50, len(flow_groups) // (self.n_workers * 4))
        batches = []
        for i in range(0, len(flow_groups), batch_size):
            batch = flow_groups[i:i+batch_size]
            batches.append((batch, pkt_data_dict, self.max_seq_len, strategy))
        
        logger.info(f"使用 {self.n_workers} 个进程，分 {len(batches)} 批处理...")
        
        # 并行处理
        all_results = {
            'extended_flows': [],
            'micro_sequences': [],
            'micro_masks': []
        }
        
        if self.n_workers == 1:
            # 单进程处理（用于调试）
            for batch_args in tqdm(batches, desc="处理批次"):
                result = process_flow_batch_worker(batch_args)
                for key in all_results:
                    all_results[key].extend(result[key])
        else:
            # 多进程处理
            with Pool(processes=self.n_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(process_flow_batch_worker, batches),
                    total=len(batches),
                    desc="并行处理"
                ))
            
            # 合并结果
            for result in batch_results:
                for key in all_results:
                    all_results[key].extend(result[key])
        
        logger.info("转换数据格式...")
        
        # 转换为最终格式
        extended_flow_df = pd.DataFrame(all_results['extended_flows'])
        micro_seq = np.array(all_results['micro_sequences'], dtype=np.float32)
        micro_mask = np.array(all_results['micro_masks'], dtype=np.float32)
        
        logger.info(f"处理完成:")
        logger.info(f"  流数据: {len(extended_flow_df)} 行")
        logger.info(f"  微观序列: {micro_seq.shape}")
        
        # 处理宏观特征 - 修改为20维
        logger.info("生成20维宏观特征...")
        macro_bag, macro_mask = self._process_macro_features_20d(extended_flow_df)
        
        # 提取标签
        logger.info("提取标签...")
        labels, label_mappings = self._extract_labels(extended_flow_df)
        
        # 标准化
        logger.info("标准化特征...")
        micro_seq = self._standardize_features(micro_seq)
        
        # 保存文件
        logger.info("保存文件...")
        self._save_results(output_path, macro_bag, macro_mask, micro_seq, micro_mask, 
                          labels, label_mappings, extended_flow_df)
        
        logger.info(f"✅ 20维特征补充完成! 保存在: {output_path}")
    
    def _process_macro_features_20d(self, extended_flow_df, max_bag_size=100):
        """处理20维宏观特征"""
        # 20维宏观特征列名（对应增强pcap_to_csv的输出）
        macro_feature_names = [
            'duration_norm', 'fwd_packets_norm', 'bwd_packets_norm',
            'fwd_bytes_norm', 'bwd_bytes_norm', 'fwd_len_mean', 'bwd_len_mean',
            'len_iqr', 'len_std', 'iat_mean', 'iat_std',
            'fwd_pkt_ratio', 'bwd_pkt_ratio', 'fwd_byte_ratio',
            'burst_count', 'max_burst_size', 'avg_burst_size',
            'byte_rate', 'packet_rate', 'tcp_ratio'
        ]
        
        # 检查哪些特征列存在
        available_features = [col for col in macro_feature_names if col in extended_flow_df.columns]
        
        if not available_features:
            # 如果没有预计算的特征，使用原有的数值特征
            logger.warning("未找到20维宏观特征，使用备用特征")
            numeric_features = [
                'duration', 'total_fwd_packets', 'total_bwd_packets',
                'total_length_fwd_packets', 'total_length_bwd_packets',
                'flow_bytes_per_sec', 'flow_packets_per_sec'
            ]
            available_features = [col for col in numeric_features if col in extended_flow_df.columns]
            
            if not available_features:
                # 最后的备用方案
                available_features = extended_flow_df.select_dtypes(include=[np.number]).columns.tolist()
                available_features = [col for col in available_features 
                                    if col not in ['unique_id', 'variant_id']]
        
        if not available_features:
            # 创建虚拟特征
            logger.warning("创建虚拟宏观特征")
            features = np.random.rand(len(extended_flow_df), 20).astype(np.float32)
        else:
            # 使用可用特征
            features = extended_flow_df[available_features].values
            features = np.nan_to_num(features, nan=0.0)
            
            # 标准化
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            
            # 如果特征不足20维，进行填充或截断
            if features.shape[1] < 20:
                # 填充到20维
                padded_features = np.zeros((features.shape[0], 20), dtype=np.float32)
                padded_features[:, :features.shape[1]] = features
                features = padded_features
            elif features.shape[1] > 20:
                # 截断到20维
                features = features[:, :20]
        
        # 生成特征袋
        num_flows = len(features)
        
        macro_bag = np.zeros((num_flows, max_bag_size, 20), dtype=np.float32)
        macro_mask = np.zeros((num_flows, max_bag_size), dtype=np.float32)
        
        macro_bag[:, 0, :] = features
        macro_mask[:, 0] = 1.0
        
        logger.info(f"宏观特征形状: {macro_bag.shape}")
        return macro_bag, macro_mask
    
    def _extract_labels(self, extended_flow_df):
        """提取标签"""
        primary_labels = []
        secondary_labels = []
        
        for _, row in extended_flow_df.iterrows():
            primary = str(row.get('class_label', 'NONVPN')).upper()
            secondary = str(row.get('activity_label', 'UNKNOWN')).upper()
            
            primary_labels.append(primary)
            secondary_labels.append(secondary)
        
        combined_labels = [f"{p}-{s}" for p, s in zip(primary_labels, secondary_labels)]
        
        # 构建映射
        unique_primary = sorted(list(set(primary_labels)))
        unique_secondary = sorted(list(set(secondary_labels)))
        unique_combined = sorted(list(set(combined_labels)))
        
        label_mappings = {
            'primary': {
                'labels': unique_primary,
                'label_to_id': {label: i for i, label in enumerate(unique_primary)},
                'num_classes': len(unique_primary)
            },
            'secondary': {
                'labels': unique_secondary,
                'label_to_id': {label: i for i, label in enumerate(unique_secondary)},
                'num_classes': len(unique_secondary)
            },
            'combined': {
                'labels': unique_combined,
                'label_to_id': {label: i for i, label in enumerate(unique_combined)},
                'num_classes': len(unique_combined)
            }
        }
        
        # 转换为ID
        primary_ids = np.array([label_mappings['primary']['label_to_id'][label] for label in primary_labels])
        secondary_ids = np.array([label_mappings['secondary']['label_to_id'][label] for label in secondary_labels])
        combined_ids = np.array([label_mappings['combined']['label_to_id'][label] for label in combined_labels])
        
        labels = {
            'primary': primary_ids,
            'secondary': secondary_ids,
            'combined': combined_ids
        }
        
        return labels, label_mappings
    
    def _standardize_features(self, micro_seq):
        """标准化微观特征"""
        for feat_idx in range(micro_seq.shape[2]):
            feat_data = micro_seq[:, :, feat_idx]
            if np.std(feat_data) > 1e-8:
                micro_seq[:, :, feat_idx] = (feat_data - np.mean(feat_data)) / np.std(feat_data)
        return micro_seq
    
    def _save_results(self, output_path, macro_bag, macro_mask, micro_seq, micro_mask, 
                     labels, label_mappings, extended_flow_df):
        """保存所有结果"""
        np.save(output_path / "macro_bag.npy", macro_bag.astype(np.float32))
        np.save(output_path / "macro_mask.npy", macro_mask.astype(np.float32))
        np.save(output_path / "micro_seq.npy", micro_seq.astype(np.float32))
        np.save(output_path / "micro_mask.npy", micro_mask.astype(np.float32))
        
        np.save(output_path / "primary_labels.npy", labels['primary'])
        np.save(output_path / "secondary_labels.npy", labels['secondary'])
        np.save(output_path / "combined_labels.npy", labels['combined'])
        np.save(output_path / "labels.npy", labels['primary'])
        
        with open(output_path / "label_mappings.pkl", "wb") as f:
            pickle.dump(label_mappings, f)
        
        with open(output_path / "cat2id.pkl", "wb") as f:
            pickle.dump(label_mappings['primary']['label_to_id'], f)
        
        extended_flow_df.to_csv(output_path / "extended_flows.csv", index=False)
        
        # 保存配置信息
        config_info = {
            'feature_dims': {
                'micro_d_in': 20,  # 微观特征维度
                'macro_d_in': 20,  # 宏观特征维度
                'num_classes': len(label_mappings['primary']['labels'])
            },
            'dataset_stats': {
                'total_samples': len(labels['primary']),
                'vpn_samples': int(np.sum(labels['primary'] == 1)) if 'VPN' in label_mappings['primary']['labels'] else 0,
                'nonvpn_samples': int(np.sum(labels['primary'] == 0)) if 'NONVPN' in label_mappings['primary']['labels'] else len(labels['primary'])
            }
        }
        
        with open(output_path / "config_info.pkl", "wb") as f:
            pickle.dump(config_info, f)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="增强版微观样本补充工具（20维特征）")
    parser.add_argument("--input_dir", default="/data3/wsb_workspace/study/data/data_test/data_vpn_flow",
                       help="输入CSV目录")
    parser.add_argument("--output_dir", default="/data3/wsb_workspace/study/models/flowcls-test/features",
                       help="输出特征目录")
    parser.add_argument("--strategy", choices=['time_split', 'mixed'], default='time_split',
                       help="补充策略")
    parser.add_argument("--max_seq_len", type=int, default=32, help="最大序列长度")
    parser.add_argument("--n_workers", type=int, default=16, help="并行进程数")
    
    args = parser.parse_args()
    
    logger.info("增强版微观样本补充工具（20维特征）")
    logger.info("=" * 50)
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"并行进程: {args.n_workers}")
    logger.info(f"特征维度: 微观20维 + 宏观20维")
    
    supplementor = OptimizedMicroSampleSupplementor(
        args.input_dir, 
        args.max_seq_len, 
        args.n_workers
    )
    
    supplementor.load_data()
    supplementor.supplement_micro_features(args.output_dir, args.strategy)
    
    logger.info("✅ 20维特征处理完成!")


if __name__ == "__main__":
    main()