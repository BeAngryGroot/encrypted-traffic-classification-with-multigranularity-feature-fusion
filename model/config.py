"""双分支模型默认配置；训练入口可按版本化实验配置覆盖。"""


class ModelConfig:
    def __init__(self):
        self.micro_d_in = 16
        self.macro_d_in = 12
        self.micro_d_model = 384
        self.micro_layers = 4
        self.d_state = 128
        self.dropout = 0.2
        self.require_official_mamba = True

        self.macro_d_model = 96
        self.macro_layers = 3
        self.macro_heads = 6

        self.fusion_hidden = 192
        self.fusion_mode = "gated"
        self.fixed_fusion_weight = 0.5

        self.num_classes = 8
        self.weight_decay = 3e-4
        self.gamma = 1.0
        self.patience = 10


class DataConfig:
    def __init__(self):
        self.max_seq_len = 64
        self.max_burst_len = 32
        self.feature_dim_micro = 16
        self.feature_dim_macro = 12
        self.alpha = 1.0
