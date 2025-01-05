from dataclasses import dataclass

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta/configuration_deberta.py

# https://github.com/microsoft/DeBERTa/blob/master/experiments/language_model/rtd_xsmall.json


@dataclass
class DebertaConfig:
    # batch_size: int = 16
    vocab_size: int = 5
    hidden_size: int = 384
    # CLS + board_size
    max_position_embeddings: int = 37
    hidden_dropout_prob: float = 0.1
    num_attention_heads: int = 6
    attention_probs_dropout_probs: float = 0.1
    intermediate_size: int = 1536
    num_hidden_layers: int = 6
    mask_rate: float = 0.15
    layer_norm_eps: float = 1e-7
    # pooler_hidden_size : int = 384
    # talking_head: bool = False
    initializer_range: float = 0.02
    absolute_position_embeddings: bool = True
    warmup_init: bool = False
