# test_data_attention_metadata.py
import torch
from dataclasses import dataclass, fields
from typing import Optional, Dict, Set, Any
from unittest.mock import Mock

# 关键修改：用 Mock 模拟 MultiModalPlaceholderMap.IndexMap
class MultiModalPlaceholderMap:
    IndexMap = Mock  # 忽略参数，避免 TypeError

# 原始 AttentionMetadata 抽象类（保持不变）
@dataclass
class AttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    multi_modal_placeholder_index_maps: Optional[Dict[str, "MultiModalPlaceholderMap.IndexMap"]]
    enable_kv_scales_calculation: bool

    @property
    def prefill_metadata(self) -> Optional["AttentionMetadata"]:
        raise NotImplementedError

    @property
    def decode_metadata(self) -> Optional["AttentionMetadata"]:
        raise NotImplementedError

    def asdict_zerocopy(self, skip_fields: Optional[Set[str]] = None) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }

# 具体子类（保持不变）
@dataclass
class ConcreteAttentionMetadata(AttentionMetadata):
    @property
    def prefill_metadata(self) -> Optional["AttentionMetadata"]:
        return self if self.num_prefill_tokens > 0 else None

    @property
    def decode_metadata(self) -> Optional["AttentionMetadata"]:
        return self if self.num_decode_tokens > 0 else None

# 数据工厂函数（保持结构，MultiModal 用 Mock）
def create_prefill_only_metadata(
    num_prefills: int = 1,
    num_prefill_tokens: int = 3,
    block_size: int = 16,
    has_multi_modal: bool = False,
    enable_kv_scales: bool = True
) -> ConcreteAttentionMetadata:
    slot_mapping = torch.tensor(
        [i * block_size + (i % block_size) for i in range(num_prefill_tokens)],
        dtype=torch.int64
    )
    
    multi_modal_maps = None
    if has_multi_modal:
        # 使用 Mock 创建 IndexMap，忽略参数
        multi_modal_maps = {
            "image_placeholder": MultiModalPlaceholderMap.IndexMap()
        }
    
    return ConcreteAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=0,
        slot_mapping=slot_mapping,
        multi_modal_placeholder_index_maps=multi_modal_maps,
        enable_kv_scales_calculation=enable_kv_scales
    )

def create_decode_only_metadata(
    num_decode_tokens: int = 2,
    block_size: int = 16,
    has_multi_modal: bool = True,
    enable_kv_scales: bool = False
) -> ConcreteAttentionMetadata:
    base_slot = 5
    slot_mapping = torch.tensor(
        [base_slot + i * block_size for i in range(num_decode_tokens)],
        dtype=torch.int64
    )
    
    multi_modal_maps = None
    if has_multi_modal:
        multi_modal_maps = {
            "image_placeholder": MultiModalPlaceholderMap.IndexMap(),
            "audio_placeholder": MultiModalPlaceholderMap.IndexMap()
        }
    
    return ConcreteAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=num_decode_tokens,
        slot_mapping=slot_mapping,
        multi_modal_placeholder_index_maps=multi_modal_maps,
        enable_kv_scales_calculation=enable_kv_scales
    )

def create_mixed_metadata(
    num_prefills: int = 2,
    num_prefill_tokens: int = 4,
    num_decode_tokens: int = 2,
    block_size: int = 16,
    enable_kv_scales: bool = True
) -> ConcreteAttentionMetadata:
    prefill_slots = [i * block_size + (i % block_size) for i in range(num_prefill_tokens)]
    decode_slots = [block_size * num_prefill_tokens + i for i in range(num_decode_tokens)]
    slot_mapping = torch.tensor(prefill_slots + decode_slots, dtype=torch.int64)
    
    multi_modal_maps = {
        "image_placeholder": MultiModalPlaceholderMap.IndexMap(),
        "text_placeholder": MultiModalPlaceholderMap.IndexMap()
    }
    
    return ConcreteAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        slot_mapping=slot_mapping,
        multi_modal_placeholder_index_maps=multi_modal_maps,
        enable_kv_scales_calculation=enable_kv_scales
    )

# 预设测试数据
TEST_DATA_PREFILL_ONLY = create_prefill_only_metadata()
TEST_DATA_DECODE_ONLY = create_decode_only_metadata()
TEST_DATA_MIXED = create_mixed_metadata()

if __name__ == "__main__":
    print("=== Prefill Only Data ===")
    print(TEST_DATA_PREFILL_ONLY)
    
    print("\n=== Decode Only Data ===")
    print(TEST_DATA_DECODE_ONLY)
    
    print("\n=== Mixed Data ===")
    print(TEST_DATA_MIXED)