import time
from contextlib import contextmanager
from functools import lru_cache
import math

import torch
from loguru import logger


@contextmanager
def measure_time(name: str, disabled: bool = False):
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if not disabled:
        logger.info(f"{name}: {elapsed_time:.4f} seconds")


infcache = lru_cache(maxsize=None)


def cantor(num1, num2):
    return (num1 + num2) * (num1 + num2 + 1) // 2 + num2


def cantor_decode(num):
    w = math.floor((math.sqrt(8 * num + 1) - 1) / 2)
    t = (w * w + w) // 2
    y = num - t
    x = w - y
    return x, y


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
