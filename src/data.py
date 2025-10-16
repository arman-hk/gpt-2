import dataclasses
from itertools import islice
from typing import Iterator, Optional, Tuple
import threading, queue
import jax
import jax.numpy as jnp
import numpy as np

Batch = Tuple[jnp.ndarray, jnp.ndarray]

@dataclasses.dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "Skylion007/openwebtext"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    add_bos_token: bool = False
    add_eos_token: bool = True
    seq_len: int = 1024
    batch_size: int = 8
    shuffle_buffer_size: int = 50_000
    seed: int = 0
    prefetch_size: int = 4
    tokenizer_batch_size: int = 64

def _load_tokenizer(cfg: DataConfig):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": tok.eos_token or tok.bos_token or "<|pad|>"})
    tok.model_max_length = int(1e9)
    return tok, tok.bos_token_id if cfg.add_bos_token else None, tok.eos_token_id if cfg.add_eos_token else None

def _text_batches(cfg: DataConfig) -> Iterator[list[str]]:
    from datasets import load_dataset
    while True:
        ds = load_dataset(cfg.dataset_name, split="train", streaming=True, trust_remote_code=True)
        yield from iter(lambda: list(islice(filter(None, (ex.get(cfg.text_column) for ex in ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer_size))), cfg.tokenizer_batch_size)), [])

def _token_stream(cfg: DataConfig) -> Iterator[int]:
    tok, bos_id, eos_id = _load_tokenizer(cfg)
    for batch in _text_batches(cfg):
        for ids in tok(batch, add_special_tokens=False, return_attention_mask=False, truncation=False)["input_ids"]:
            if bos_id is not None: yield bos_id
            yield from map(int, ids)
            if eos_id is not None: yield eos_id

def _batch_iterator(cfg: DataConfig) -> Iterator[Batch]:
    stream, b, t, span = _token_stream(cfg), cfg.batch_size, cfg.seq_len, cfg.batch_size * (cfg.seq_len + 1)
    while True:
        if (tokens := np.fromiter(islice(stream, span), dtype=np.int32, count=span)).size == span:
            arr = tokens.reshape(b, t + 1)
            yield jnp.asarray(arr[:, :t], dtype=jnp.int32), jnp.asarray(arr[:, 1:], dtype=jnp.int32)

def _prefetch(it: Iterator[Batch], size: int) -> Iterator[Batch]:
    if size <= 0: return it
    q = queue.Queue(maxsize=size)
    def worker():
        for item in it: q.put(item)
        q.put(None)
    threading.Thread(target=worker, daemon=True).start()
    return iter(q.get, None)

def _shard(batch: Batch) -> Batch:
    xs, ys, n = *batch, jax.local_device_count()
    if xs.shape[0] % n: raise ValueError("batch size must be divisible by local_device_count()")
    return tuple(a.reshape((n, xs.shape[0] // n) + a.shape[1:]) for a in (xs, ys))

def build_dataloader(cfg: DataConfig, *, shard: bool = False, prefetch: Optional[int] = None) -> Iterator[Batch]:
    it = _prefetch(_batch_iterator(cfg), (cfg.prefetch_size if prefetch is None else prefetch) or 0)
    return (_shard(batch) for batch in it) if shard else it