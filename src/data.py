import dataclasses
from typing import Iterator, Tuple, List, Optional
import jax
import jax.numpy as jnp

@dataclasses.dataclass
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

def _load_tokenizer(name: str, add_bos: bool, add_eos: bool):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        if tok.eos_token is None:
            tok.add_special_tokens({"eos_token": ""})
        tok.pad_token = tok.eos_token
    tok.model_max_length = int(1e9)  # avoid truncation warnings; we handle packing
    bos_id = tok.bos_token_id if add_bos else None
    eos_id = tok.eos_token_id if add_eos else None
    return tok, bos_id, eos_id

def _text_stream(cfg: DataConfig) -> Iterator[str]:
    from datasets import load_dataset
    while True:
        ds = load_dataset(cfg.dataset_name, split="train", streaming=True, trust_remote_code=True)
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer_size)
        for ex in ds:
            t = ex.get(cfg.text_column)
            if t:
                yield t

def _token_id_stream(cfg: DataConfig) -> Iterator[int]:
    tok, bos_id, eos_id = _load_tokenizer(cfg.tokenizer_name, cfg.add_bos_token, cfg.add_eos_token)
    mb: List[str] = []
    mb_size = 64
    for txt in _text_stream(cfg):
        mb.append(txt)
        if len(mb) < mb_size:
            continue
        enc = tok(
            mb,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )["input_ids"]
        mb.clear()
        for ids in enc:
            if bos_id is not None:
                yield bos_id
            for t in ids:
                yield int(t)
            if eos_id is not None:
                yield eos_id

def _batch_iterator(cfg: DataConfig) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    B, T = cfg.batch_size, cfg.seq_len
    stream = _token_id_stream(cfg)
    need = B * (T + 1)
    while True:
        flat = [next(stream) for _ in range(need)]
        arr = jnp.asarray(flat, dtype=jnp.int32).reshape(B, T + 1)
        # to avoids double-donation errors when using donate_argnums in JIT.
        xs = arr[:, :T].copy()
        ys = arr[:, 1:].copy()
        yield xs, ys

def _prefetch(it: Iterator[Tuple[jnp.ndarray, jnp.ndarray]], size: int) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    import threading, queue
    q: "queue.Queue" = queue.Queue(maxsize=size)
    def worker():
        try:
            for x in it:
                q.put(x)
        finally:
            q.put(None)
    threading.Thread(target=worker, daemon=True).start()
    while True:
        x = q.get()
        if x is None:
            return
        yield x

def _shard(batch: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xs, ys = batch
    n = jax.local_device_count()
    assert xs.shape[0] % n == 0, "batch size must be divisible by local_device_count()"
    def r(x):
        return x.reshape((n, x.shape[0] // n) + x.shape[1:])
    return r(xs), r(ys)

def build_dataloader(cfg: DataConfig, *, shard: bool = False, prefetch: Optional[int] = None) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    it = _batch_iterator(cfg)
    pf = cfg.prefetch_size if prefetch is None else prefetch
    if pf and pf > 0:
        it = _prefetch(it, pf)
    if shard:
        def gen():
            for b in it:
                yield _shard(b)
        return gen()
    return it