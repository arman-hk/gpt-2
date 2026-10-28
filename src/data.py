import dataclasses
from itertools import islice
from typing import Iterator, Optional, Tuple
import threading, queue
import jax, jax.numpy as jnp, numpy as np

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

def _close_iterator(it: Iterator) -> None:
    close = getattr(it, "close", None)
    if close is not None: close()

def _text_batches(cfg: DataConfig) -> Iterator[list[str]]:
    if cfg.tokenizer_batch_size <= 0:
        raise ValueError("tokenizer_batch_size must be positive")
    from datasets import load_dataset
    while True:
        ds = load_dataset(cfg.dataset_name, split="train", streaming=True, trust_remote_code=True)
        docs = iter(ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer_size))
        texts = filter(None, (ex[cfg.text_column] for ex in docs))
        found_text = False
        try:
            while batch := list(islice(texts, cfg.tokenizer_batch_size)):
                found_text = True
                yield batch
        finally:
            _close_iterator(docs)
        if not found_text:
            raise ValueError(f"Dataset {cfg.dataset_name!r} has no non-empty text in column {cfg.text_column!r}")

def _token_stream(cfg: DataConfig) -> Iterator[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    if tok.pad_token is None: tok.add_special_tokens({"pad_token": tok.eos_token or tok.bos_token or "<|pad|>"})
    tok.model_max_length = int(1e9)
    bos_id, eos_id = (tok.bos_token_id if cfg.add_bos_token else None, tok.eos_token_id if cfg.add_eos_token else None)
    batches = _text_batches(cfg)
    try:
        for batch in batches:
            for ids in tok(batch, add_special_tokens=False, return_attention_mask=False, truncation=False)["input_ids"]:
                if bos_id is not None: yield bos_id
                yield from map(int, ids)
                if eos_id is not None: yield eos_id
    finally:
        _close_iterator(batches)

def _batch_iterator(cfg: DataConfig) -> Iterator[Batch]:
    if cfg.batch_size <= 0 or cfg.seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    stream, b, t, span = _token_stream(cfg), cfg.batch_size, cfg.seq_len, cfg.batch_size * (cfg.seq_len + 1)
    try:
        while True:
            tokens = np.fromiter(islice(stream, span), dtype=np.int32)
            if tokens.size < span: return  # Drop an incomplete final batch.
            arr = tokens.reshape(b, t + 1)
            yield jnp.asarray(arr[:, :t], dtype=jnp.int32), jnp.asarray(arr[:, 1:], dtype=jnp.int32)
    finally:
        _close_iterator(stream)

def _prefetch(it: Iterator[Batch], size: int) -> Iterator[Batch]:
    """Prefetch lazily; close cancels queued work, not an in-flight source read."""
    if size <= 0: return it
    def batches():
        q = queue.Queue(maxsize=size)
        stopped, done = threading.Event(), object()

        def put(item):
            while not stopped.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def worker():
            try:
                while not stopped.is_set():
                    try: item = next(it)
                    except StopIteration: break
                    if not put(item): break
            except BaseException as error:
                put(error)
            finally:
                try: _close_iterator(it)
                except BaseException as error: put(error)
                put(done)

        thread = threading.Thread(target=worker, name="gpt2-prefetch", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is done: return
                if isinstance(item, BaseException): raise item
                yield item
        finally:
            stopped.set()
            thread.join(timeout=1.0)
    return batches()

def _shard(batch: Batch) -> Batch:
    xs, ys, n = *batch, jax.local_device_count()
    if xs.shape[0] % n: raise ValueError("batch size must be divisible by local_device_count()")
    return tuple(a.reshape((n, xs.shape[0] // n) + a.shape[1:]) for a in (xs, ys))

def build_dataloader(cfg: DataConfig, *, shard: bool = False, prefetch: Optional[int] = None) -> Iterator[Batch]:
    it = _prefetch(_batch_iterator(cfg), (cfg.prefetch_size if prefetch is None else prefetch) or 0)
    try:
        for batch in it:
            yield _shard(batch) if shard else batch
    finally:
        _close_iterator(it)
