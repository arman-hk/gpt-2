import queue
import sys
import threading
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

from src import data


BATCH = (np.array([[1]], dtype=np.int32), np.array([[2]], dtype=np.int32))


def bounded_call(fn):
    """A regression must fail the test instead of hanging the test process."""
    result = queue.Queue()

    def run():
        try: result.put((True, fn()))
        except BaseException as error: result.put((False, error))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "Data-loader operation hung"
    success, value = result.get_nowait()
    if not success: raise value
    return value


@pytest.mark.parametrize("size", [0, 1, 3])
@pytest.mark.parametrize("count", [0, 3])
def test_prefetch_preserves_order_and_finishes(size, count):
    batches = [tuple(a + i for a in BATCH) for i in range(count)]
    closed = threading.Event()

    def source():
        try: yield from batches
        finally: closed.set()

    actual = bounded_call(lambda: list(data._prefetch(source(), size)))
    assert len(actual) == len(batches)
    assert all(a is b for a, b in zip(actual, batches))
    assert closed.is_set()


@pytest.mark.parametrize("size", [0, 1, 3])
@pytest.mark.parametrize("error", [RuntimeError("download failed"), ValueError("bad tokens"), SystemExit("worker exited")])
def test_prefetch_propagates_original_error_and_traceback(size, error):
    closed = threading.Event()

    def source():
        try:
            yield BATCH
            raise error
        finally:
            closed.set()

    loader = data._prefetch(source(), size)
    try:
        assert bounded_call(lambda: next(loader)) is BATCH
        with pytest.raises(type(error)) as raised:
            bounded_call(lambda: next(loader))
        assert raised.value is error
        assert "source" in [frame.name for frame in traceback.extract_tb(raised.value.__traceback__)]
        assert closed.is_set()
    finally:
        loader.close()


def test_prefetch_propagates_source_close_error():
    error = RuntimeError("source cleanup failed")

    class Source:
        def __next__(self): raise StopIteration
        def close(self): raise error

    with pytest.raises(RuntimeError) as raised:
        bounded_call(lambda: next(data._prefetch(Source(), 1)))
    assert raised.value is error


def test_prefetch_close_cancels_a_full_queue():
    queue_filled, closed = threading.Event(), threading.Event()

    def source():
        try:
            for i in range(100):
                if i == 2: queue_filled.set()
                yield BATCH
        finally:
            closed.set()

    loader = data._prefetch(source(), 1)
    try:
        assert bounded_call(lambda: next(loader)) is BATCH
        # One batch was consumed, one is queued, and a third is being offered.
        assert queue_filled.wait(timeout=5)
    finally:
        bounded_call(loader.close)
    assert closed.is_set()


def test_prefetch_close_does_not_wait_forever_for_an_inflight_read():
    reading, release, closed = threading.Event(), threading.Event(), threading.Event()

    def source():
        try:
            yield BATCH
            reading.set()
            release.wait()
            yield BATCH
        finally:
            closed.set()

    loader = data._prefetch(source(), 1)
    try:
        assert bounded_call(lambda: next(loader)) is BATCH
        assert reading.wait(timeout=5)
        bounded_call(loader.close)
        assert not closed.is_set()
    finally:
        release.set()
        loader.close()
    assert closed.wait(timeout=5)


def test_prefetch_does_not_start_until_consumed():
    started = threading.Event()

    def source():
        started.set()
        yield BATCH

    loader = data._prefetch(source(), 1)
    loader.close()
    assert not started.is_set()


@pytest.mark.parametrize("token_count,prefetch,shard", [
    (0, 1, False), (7, 0, False), (8, 1, False),
    (11, 1, False), (16, 0, False), (16, 1, True),
])
def test_dataloader_packs_tokens_and_drops_only_incomplete_tail(monkeypatch, token_count, prefetch, shard):
    closed = threading.Event()

    def tokens(cfg):
        try: yield from range(token_count)
        finally: closed.set()

    monkeypatch.setattr(data, "_token_stream", tokens)
    monkeypatch.setattr(data.jax, "local_device_count", lambda: 2)
    cfg = data.DataConfig(batch_size=2, seq_len=3, prefetch_size=prefetch)
    batches = bounded_call(lambda: list(data.build_dataloader(cfg, shard=shard)))
    assert len(batches) == token_count // 8
    for i, (xs, ys) in enumerate(batches):
        expected = np.arange(i * 8, (i + 1) * 8, dtype=np.int32).reshape(2, 4)
        shape = (2, 1, 3) if shard else (2, 3)
        np.testing.assert_array_equal(xs, expected[:, :-1].reshape(shape))
        np.testing.assert_array_equal(ys, expected[:, 1:].reshape(shape))
        assert xs.dtype == ys.dtype == np.int32
    assert closed.is_set()


@pytest.mark.parametrize("prefetch", [0, 1])
def test_sharding_error_closes_upstream(monkeypatch, prefetch):
    closed = threading.Event()

    def tokens(cfg):
        try:
            while True: yield from range(8)
        finally:
            closed.set()

    monkeypatch.setattr(data, "_token_stream", tokens)
    monkeypatch.setattr(data.jax, "local_device_count", lambda: 2)
    cfg = data.DataConfig(batch_size=1, seq_len=3, prefetch_size=prefetch)
    with pytest.raises(ValueError, match="batch size must be divisible"):
        bounded_call(lambda: next(data.build_dataloader(cfg, shard=True)))
    assert closed.is_set()


@pytest.fixture
def local_dataset(monkeypatch):
    def install(documents):
        loaded, closed = [], []

        class Dataset:
            def shuffle(self, **kwargs):
                try: yield from documents
                finally: closed.append(True)

        def load_dataset(*args, **kwargs):
            loaded.append(True)
            if len(loaded) > 3:
                pytest.fail("Dataset was repeatedly reopened without producing text")
            return Dataset()

        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
        return loaded, closed

    return install


@pytest.mark.parametrize("documents", [[], [{"text": ""}, {"text": None}]])
def test_empty_dataset_fails_instead_of_restarting(local_dataset, documents):
    loaded, closed = local_dataset(documents)
    with pytest.raises(ValueError, match="no non-empty text"):
        bounded_call(lambda: next(data._text_batches(data.DataConfig())))
    assert len(loaded) == len(closed) == 1


def test_text_batches_repeat_valid_epochs_and_close(local_dataset):
    loaded, closed = local_dataset([{"text": "a"}, {"text": ""}, {"text": "b"}, {"text": "c"}])
    batches = data._text_batches(data.DataConfig(tokenizer_batch_size=2))
    try:
        assert next(batches) == ["a", "b"]
        assert next(batches) == ["c"]
        assert next(batches) == ["a", "b"]
    finally:
        batches.close()
    assert len(loaded) == len(closed) == 2


def test_missing_text_column_fails_and_closes(local_dataset):
    loaded, closed = local_dataset([{"other": "wrong column"}])
    with pytest.raises(KeyError, match="text"):
        bounded_call(lambda: next(data._text_batches(data.DataConfig())))
    assert len(loaded) == len(closed) == 1


def test_tokenizer_error_reaches_dataloader_and_closes_documents(monkeypatch, local_dataset):
    loaded, closed = local_dataset([{"text": "hello"}])
    error = ValueError("tokenization failed")

    class Tokenizer:
        pad_token, bos_token_id, eos_token_id = "<pad>", 0, 1

        def __call__(self, *args, **kwargs): raise error

    tokenizer_factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer_factory))
    cfg = data.DataConfig(batch_size=1, seq_len=2, prefetch_size=1)
    with pytest.raises(ValueError) as raised:
        bounded_call(lambda: next(data.build_dataloader(cfg)))
    assert raised.value is error
    assert len(loaded) == len(closed) == 1


@pytest.mark.parametrize("field", ["batch_size", "seq_len", "tokenizer_batch_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_batch_dimensions_fail_clearly(field, value):
    cfg = data.DataConfig(**{field: value})
    iterator = data._text_batches(cfg) if field == "tokenizer_batch_size" else data._batch_iterator(cfg)
    with pytest.raises(ValueError, match="must be positive"):
        bounded_call(lambda: next(iterator))
