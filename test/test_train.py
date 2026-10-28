import dataclasses
import pickle
import re
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src import data, model, train


def test_train_config_has_independent_data_defaults():
    first, second = train.TrainConfig(), train.TrainConfig()
    assert isinstance(first.data, data.DataConfig)
    assert first.data is not second.data


def test_lr_schedule_default_boundaries():
    cfg = train.TrainConfig()
    schedule = jax.jit(jax.vmap(train.make_lr_schedule(cfg)))
    rates = schedule(jnp.array([0, 25, 50, 125, 150, 200, 250]))
    assert rates.tolist() == pytest.approx(
        [cfg.base_lr * scale for scale in [0.0, 0.5, 1.0, 0.55, 0.325, 0.1, 0.1]],
    )


@pytest.mark.parametrize("total_steps,warmup_steps", [(100, 50), (3, 2), (20, 0), (1, 0)])
def test_lr_schedule_short_runs_and_no_warmup(total_steps, warmup_steps):
    cfg = train.TrainConfig(total_steps=total_steps, warmup_steps=warmup_steps)
    schedule = train.make_lr_schedule(cfg)
    end_lr = 0.1 * cfg.base_lr
    assert float(schedule(0)) == pytest.approx(0.0 if warmup_steps else cfg.base_lr)
    assert float(schedule(warmup_steps)) == pytest.approx(cfg.base_lr)
    assert float(schedule(total_steps - 1)) > end_lr
    assert float(schedule(total_steps)) == pytest.approx(end_lr)
    assert float(schedule(total_steps + 10)) == pytest.approx(end_lr)
    rates = jax.vmap(schedule)(jnp.arange(total_steps + 1))
    assert bool(jnp.isfinite(rates).all())
    assert bool((jnp.diff(rates[:warmup_steps + 1]) >= 0).all())
    assert bool((jnp.diff(rates[warmup_steps:]) <= 0).all())


@pytest.mark.parametrize("total_steps,warmup_steps,message", [
    (0, 0, "total_steps must be positive"),
    (-1, 0, "total_steps must be positive"),
    (10, -1, "warmup_steps must satisfy"),
    (50, 50, "warmup_steps must satisfy"),
    (10, 11, "warmup_steps must satisfy"),
])
def test_lr_schedule_rejects_invalid_durations(total_steps, warmup_steps, message):
    cfg = train.TrainConfig(total_steps=total_steps, warmup_steps=warmup_steps)
    with pytest.raises(ValueError, match=message):
        train.make_lr_schedule(cfg)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("grad_accum_steps", [1, 2, 4])
def test_train_on_local_batches(monkeypatch, tmp_path, capsys, dtype, grad_accum_steps):
    cfg = train.TrainConfig(
        total_steps=4,
        warmup_steps=1,
        base_lr=0.01,
        grad_accum_steps=grad_accum_steps,
        log_every=2,
        save_every=4,
        save_checkpoints=True,
        ckpt_dir=str(tmp_path),
        use_wandb=False,
        gpt=model.GPTConfig(
            vocab_size=16, ctx_len=4, n_layer=1, n_head=2,
            d_model=8, dropout_rate=0.0, dtype=dtype,
        ),
        data=data.DataConfig(batch_size=4, seq_len=4, prefetch_size=1),
    )
    xs = jnp.array([[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13], [15, 0, 1, 2]], dtype=jnp.int32)
    ys = (xs + 1) % cfg.gpt.vocab_size
    closed = threading.Event()

    def tokens(config):
        try:
            while True: yield from (i % cfg.gpt.vocab_size for i in range(20))
        finally:
            closed.set()

    monkeypatch.setattr(data, "_token_stream", tokens)
    initial, forward, _, _ = train.init_train_state(cfg, jax.random.PRNGKey(cfg.seed))
    initial_loss = train.xent_loss(forward(initial.params, xs, initial.rng_key), ys)

    train.train(cfg)
    assert closed.is_set()
    logged = re.findall(r"step\s+(\d+).*?lr\s+(\S+)", capsys.readouterr().out)
    assert [int(step) for step, _ in logged] == [1, 2, 4]
    assert [float(lr) for _, lr in logged] == pytest.approx([0.0, 0.01, 0.00325])

    with (tmp_path / "step_4.pkl").open("rb") as f:
        checkpoint = pickle.load(f)
    assert checkpoint["step"] == cfg.total_steps
    assert int(checkpoint["opt_state"].gradient_step) == cfg.total_steps
    assert int(checkpoint["opt_state"].mini_step) == 0
    params = jax.tree.map(
        lambda old, new: jnp.asarray(new, dtype=old.dtype),
        initial.params, checkpoint["master_params"],
    )
    final_loss = train.xent_loss(forward(params, xs, initial.rng_key), ys)
    assert bool(jnp.isfinite(final_loss))
    assert float(final_loss) < float(initial_loss)


@pytest.mark.parametrize("error", [RuntimeError("update failed"), KeyboardInterrupt()])
def test_train_closes_loader_if_update_fails(monkeypatch, error):
    closed = threading.Event()

    def tokens(config):
        try:
            while True: yield from range(4)
        finally:
            closed.set()

    def fail_update(*args): raise error

    monkeypatch.setattr(data, "_token_stream", tokens)
    monkeypatch.setattr(train, "init_train_state", lambda *args: (None, None, None, None))
    monkeypatch.setattr(train, "compile_train_step", lambda *args: fail_update)
    cfg = train.TrainConfig(use_wandb=False, data=data.DataConfig(batch_size=1, seq_len=3, prefetch_size=1))
    with pytest.raises(type(error)) as raised:
        train.train(cfg)
    assert raised.value is error
    assert closed.is_set()


@pytest.mark.parametrize("accum_steps,batch_size,message", [
    (0, 4, "grad_accum_steps must be a positive integer"),
    (-1, 4, "grad_accum_steps must be a positive integer"),
    (1.5, 4, "grad_accum_steps must be a positive integer"),
    ("2", 4, "grad_accum_steps must be a positive integer"),
    (True, 4, "grad_accum_steps must be a positive integer"),
    (1, 0, "data.batch_size must be a positive integer"),
    (1, -2, "data.batch_size must be a positive integer"),
    (1, 2.5, "data.batch_size must be a positive integer"),
    (1, True, "data.batch_size must be a positive integer"),
    (2, 3, "data.batch_size must be divisible"),
    (4, 2, "data.batch_size must be divisible"),
])
def test_invalid_accumulation_fails_before_model_allocation(monkeypatch, accum_steps, batch_size, message):
    def unexpected_build(*args, **kwargs): pytest.fail("Allocated the model before validating accumulation")

    monkeypatch.setattr(model, "build", unexpected_build)
    cfg = train.TrainConfig(grad_accum_steps=accum_steps, data=data.DataConfig(batch_size=batch_size))
    with pytest.raises(ValueError, match=re.escape(message)):
        train.init_train_state(cfg, jax.random.PRNGKey(0))


@pytest.mark.parametrize("xs_shape,ys_shape", [((3, 2), (3, 2)), ((4, 2), (3, 2)), ((4,), (4,))])
def test_invalid_batch_fails_before_any_microbatch_update(monkeypatch, xs_shape, ys_shape):
    closed = threading.Event()

    def batches(*args, **kwargs):
        try: yield jnp.zeros(xs_shape, dtype=jnp.int32), jnp.zeros(ys_shape, dtype=jnp.int32)
        finally: closed.set()

    def unexpected_update(*args): pytest.fail("Applied an update to an invalid batch")

    monkeypatch.setattr(train, "init_train_state", lambda *args: (None, None, None, None))
    monkeypatch.setattr(train, "compile_train_step", lambda *args: unexpected_update)
    monkeypatch.setattr(train, "build_dataloader", batches)
    cfg = train.TrainConfig(grad_accum_steps=2, use_wandb=False, data=data.DataConfig(batch_size=4))
    with pytest.raises(ValueError, match="batch size 4"):
        train.train(cfg)
    assert closed.is_set()


@pytest.mark.parametrize("accum_steps", [2, 4])
@pytest.mark.parametrize("grad_clip_norm", [1.0, 0.05])
def test_accumulated_updates_match_full_batch(accum_steps, grad_clip_norm):
    # A small float32 softmax model isolates accumulation from dropout and attention numerics.
    params = {
        "w": 0.1 * jax.random.normal(jax.random.PRNGKey(0), (16, 16)),
        "b": jnp.zeros((16,), dtype=jnp.float32),
    }

    def forward(p, idx, key, *, train): return p["w"][idx] + p["b"]

    cfg = train.TrainConfig(
        total_steps=4, warmup_steps=1, base_lr=0.01,
        grad_clip_norm=grad_clip_norm, data=data.DataConfig(batch_size=4),
    )
    schedule = train.make_lr_schedule(cfg)

    def initialize(k):
        config = dataclasses.replace(cfg, grad_accum_steps=k)
        opt = train.make_optim(config, params, schedule)
        state = train.TrainState(params, params, opt.init(params), jax.random.PRNGKey(1))
        return state, train.compile_train_step(config, forward, opt)

    full, full_step = initialize(1)
    accumulated, micro_step = initialize(accum_steps)
    xs = jnp.arange(12, dtype=jnp.int32).reshape(4, 3)
    ys = (xs + 1) % 16
    micro_size = 4 // accum_steps
    for update in range(cfg.total_steps):
        full = full_step(full, (xs, ys))
        before = accumulated.master_params
        for i in range(accum_steps):
            sl = slice(i * micro_size, (i + 1) * micro_size)
            accumulated = micro_step(accumulated, (xs[sl], ys[sl]))
            completed = i == accum_steps - 1
            assert int(accumulated.opt_state.gradient_step) == update + int(completed)
            assert int(accumulated.opt_state.mini_step) == (0 if completed else i + 1)
            if not completed:
                for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(accumulated.master_params)):
                    np.testing.assert_array_equal(old, new)
        # Float32 reductions can differ slightly when grouped into microbatches.
        for expected, actual in zip(jax.tree.leaves(full.master_params), jax.tree.leaves(accumulated.master_params)):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
        for expected, actual in zip(
            jax.tree.leaves(full.opt_state.inner_opt_state),
            jax.tree.leaves(accumulated.opt_state.inner_opt_state),
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
        assert int(full.opt_state.gradient_step) == update + 1
        assert all(bool((g == 0).all()) for g in jax.tree.leaves(accumulated.opt_state.acc_grads))
