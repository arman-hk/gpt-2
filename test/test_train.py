import pickle

import jax
import jax.numpy as jnp
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
def test_train_on_local_batches(monkeypatch, tmp_path, dtype):
    cfg = train.TrainConfig(
        total_steps=4,
        warmup_steps=1,
        base_lr=0.01,
        log_every=2,
        save_every=4,
        save_checkpoints=True,
        ckpt_dir=str(tmp_path),
        use_wandb=False,
        gpt=model.GPTConfig(
            vocab_size=16, ctx_len=4, n_layer=1, n_head=2,
            d_model=8, dropout_rate=0.0, dtype=dtype,
        ),
        data=data.DataConfig(batch_size=2, seq_len=4, prefetch_size=0),
    )
    xs = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
    ys = xs + 1
    monkeypatch.setattr(
        train, "build_dataloader",
        lambda config, *, shard: iter([(xs, ys)] * cfg.total_steps),
    )
    initial, forward, _, _ = train.init_train_state(cfg, jax.random.PRNGKey(cfg.seed))
    initial_loss = train.xent_loss(forward(initial.params, xs, initial.rng_key), ys)

    train.train(cfg)

    with (tmp_path / "step_4.pkl").open("rb") as f:
        checkpoint = pickle.load(f)
    assert checkpoint["step"] == cfg.total_steps
    assert int(checkpoint["opt_state"].gradient_step) == cfg.total_steps
    params = jax.tree.map(
        lambda old, new: jnp.asarray(new, dtype=old.dtype),
        initial.params, checkpoint["master_params"],
    )
    final_loss = train.xent_loss(forward(params, xs, initial.rng_key), ys)
    assert bool(jnp.isfinite(final_loss))
    assert float(final_loss) < float(initial_loss)
