import pickle

import jax
import jax.numpy as jnp
import pytest

from src import data, model, train


def test_train_config_has_independent_data_defaults():
    first, second = train.TrainConfig(), train.TrainConfig()
    assert isinstance(first.data, data.DataConfig)
    assert first.data is not second.data


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
