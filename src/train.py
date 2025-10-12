import dataclasses
from typing import Any, Tuple, NamedTuple, Optional, Callable
import jax
import jax.numpy as jnp
import optax
from . import model, data as data_mod
from functools import partial

from jax import config as jax_config
jax_config.update("jax_default_matmul_precision", "highest")

# ---------- (1) configs and train state ------------
@dataclasses.dataclass
class TrainConfig:
    total_steps: int = 200  # currently set for debug
    save_every: int = 100  # currently set for debug
    log_every: int = 10  # currently set for debug
    seed: int = 0
    # AdamW
    base_lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    # schedule / stability
    warmup_steps: int = 50
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    # runtime / system
    ckpt_dir: str = "ckpts"
    save_checkpoints: bool = False
    # logging
    use_wandb: bool = True
    wandb_project: str = "gpt-speedrun"
    wandb_run_name: Optional[str] = None
    # other configs
    gpt: model.GPTConfig = dataclasses.field(
        default_factory=lambda: model.GPTConfig(
            vocab_size=50257, ctx_len=128, n_layer=2, n_head=2, d_model=64, dropout_rate=0.0, dtype=jnp.bfloat16
        )  # small debug config; bf16 params for TPU
    )
    data: data_mod.DataConfig = dataclasses.field(
        default_factory=lambda: data_mod.DataConfig(batch_size=8, seq_len=128)
    )

class TrainState(NamedTuple):
    params: model.ModelParams  # bf16 params used for forward/backprop
    master_params: model.ModelParams  # f32 master copy used by optimizer
    opt_state: Any
    step: jnp.ndarray
    rng_key: jax.Array

def init_train_state(cfg: TrainConfig, key: jax.Array) -> Tuple[TrainState, Callable[..., jnp.ndarray]]:
    k_init, k_state = jax.random.split(key, 2)
    params, forward = model.build(cfg.gpt, key=k_init)
    master_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
    opt = make_optim(cfg, master_params)
    state = TrainState(
        params=params,
        master_params=master_params,
        opt_state=opt.init(master_params),
        step=jnp.asarray(0, dtype=jnp.int32),
        rng_key=k_state,
    )
    return state, forward

# ---------- (2) optim ------------
def lr_schedule(s: jnp.ndarray, warmup: int, total: int, base: float) -> jnp.ndarray:
    # linear warmup -> cosine decay to 0.1*base
    w = base * s / jnp.maximum(1, warmup)
    prog = jnp.clip((s - warmup) / jnp.maximum(1.0, total - warmup), 0.0, 1.0)
    c = 0.1 * base + 0.45 * base * (1.0 + jnp.cos(jnp.pi * prog))
    return jnp.where(s < warmup, w, c)

def make_optim(cfg: TrainConfig, params: model.ModelParams) -> optax.GradientTransformation:
    # AdamW: clip -> adam -> weight decay (masked) -> lr schedule -> descent
    mask = jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)  # apply decay only to matrices/embeddings
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=cfg.eps),
        optax.add_decayed_weights(cfg.weight_decay, mask=mask),
        optax.scale_by_schedule(lambda s: lr_schedule(s, cfg.warmup_steps, cfg.total_steps, cfg.base_lr)),
        optax.scale(-1.0),
    )

# ---------- (3) loss and metrics ------------
def xent_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    # logits: (B,T,V), targets: (B,T) -> scalar loss
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1))

class Metrics(NamedTuple):
    loss: jnp.ndarray
    accuracy: jnp.ndarray
    perplexity: jnp.ndarray

def compute_metrics(logits: jnp.ndarray, targets: jnp.ndarray) -> Metrics:
    # logits: (B,T,V), targets: (B,T) -> {loss, acc, ppl}
    loss = xent_loss(logits, targets)
    preds = jnp.argmax(logits, axis=-1)  # (B,T)
    acc = jnp.mean((preds == targets).astype(jnp.float32))
    ppl = jnp.exp(loss)
    return Metrics(loss=loss, accuracy=acc, perplexity=ppl)

def loss_fn(params: model.ModelParams, xs: jnp.ndarray, ys: jnp.ndarray, forward: Callable, key: jax.Array) -> jnp.ndarray:
    logits = forward(params, xs, key, train=True)  # (B,T,V)
    return xent_loss(logits, ys)

# ---------- (4) single-device train step ------------
def compile_train_step(
    cfg: TrainConfig,
    forward: Callable[..., jnp.ndarray],
    master_params_example: model.ModelParams,
):
    # returns a jitted single-device train step closure: (state, batch) -> new_state.
    opt = make_optim(cfg, master_params_example)
    grad_fn = jax.grad(lambda p, x, y, k: loss_fn(p, x, y, forward, k))

    @partial(jax.jit, donate_argnums=(0, 1))
    def step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]):
        xs, ys = batch
        gs = cfg.grad_accum_steps
        B, msz = xs.shape[0], xs.shape[0] // gs
        assert (B % gs) == 0, "batch_size must be divisible by grad_accum_steps"
        keys = jax.random.split(state.rng_key, gs + 1)
        k_micro, k_next = keys[:-1], keys[-1]
        xs_m = xs.reshape(gs, msz, xs.shape[1])
        ys_m = ys.reshape(gs, msz, ys.shape[1])

        # vmap grad over microbatches, broadcast params
        grads_m = jax.vmap(grad_fn, in_axes=(None, 0, 0, 0))(state.params, xs_m, ys_m, k_micro)
        grads = jax.tree_util.tree_map(lambda g: jnp.mean(g, axis=0), grads_m)
        grads_f32 = jax.tree_util.tree_map(lambda g: g.astype(jnp.float32), grads)

        updates, new_opt_state = opt.update(grads_f32, state.opt_state, state.master_params)
        new_master_params = optax.apply_updates(state.master_params, updates)
        new_params = jax.tree_util.tree_map(lambda old, new: new.astype(old.dtype), state.params, new_master_params)

        new_state = TrainState(
            params=new_params,
            master_params=new_master_params,
            opt_state=new_opt_state,
            step=state.step + 1,
            rng_key=k_next,
        )
        return new_state

    return step

# ---------- (5) train loop ------------
def train(cfg: TrainConfig) -> None:
    key = jax.random.PRNGKey(cfg.seed)
    state, forward = init_train_state(cfg, key)
    step = compile_train_step(cfg, forward, state.master_params)
    loader = data_mod.build_dataloader(cfg.data, shard=False)

    import os, time, pickle
    t0 = time.time()

    # optional wandb
    wb = None
    if cfg.use_wandb:
        try:
            import wandb
            wb = wandb
            wb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name,
                config={
                    "total_steps": int(cfg.total_steps),
                    "batch_size": int(cfg.data.batch_size),
                    "seq_len": int(cfg.data.seq_len),
                    "base_lr": float(cfg.base_lr),
                    "warmup_steps": int(cfg.warmup_steps),
                    "weight_decay": float(cfg.weight_decay),
                    "grad_clip_norm": float(cfg.grad_clip_norm),
                    "grad_accum_steps": int(cfg.grad_accum_steps),
                    "n_layer": int(cfg.gpt.n_layer),
                    "n_head": int(cfg.gpt.n_head),
                    "d_model": int(cfg.gpt.d_model),
                    "ctx_len": int(cfg.gpt.ctx_len),
                    "vocab_size": int(cfg.gpt.vocab_size),
                },
            )
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    for _ in range(int(cfg.total_steps)):
        batch = next(loader)
        state = step(state, batch)
        s = int(state.step)

        if (s % cfg.log_every) == 0 or s == 1:
            # report lr used for the previous update (s-1), matching opt schedule step
            lr = float(lr_schedule(jnp.asarray(max(s - 1, 0), dtype=jnp.int32), cfg.warmup_steps, cfg.total_steps, cfg.base_lr))
            dt = time.time() - t0
            steps_done = 1 if s == 1 else int(cfg.log_every)
            toks = steps_done * int(cfg.data.batch_size) * int(cfg.data.seq_len)
            tps = toks / max(dt, 1e-9)
            xs, ys = batch
            logits = forward(state.params, xs, state.rng_key, train=False)
            m = compute_metrics(logits, ys)
            print(f"step {s:6d}  loss {float(m.loss):.4f}  acc {float(m.accuracy):.4f}  ppl {float(m.perplexity):.2f}  lr {lr:.6f}  {dt:.2f}s  {tps:.0f} tok/s")
            if wb is not None:
                wb.log({
                    "train/loss": float(m.loss),
                    "train/accuracy": float(m.accuracy),
                    "train/perplexity": float(m.perplexity),
                    "train/lr": lr,
                    "throughput/tokens_per_sec": tps,
                }, step=s)
            t0 = time.time()

        if cfg.save_checkpoints and (cfg.save_every > 0) and ((s % cfg.save_every) == 0):
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            path = os.path.join(cfg.ckpt_dir, f"step_{s}.pkl")
            with open(path, "wb") as f:
                payload = {
                    "master_params": state.master_params,
                    "opt_state": state.opt_state,
                    "step": s,
                }
                pickle.dump(jax.device_get(payload), f)

    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    train(TrainConfig())