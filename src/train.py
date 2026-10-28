import dataclasses
from contextlib import closing
from numbers import Integral
from typing import Any, Tuple, NamedTuple, Optional, Callable
import jax, optax
import jax.numpy as jnp
from . import model
from .data import DataConfig, build_dataloader

# ---------- (1) configs and train state ------------
@dataclasses.dataclass
class TrainConfig:
    """steps count optimizer updates; data.batch_size includes all microbatches."""
    total_steps: int = 200; save_every: int = 100; log_every: int = 10; seed: int = 0
    # AdamW
    base_lr: float = 3e-4; weight_decay: float = 0.1; beta1: float = 0.9; beta2: float = 0.95; eps: float = 1e-8
    # schedule / stability
    warmup_steps: int = 50; grad_clip_norm: float = 1.0; grad_accum_steps: int = 1
    # runtime / system
    ckpt_dir: str = "ckpts"; save_checkpoints: bool = False; use_wandb: bool = True
    # logging
    wandb_project: str = "gpt-speedrun"; wandb_run_name: Optional[str] = None
    # other configs
    gpt: model.GPTConfig = dataclasses.field(default_factory=lambda: model.GPTConfig(vocab_size=50257, ctx_len=128, n_layer=2, n_head=2, d_model=64, dropout_rate=0.0, dtype=jnp.bfloat16))
    data: DataConfig = dataclasses.field(default_factory=lambda: DataConfig(batch_size=8, seq_len=128))

def _microbatch_size(cfg: TrainConfig) -> int:
    for name, value in (("grad_accum_steps", cfg.grad_accum_steps), ("data.batch_size", cfg.data.batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg.data.batch_size % cfg.grad_accum_steps:
        raise ValueError("data.batch_size must be divisible by grad_accum_steps")
    return int(cfg.data.batch_size // cfg.grad_accum_steps)

class TrainState(NamedTuple):
    params: model.ModelParams  # bf16 params used for forward/backprop
    master_params: model.ModelParams  # f32 master copy used by optimizer
    opt_state: Any
    rng_key: jax.Array

def init_train_state(cfg: TrainConfig, key: jax.Array) -> Tuple[TrainState, Callable[..., jnp.ndarray], Any, Callable[[int], float]]:
    _microbatch_size(cfg)
    schedule = make_lr_schedule(cfg)
    k_init, k_state = jax.random.split(key, 2)
    params, forward = model.build(cfg.gpt, key=k_init)
    master_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
    opt = make_optim(cfg, master_params, schedule)
    return TrainState(params, master_params, opt.init(master_params), k_state), forward, opt, schedule

# ---------- (2) optim ------------
def make_lr_schedule(cfg: TrainConfig):
    total_steps, warmup_steps = int(cfg.total_steps), int(cfg.warmup_steps)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.base_lr),
        warmup_steps=warmup_steps,
        decay_steps=total_steps,  # optax includes warmup in the schedule duration
        end_value=float(0.1 * cfg.base_lr),
    )

def make_optim(cfg: TrainConfig, params: model.ModelParams, schedule: Callable[[int], float]) -> Any:
    # AdamW with schedule; clip -> AdamW (masked); wrapped with MultiSteps
    mask = jax.tree_util.tree_map(lambda x: x.ndim >= 2, params)
    inner = optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=cfg.beta1,
            b2=cfg.beta2,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            mask=mask,
        ),
    )
    return optax.MultiSteps(inner, every_k_schedule=int(cfg.grad_accum_steps))

# ---------- (3) loss and metrics ------------
def xent_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1), targets[..., None], axis=-1).squeeze(-1))

class Metrics(NamedTuple): loss: jnp.ndarray; perplexity: jnp.ndarray

def compute_metrics(logits: jnp.ndarray, targets: jnp.ndarray) -> Metrics:
    loss = xent_loss(logits, targets)
    return Metrics(loss=loss, perplexity=jnp.exp(loss))

def loss_fn(params: model.ModelParams, xs: jnp.ndarray, ys: jnp.ndarray, forward: Callable, key: jax.Array) -> jnp.ndarray:
    return xent_loss(forward(params, xs, key, train=True), ys)

# ---------- (4) single-device train step ------------
def compile_train_step(cfg: TrainConfig, forward: Callable[..., jnp.ndarray], opt: Any):
    """compile one microbatch step; MultiSteps applies an update every K calls."""
    grad_fn = jax.grad(lambda p, x, y, k: loss_fn(p, x, y, forward, k))
    @jax.jit
    def step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray]):
        xs, ys = batch
        k_use, k_next = jax.random.split(state.rng_key, 2)
        grads = grad_fn(state.params, xs, ys, k_use)
        grads_f32 = jax.tree_util.tree_map(lambda t: t.astype(jnp.float32), grads)
        updates, new_opt_state = opt.update(grads_f32, state.opt_state, state.master_params)
        new_master_params = optax.apply_updates(state.master_params, updates)
        new_params = jax.tree_util.tree_map(lambda old, new: new.astype(old.dtype), state.params, new_master_params)
        return TrainState(params=new_params, master_params=new_master_params, opt_state=new_opt_state, rng_key=k_next)
    return step

# ---------- (5) train loop ------------
def train(cfg: TrainConfig) -> None:
    msz = _microbatch_size(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    state, forward, opt, schedule = init_train_state(cfg, key)
    step = compile_train_step(cfg, forward, opt)
    loader = build_dataloader(cfg.data, shard=False)
    
    import os, time, pickle
    t0 = time.time()
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

    with closing(loader):
        for s in range(1, int(cfg.total_steps) + 1):
            batch = next(loader)
            xs, ys = batch
            if xs.ndim != 2 or xs.shape != ys.shape or xs.shape[0] != cfg.data.batch_size:
                raise ValueError(
                    f"inputs and targets must have matching rank-2 shapes and batch size {cfg.data.batch_size}; "
                    f"got {xs.shape} and {ys.shape}"
                )
            # average K equal microbatches and apply exactly one optimizer update.
            for i in range(int(cfg.grad_accum_steps)):
                state = step(state, (xs[i*msz:(i+1)*msz], ys[i*msz:(i+1)*msz]))
            if (s % cfg.log_every) == 0 or s == 1:
                # derive LR used for the just-applied update: schedule(gradient_step - 1)
                gstep = int(jax.device_get(state.opt_state.gradient_step))
                lr = float(schedule(max(gstep - 1, 0)))
                dt = time.time() - t0
                steps_done = 1 if s == 1 else int(cfg.log_every)
                toks = steps_done * int(cfg.data.batch_size) * int(cfg.data.seq_len)
                tps = toks / max(dt, 1e-9)
                xs, ys = batch
                logits = forward(state.params, xs, state.rng_key, train=False)
                m = compute_metrics(logits, ys)
                print(f"step {s:6d}  loss {float(m.loss):.4f}  ppl {float(m.perplexity):.2f}  lr {lr:.6f}  {dt:.2f}s  {tps:.0f} tok/s")
                if wb is not None:
                    wb.log({
                        "train/loss": float(m.loss),
                        "train/perplexity": float(m.perplexity),
                        "train/lr": lr,
                        "throughput/tokens_per_sec": tps,
                    }, step=s)
                t0 = time.time()

            if cfg.save_checkpoints and (cfg.save_every > 0) and ((s % cfg.save_every) == 0):
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                path = os.path.join(cfg.ckpt_dir, f"step_{s}.pkl")
                with open(path, "wb") as f:
                    payload = {"master_params": state.master_params, "opt_state": state.opt_state, "step": s}
                    pickle.dump(jax.device_get(payload), f)
    if wb is not None:
        try: wb.finish()
        except Exception: pass

if __name__ == "__main__":
    train(TrainConfig())
