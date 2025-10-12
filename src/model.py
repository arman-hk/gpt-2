import dataclasses
from typing import Any, Tuple, NamedTuple, Optional, Callable
import jax
import jax.numpy as jnp

# ---------- (1) configs ------------
@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 50257
    ctx_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16

# ---------- (2) util funcs ------------
def causal_mask(seq_len: int, dtype: Any = jnp.float32) -> jnp.ndarray:
    if seq_len <= 0: raise ValueError("<seq_len> must be a positive int")
    return jnp.where(jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, ...], jnp.array(0.0, dtype), jnp.array(-jnp.inf, dtype))

def gelu(x: jnp.ndarray) -> jnp.ndarray:
    half = jnp.asarray(0.5, x.dtype)
    return half * x * jax.lax.erfc(-x * jnp.sqrt(half))

def dropout(x: jnp.ndarray, rate: float, key: jax.Array, *, train: bool) -> jnp.ndarray:
    if (rate == 0.0) or (not train): return x
    keep_prob = jnp.asarray(1.0 - rate, x.dtype)
    return (jax.random.bernoulli(key, keep_prob, x.shape)).astype(x.dtype) * x / keep_prob

def layernorm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    # layernorm over last dim: statistics and affine in f32, single cast at end
    x_f32 = x.astype(jnp.float32)
    centered = x_f32 - jnp.mean(x_f32, axis=-1, keepdims=True)
    inv_std = jax.lax.rsqrt(jnp.mean(jnp.square(centered), axis=-1, keepdims=True) + jnp.asarray(eps, jnp.float32))
    y_f32 = centered * inv_std
    return (y_f32 * gamma + beta).astype(x.dtype)

# ---------- (3) parameter pytrees ------------
class LayerNormParams(NamedTuple):
    gamma: jnp.ndarray
    beta: jnp.ndarray

class AttnParams(NamedTuple):
    W_qkv: jnp.ndarray
    b_qkv: jnp.ndarray
    W_o: jnp.ndarray
    b_o: jnp.ndarray

class MlpParams(NamedTuple):
    W1: jnp.ndarray
    b1: jnp.ndarray
    W2: jnp.ndarray
    b2: jnp.ndarray

class BlockParams(NamedTuple):
    ln1: LayerNormParams
    attn: AttnParams
    ln2: LayerNormParams
    mlp: MlpParams

class ModelParams(NamedTuple):
    tok_embed: jnp.ndarray
    pos_embed: jnp.ndarray
    blocks: Tuple[BlockParams, ...]
    ln_f: LayerNormParams
    head_b: jnp.ndarray

# ---------- (4) param init ------------
def init_block_params(key: jax.Array, cfg: GPTConfig) -> BlockParams:
    # a block: attn + mlp + two pre-LNs
    def n(k: jax.Array, s: Tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(k, s, cfg.dtype) / jnp.sqrt(jnp.array(s[0], jnp.float32))
    k1, k2, k3, k4 = jax.random.split(key, 4)
    C = cfg.d_model  # c for channel
    # mats: qkv (C,3C), o (C,C), mlp (C,4C),(4C,C); biases = 0
    W_qkv = n(k1, (C, 3 * C)); b_qkv = jnp.zeros((3 * C,), cfg.dtype)
    W_o   = n(k2, (C, C));     b_o   = jnp.zeros((C,), cfg.dtype)
    W1    = n(k3, (C, 4 * C)); b1    = jnp.zeros((4 * C,), cfg.dtype)
    W2    = n(k4, (4 * C, C)); b2    = jnp.zeros((C,), cfg.dtype)
    ln = lambda: LayerNormParams(gamma=jnp.ones((C,), jnp.float32), beta=jnp.zeros((C,), jnp.float32))  # pre-LN: gamma=1, beta=0
    return BlockParams(ln1=ln(), attn=AttnParams(W_qkv=W_qkv, b_qkv=b_qkv, W_o=W_o, b_o=b_o), ln2=ln(), mlp=MlpParams(W1=W1, b1=b1, W2=W2, b2=b2))

def init_gpt_params(cfg: GPTConfig, key: jax.Array) -> ModelParams:
    # model: token/pos embeddings, N blocks, final LN, tied head bias
    k_tok, k_pos, k_blocks = jax.random.split(key, 3)
    C, V, T = cfg.d_model, cfg.vocab_size, cfg.ctx_len
    return ModelParams(
        tok_embed = 0.02 * jax.random.normal(k_tok, (V, C), cfg.dtype),  # small N(0,1) init like GPT-2
        pos_embed = 0.02 * jax.random.normal(k_pos, (T, C), cfg.dtype),  # learned absolute positions
        blocks = tuple(init_block_params(k, cfg) for k in jax.random.split(k_blocks, cfg.n_layer)),  # per-layer params
        ln_f = LayerNormParams(gamma=jnp.ones((C,), jnp.float32), beta=jnp.zeros((C,), jnp.float32)),  # final LN in f32
        head_b = jnp.zeros((V,), cfg.dtype),  # output bias (tied weights via tok_embed in forward)
    )

# ---------- (5) model components ------------
def self_attention(x: jnp.ndarray, params: AttnParams, cfg: GPTConfig, key: jax.Array, mask: jnp.ndarray, *, train: bool) -> jnp.ndarray:
    B, T, C = x.shape; H = cfg.n_head; D = C // H  # shapes of batch, sequence len, channel (model width); heads; per head dim
    # project x -> [q|k|v] in one matmul, pack as (B,T,H,3D), then split along last dim
    qkv = (jnp.einsum("btc,cf->btf", x, params.W_qkv) + params.b_qkv).reshape(B, T, H, 3 * D)  # (B,T,C) x (C,3C) -> (B,T,3C); f = concatenated qkv (size 3C)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    # sdp attention (1/sqrt(D)) + causal mask; compute in f32 for stability
    scale = jax.lax.rsqrt(jnp.asarray(D, jnp.float32))
    attn_logits = jnp.einsum("bthd,bThd->bhtT", q.astype(jnp.float32), k.astype(jnp.float32)) * scale + mask
    k1, k2 = jax.random.split(key)
    probs = dropout(jax.nn.softmax(attn_logits, axis=-1).astype(x.dtype), cfg.dropout_rate, k1, train=train)
    # weighted sum of values -> merge heads -> output proj -> dropout
    y = jnp.einsum("bhtT,bThd->bthd", probs, v).reshape(B, T, C)  # (B,H,T,T) x (B,T,H,D) -> (B,T,H,D); sum over T
    y = jnp.einsum("btc,co->bto", y, params.W_o) + params.b_o  # (B,T,C) x (C,C) -> (B,T,C); o = output features (size C)
    return dropout(y, cfg.dropout_rate, k2, train=train).astype(x.dtype)

def mlp(x: jnp.ndarray, params: MlpParams, cfg: GPTConfig, key: jax.Array, train: bool) -> jnp.ndarray:
    k1, k2 = jax.random.split(key)
    # affine -> gelu -> dropout -> affine -> residual dropout
    h = dropout(gelu(jnp.einsum("btc,cm->btm", x, params.W1) + params.b1), cfg.dropout_rate, k1, train=train)  # (B,T,C) x (C,4C) -> (B,T,4C); c = model channel (C), m = mlp hidden (4C)
    y = jnp.einsum("btm,mc->btc", h, params.W2) + params.b2  # (B,T,4C) x (4C,C) -> (B,T,C); m = mlp hidden (4C), c = model channel (C)
    return dropout(y, cfg.dropout_rate, k2, train=train).astype(x.dtype)

def transformer(x: jnp.ndarray, params: BlockParams, cfg: GPTConfig, key: jax.Array, mask: jnp.ndarray, train: bool) -> jnp.ndarray:
    k1, k2 = jax.random.split(key)
    # pre-LN -> attention -> residual
    x = x + self_attention(layernorm(x, params.ln1.gamma, params.ln1.beta), params.attn, cfg, k1, mask, train=train)
    # pre-LN -> mlp -> residual
    return x + mlp(layernorm(x, params.ln2.gamma, params.ln2.beta), params.mlp, cfg, k2, train)

# ---------- (6) forward ------------
def gpt_forward(
    params: ModelParams,
    idx: jnp.ndarray,
    cfg: GPTConfig,
    key: jax.Array,
    train: bool = False,
    attn_bias: Optional[jnp.ndarray] = None,
    pos_ids: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    T = idx.shape[1]
    k_e, k_l = jax.random.split(key, 2)  # RNGs for embed / layers
    x = params.tok_embed[idx] + (params.pos_embed[:T] if pos_ids is None else params.pos_embed[pos_ids])  # (B,T,C)
    x = dropout(x, cfg.dropout_rate, k_e, train=train)  # embed dropout
    m = causal_mask(T, dtype=jnp.float32)
    if attn_bias is not None: m = m + attn_bias.astype(jnp.float32)  # additive bias (currently optional)
    # scan step: takes carry (x_c, k_c) and BlockParams p; splits RNG, applies one transformer block with mask m and train flag; returns updated carry (x_next, k_next) and None
    def step(c, p): x_c, k_c = c; k_c, k_s = jax.random.split(k_c); return (transformer(x_c, p, cfg, k_s, m, train), k_c), None  # one block
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *params.blocks)  # stack layer params
    (x, _), _ = jax.lax.scan(step, (x, k_l), stacked)  # iterate N layers
    x = layernorm(x, params.ln_f.gamma, params.ln_f.beta)  # final LN
    return jnp.einsum("btc,vc->btv", x, params.tok_embed) + params.head_b  # tied head

# ---------- (7) interfaces ------------
def compile_forward(cfg: GPTConfig):
    # returns a jitted forward function bound to cfg
    def forward(
        params: ModelParams,
        idx: jnp.ndarray,
        key: jax.Array,
        train: bool = False,
        attn_bias: Optional[jnp.ndarray] = None,
        pos_ids: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return gpt_forward(params, idx, cfg, key=key, train=train, attn_bias=attn_bias, pos_ids=pos_ids)
    return jax.jit(forward, static_argnames=("train",))

def build(cfg: GPTConfig, key: jax.Array) -> Tuple[ModelParams, Callable[..., jnp.ndarray]]:
    # convenience: (params, forward)
    params = init_gpt_params(cfg, key)
    return params, compile_forward(cfg)