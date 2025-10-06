import pytest
import jax
import jax.numpy as jnp
import model

@pytest.fixture
def cfg_small():
    return model.GPTConfig(vocab_size=128, ctx_len=16, n_layer=2, n_head=2, d_model=32, dropout_rate=0.0)

def test_causal_mask_values():
    mask = model.causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    tri = jnp.tril(jnp.ones((4, 4), dtype=bool))
    assert (mask[0, 0][tri] == 0.0).all()
    assert jnp.isneginf(mask[0, 0][~tri]).all()

def test_layernorm_normalizes_last_dim():
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    gamma = jnp.ones((4,), x.dtype)
    beta = jnp.zeros((4,), x.dtype)
    y = model.layernorm(x, gamma, beta)
    m = jnp.mean(y, axis=-1)
    v = jnp.var(y, axis=-1)
    assert jnp.allclose(m, 0.0, atol=1e-5)
    assert jnp.allclose(v, 1.0, atol=1e-4)

@pytest.mark.parametrize("train,rate", [(False, 0.5), (True, 0.0)])
def test_dropout_disabled_is_identity(train, rate):
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    y = model.dropout(x, rate, jax.random.PRNGKey(0), train=train)
    assert jnp.allclose(x, y)

def test_param_init_shapes(cfg_small):
    cfg = cfg_small
    params = model.init_gpt_params(cfg, jax.random.PRNGKey(0))
    C, V, T = cfg.d_model, cfg.vocab_size, cfg.ctx_len
    assert params.tok_embed.shape == (V, C)
    assert params.pos_embed.shape == (T, C)
    assert len(params.blocks) == cfg.n_layer
    assert params.ln_f.gamma.shape == (C,)
    assert params.ln_f.beta.shape == (C,)
    assert params.head_b.shape == (V,)
    for b in params.blocks:
        assert b.attn.W_qkv.shape == (C, 3 * C)
        assert b.attn.b_qkv.shape == (3 * C,)
        assert b.attn.W_o.shape == (C, C)
        assert b.attn.b_o.shape == (C,)
        assert b.mlp.W1.shape == (C, 4 * C)
        assert b.mlp.b1.shape == (4 * C,)
        assert b.mlp.W2.shape == (4 * C, C)
        assert b.mlp.b2.shape == (C,)

def test_forward_and_compiled_match_and_finite(cfg_small):
    cfg = cfg_small
    params, fwd = model.build(cfg, key=jax.random.PRNGKey(0))
    B, T = 2, 8
    idx = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    logits_ref = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False)
    logits_jit = fwd(params, idx, key=jax.random.PRNGKey(2), train=False)
    assert logits_ref.shape == (B, T, cfg.vocab_size)
    assert logits_jit.shape == (B, T, cfg.vocab_size)
    assert jnp.allclose(logits_ref, logits_jit)
    assert jnp.isfinite(logits_ref).all()

def test_train_mode_same_key_is_reproducible():
    cfg = model.GPTConfig(vocab_size=64, ctx_len=8, n_layer=2, n_head=2, d_model=16, dropout_rate=0.3)
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    idx = jnp.arange(cfg.ctx_len).reshape(1, -1) % cfg.vocab_size
    key = jax.random.PRNGKey(3)
    out1 = model.gpt_forward(params, idx, cfg, key=key, train=True)
    out2 = model.gpt_forward(params, idx, cfg, key=key, train=True)
    assert jnp.allclose(out1, out2)

def test_pos_ids_changes_outputs(cfg_small):
    cfg = cfg_small
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    B, T = 1, 8
    idx = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    pos_ids = jnp.arange(T - 1, -1, -1, dtype=jnp.int32)
    a = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False)
    b = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False, pos_ids=pos_ids)
    assert not jnp.allclose(a, b)

def test_attn_bias_changes_outputs(cfg_small):
    cfg = cfg_small
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    B, T = 1, 8
    idx = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    bias = jnp.zeros((1, 1, T, T), dtype=jnp.float32).at[:, :, :, 0].set(-1e9)
    a = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False)
    b = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False, attn_bias=bias)
    assert not jnp.allclose(a, b)

def test_train_false_ignores_rng():
    cfg = model.GPTConfig(vocab_size=64, ctx_len=8, n_layer=2, n_head=2, d_model=16, dropout_rate=0.3)
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    idx = jnp.arange(cfg.ctx_len).reshape(1, -1) % cfg.vocab_size
    a = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(1), train=False)
    b = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False)
    assert jnp.allclose(a, b)

def test_dropout_train_mode_different_keys():
    cfg = model.GPTConfig(vocab_size=64, ctx_len=8, n_layer=2, n_head=2, d_model=16, dropout_rate=0.5)
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    idx = jnp.arange(cfg.ctx_len).reshape(1, -1) % cfg.vocab_size
    a = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(1), train=True)
    b = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=True)
    assert not jnp.allclose(a, b)

def test_no_future_information_leak(cfg_small):
    cfg = cfg_small
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    B, T = 1, 8
    idx = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)
    a = model.gpt_forward(params, idx, cfg, key=jax.random.PRNGKey(2), train=False)
    idx_mod = idx.at[:, -1].set((idx[:, -1] + 1) % cfg.vocab_size)
    b = model.gpt_forward(params, idx_mod, cfg, key=jax.random.PRNGKey(2), train=False)
    assert jnp.allclose(a[:, :-1, :], b[:, :-1, :])

def test_gradients_flow_finite_nonzero(cfg_small):
    cfg = cfg_small
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    B, T = 2, 8
    idx = jax.random.randint(jax.random.PRNGKey(1), (B, T), 0, cfg.vocab_size)

    def loss_fn(p, k):
        logits = model.gpt_forward(p, idx, cfg, key=k, train=False)
        logp = jax.nn.log_softmax(logits, axis=-1)
        target = idx[:, 1:]
        nll = -jnp.take_along_axis(logp[:, :-1, :], target[..., None], axis=-1).squeeze(-1)
        return jnp.mean(nll)

    grads = jax.grad(loss_fn)(params, jax.random.PRNGKey(3))
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.isfinite(g).all() for g in leaves)
    total_norm = sum(float(jnp.linalg.norm(g)) for g in leaves)
    assert total_norm > 0.0

def test_params_tree_roundtrip(cfg_small):
    cfg = cfg_small
    params, _ = model.build(cfg, key=jax.random.PRNGKey(0))
    treedef = jax.tree_util.tree_structure(params)
    leaves = jax.tree_util.tree_leaves(params)
    recon = jax.tree_util.tree_unflatten(treedef, leaves)
    same = jax.tree_util.tree_map(lambda a, b: jnp.array_equal(a, b), params, recon)
    assert all(bool(x) for x in jax.tree_util.tree_leaves(same))
