<h1 align="center">gpt-2</h1>

> **JAX is all you need 💜**

This is a compact, functional implementation of gpt-2 using only `jax` and `jax.numpy`.

### Structure:
- [`data.py`](./src/data.py)
- [`model.py`](./src/model.py)
- [`train.py`](./src/train.py)

That's basically it.

### Ideas and phil:
- JAX functional style: immutable pytrees, pure functions, and explicit state threading. (will expand on this later)
- Education: did this project to better understand `jax` programming, implement gpt-2 from scratch, and code as clear and compact as possible so others can also see this as an educational resource.

### Training steps:

- `data.batch_size` is the total number of sequences per optimizer update.
- `grad_accum_steps` splits that batch into equal microbatches. For example, batch size 8 and accumulation 4 means four microbatches of two sequences, followed by one optimizer update.
- Both values must be positive integers, and batch size must be divisible by the accumulation count.
- `total_steps`, warmup/decay, logging, and checkpoint intervals count optimizer updates.

### WIP:
- [ ] train with TPU
- [ ] optimize and experiment
- [ ] have fun and see where this goes
