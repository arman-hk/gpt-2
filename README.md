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

### WIP:
- [ ] train with TPUs
- [ ] optimize and experiment
- [ ] have fun and see where this goes