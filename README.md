# gpt-2

> **JAX is all you need** ❤️

A mini, functional implementation of GPT-2 using nothing but JAX and `jax.numpy`.

- **`data.py`**
- **`model.py`**
- **`train.py`**

That's basically it.

#### Ideas and phil:
- Full transparency: every layer, every operation is explicit.
- JAX Functional style: immutable pytrees, pure functions, and explicit state threading. (will expand on this later)
- Education: did this project to better understand jax programming, implement gpt-2 from scratch, and code as clear, performant, compact, and readable as possible so others can also use it as an educational resource.
Traditional PyTorch/TensorFlow code is imperative and stateful. JAX forces a different paradigm:

#### WIP
- [ ] train with TPUs
- [ ] optimize and experiment
- [ ] having fun with it and seeing where this goes