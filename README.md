<h1 align="center">gpt-2</h1>

<p align="center"><b>JAX is all you need</b> 💜</p>

<p align="center">
  <em>A compact, functional implementation of gpt-2 using only <code>jax</code> and <code>jax.numpy</code></em>
</p>

---

### Structure:
- [`data.py`](./src/data.py)
- [`model.py`](./src/model.py)
- [`train.py`](./src/train.py)

That's basically it.

### Ideas and phil:
- JAX functional style: immutable pytrees, pure functions, and explicit state threading. (will expand on this later)
- Education: did this project to better understand `jax` programming, implement gpt-2 from scratch, and code as clear, performant, compact, and readable as possible so others can also use it as an educational resource.

### WIP:
- [ ] train with TPUs
- [ ] optimize and experiment
- [ ] have fun and see where this goes