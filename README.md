# attention.🔥

## TODO

- [ ] Continue looking at Mojo docs and examples -> upgrade Mojo skills

### Option 1. Improve existing Llama2.mojo implementation

At first sight, there seems to be plenty of optimizations that can still be done on the
code.

- [x] Clone my fork of the repo
- [ ] Study Llama2.mojo implementation
- [ ] Benchmark base performance on my laptop
- [ ] Ideally implement some tests to ensure correctness after changing the code
- [ ] Add optimizations:
  - [ ] Mojo autotune
  - [ ] fused operations
  - [ ] reduce # of function calls to e.g. parallelize/vectorize by merging functions (?)
    - [ ] softmax in the style of FlashAttention2

### Option 2: write optimized MHA implementation

It might be interesting/useful to start with this to get a "feel" for Mojo and just
start typing some code.

- [ ] write code for benchmarking (look at Mojo examples, e.g. matmul, for inspiration)
- [ ] benchmark:
  - [ ] Python
  - [ ] Mojo unoptimized
  - [ ] ...
