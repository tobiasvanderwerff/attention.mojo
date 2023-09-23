# attention.🔥

A straightforward implementation of attention in Mojo. We compute the following:

```
O = softmax(Q * K^T) * V
```

where `Q`, `K`, `V`, `O`, are matrices of shape `N x d`.

## Benchmarking

```
OS:         Ubuntu 22.04
CPU(s):     24
Model name: Intel(R) Core(TM) i7-13700HX CPU @ 3.70GHz
```


Time in microseconds (us):


|                    | Numpy/Python  | Mojo          | Speedup vs. Numpy |
| ------------------ | ------------- | ------------- | ----------------- |
| N=64,d=64          | 520           | 3791          | 0.13x             |
| N=512,d=512        | 12629         | 7100          | 1.8x              |
| N=1024,d=256       | 31835         | 11192         | 2.8x              |


## Why is the Mojo vs. Numpy comparison interesting? 
Although Python is known to be slow for compute-intensive operations, Numpy is
written in high performance C code and could in theory be used to implement a
relatively fast implementation of multi-head attention. Obviously you would
ideally use a GPU to run a highly parallel operation like attention, but within
the CPU domain there are also options for parallization and vectorization, e.g.
by using vector operations. I'm curious how Mojo's language features like
easy vectorization, pararallization, autotuning, etc., fare on CPU against a
highly optimized library like Numpy.

One limitation of this comparison is that it's unclear how much overhead Python itself is adding, e.g. when making consecutive Numpy calls.


## TODO
- [ ] See if `autotune` and `search` can be used to make useful hardware-specific optimizations at compile time. I.e. write multiple implementations (e.g. tiling or not) and pick the best one for the hardware it's running on by runnning benchmarks for all of them at compile time.
- [ ] Investigate: what lessons can be drawn from [Flash Attention](https://arxiv.org/pdf/2205.14135.pdf) for making CPU implementations of attention more io-aware?