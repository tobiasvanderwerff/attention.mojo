# attention.🔥

WIP

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
