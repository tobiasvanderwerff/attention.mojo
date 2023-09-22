import numpy as np
from timeit import timeit

def attention(Q, K, V):
    """ 
    Args:
        Q: query matrix, shape (seq_len, d_model)
        K: key matrix, shape (seq_len, d_model)
        V: value matrix, shape (seq_len, d_model)
    """
    QK = Q @ K.T
    max_QK = np.amax(QK, axis=-1, keepdims=True)
    S = np.exp(QK - max_QK)
    S /= S.sum(axis=-1, keepdims=True)
    O = S @ V
    return O

def benchmark_attention_python(Q, K, V, print_res=False):
    N = 20
    # usecs = timeit(lambda: attention(Q, K, V), number=N) / N * 1000 * 1000  # warmup
    usecs = timeit(lambda: attention(Q, K, V), number=N) / N * 1000 * 1000
    if print_res:
        print(f"Python res:\n{attention(Q, K, V)}")
    return usecs

# def benchmark_attention_python(N, d):
#     Q = np.random.rand(N, d)
#     K = np.random.rand(N, d)
#     V = np.random.rand(N, d)
#     usecs = timeit(lambda: attention(Q, K, V), number=2) / 2 * 1000000
#     # gflops = ((2*M*N*K)/secs) / 1e9
#     # print(gflops, "GFLOP/s")
#     return usecs


if __name__ == "__main__":
    N = 32
    d = 8

    usecs_python = benchmark_attention_python(N, d)
    print(usecs_python, "us")