from python import Python
from benchmark import Benchmark
from time import time_function
from testing import assert_true

from attention import attention_naive
from matrix import Matrix


def np_to_matrix(inout m_mojo: Matrix, borrowed m_np: PythonObject, size: Int) -> None:
    data = m_np.flatten().tolist()
    for i in range(size):
        # TODO: is the cast to float64 necessary?
        m_mojo[i] = data[i].to_float64().cast[DType.float32]()


fn main() raises:
    let np = Python.import_module("numpy")
    Python.add_to_path(".")
    let py_att_mod: PythonObject = Python.import_module("attention")

    # let N = 512
    # let d = 256
    let N = 3
    let d = 2

    let Q_np = np.random.rand(N, d)
    let K_np = np.random.rand(N, d)
    let V_np = np.random.rand(N, d)

    # Create Mojo matrices containing same data as NP arrays 
    var Q = Matrix(N, d)
    var K = Matrix(N, d)
    var V = Matrix(N, d)
    var out = Matrix(N, d)
    np_to_matrix(Q, Q_np, N*d)
    np_to_matrix(K, K_np, N*d)
    np_to_matrix(V, V_np, N*d)
    out.zero()

    # print(V_np)
    # V.dump()

    # Benchmark Python/Numpy
    var usecs_python: Float64 = 0.0
    if py_att_mod:
        # let usecs_python = py_att_mod.benchmark_attention_python(N, d)
        usecs_python = py_att_mod.benchmark_attention_python(Q_np, K_np, V_np).to_float64()
    else:
        print("Unable to load 'attention' module")

    # let t = time_function[py_att_mod.attention(Q, K, V)]()
    # print("Time spent in function:", t, "ns")

    @always_inline
    @parameter
    fn test_fn():
        attention_naive(out, Q, K, V)

    attention_naive(out, Q, K, V)
    print("Mojo res:")
    out.dump()

    # Benchmark Mojo
    let usecs_mojo = Float64(Benchmark().run[test_fn]()) / 1_000

    print("\nNumpy/Python:\t", usecs_python, "us")
    print("Mojo:\t\t", usecs_mojo, "us")

    # The line below is necessary to prevent the matrices from being freed
    # before the benchmark run, which would otherwise lead to the program crashing.
    _ = (out, Q, K, V)