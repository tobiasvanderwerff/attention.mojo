from python import Python
from benchmark import Benchmark
from time import time_function

@value
struct AttentionParams:
    var seq_len: Int
    var num_heads: Int


# fn benchmark_attention[AttParams: AttentionParams]():
fn benchmark_attention[seq_len: Int, num_heads: Int]():
    alias N = seq_len
    let x = N * num_heads  

    # Benchmark the function
    # let cur_time = BenchMark(1).run[]()


fn main() raises:
    let np = Python.import_module("numpy")
    Python.add_to_path(".")
    let py_att_mod: PythonObject = Python.import_module("attention")


    # var Q = np.arange(10).reshape(2, 5)
    let Q = np.ones((2, 3))
    let K = Q
    let V = Q

    if py_att_mod:
        let weights = py_att_mod.attention(Q, K, V)
        print(weights)
    else:
        print("Unable to load 'attention' module")

    # let att_params = AttentionParams(16, 2)
    _ = benchmark_attention[16, 2]()

    # let t = time_function[py_att_mod.attention(Q, K, V)]()
    # print("Time spent in function:", t, "ns")
