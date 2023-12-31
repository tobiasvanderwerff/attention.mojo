import math
from python import Python
from benchmark import Benchmark
from time import time_function
from testing import assert_true
from runtime.llcl import Runtime

from attention import attention
from matrix import Matrix, transpose


# Test for equality between the Mojo and Python implementation of attention.


def np_to_matrix(inout m_mojo: Matrix, borrowed m_np: PythonObject, size: Int) -> None:
    data = m_np.flatten().tolist()
    for i in range(size):
        # TODO: is the cast to float64 necessary?
        m_mojo[i] = data[i].to_float64().cast[DType.float32]()


fn test(N: Int, d: Int) raises:
    let np = Python.import_module("numpy")
    if not np:
        print("Unable to load 'numpy' module")
        return

    # Generate random data
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

    Python.add_to_path(".")
    let py_att_mod: PythonObject = Python.import_module("attention")
    if not py_att_mod:
        print("Unable to load 'attention' module")
        return

    # Call Python impl
    let out_py_np = py_att_mod.attention(Q_np, K_np, V_np)

    # Convert Python result to Mojo
    var out_py = Matrix(N, d)
    np_to_matrix(out_py, out_py_np, N*d)

    # Call a separate function in order to instantiate a runtime, which is not
    # possible within a "raises" function.
    _compare(out_py, out, Q, K, V)


fn _compare(out_py: Matrix, inout out: Matrix, Q: Matrix, K: Matrix, V: Matrix):
    # Call Mojo impl
    with Runtime() as rt:
        attention(out, Q, K, V, rt)

        # Compare the two results
        for y in range(out.rows):
            for x in range(out.cols):
                let a = out[y, x]
                let b = out_py[y, x]
                if not assert_true(math.isclose(a, b, 0, 5e-5), "Fail"):
                    print("TEST FAILED: a != b at (", y, ",", x, ") (a =", a, ", b =", b, ")")
                    return 

        print("TEST PASSED")


fn main() raises:
    var N = 256
    var d = 128
    test(N, d)

    N = 256
    d = 100
    test(N, d)

