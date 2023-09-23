import math
from math import exp
from algorithm import vectorize, parallelize
from sys.info import simdwidthof
from runtime.llcl import Runtime

from matrix import Matrix, matmul, transpose, matmul_transposed


alias nelts = simdwidthof[DType.float32]()


@value
@register_passable("trivial")
struct AttentionParams:
    var seq_len: Int
    var num_heads: Int



fn attention(out: Matrix, Q: Matrix, K: Matrix, V: Matrix, rt: Runtime):
    """ Calculate 'out = softmax(Q * K^T) * V'.
    
    Args: 
      out: Matrix (N, d)
      Q: Matrix (N, d)
      K: Matrix (N, d)
      V: Matrix (N, d)
    """
    let N = Q.rows
    let d = Q.cols

    var QK = Matrix(N, N)
    QK.zero()
    matmul_transposed(QK, Q, K, rt)
    softmax(QK, QK, rt) 
    matmul(out, QK, V, rt)
    

fn softmax(inout out: Matrix, inp: Matrix, rt: Runtime):
    @parameter
    fn softmax_(y: Int):
        # TODO: CONTINUE HERE - see what optimizations are still possible

        var max_val: Float32 = -1e9  # TODO: how reliable is this?

        @parameter
        fn _max[nelts: Int](x: Int):
            let m = inp.load[nelts](y, x).reduce_max()
            if m > max_val:
                max_val = m
        vectorize[nelts, _max](inp.cols)

        var tmp = SIMD[DType.float32, nelts](0)

        @parameter
        fn _exp[nelts_: Int](x: Int):
            let v = exp(inp.load[nelts](y, x) - max_val)
            out.store[nelts](y, x, v)
            tmp += v
        vectorize[nelts, _exp](inp.cols)

        var sum: Float32 = tmp.reduce_add()

        @parameter
        fn _div[nelts_: Int](x: Int):
            out.store[nelts](y, x, out.load[nelts](y, x) / sum)
        vectorize[nelts, _div](inp.cols)
    
    parallelize[softmax_](rt, inp.rows)


fn softmax_naive(inout out: Matrix, inp: Matrix):
    for y in range(inp.rows):
        var sum: Float32 = 0.0
        var max_val: Float32 = -1e9  # TODO: how reliable is this?
        for xv in range(0, inp.cols, nelts):
            let max_val_v = inp.load[nelts](y, xv).reduce_max()
            if max_val_v > max_val:
                max_val = max_val_v
        for x in range(inp.cols):
            # out.store(y, x, exp(inp.load[nelts](y, x)))  # vectorized
            let e_x = exp(inp[y, x] - max_val)
            out[y, x] = e_x
            sum += e_x
        for x in range(inp.cols):
            out[y, x] /= sum
    

fn main():
    alias N = nelts // 2 
    alias d = nelts // 2

    print("Nelts:", nelts)

    # constrained[N % nelts == 0, "Only multiples of nelts supported for now"]()
    # constrained[d % nelts == 0, "Only multiples of nelts supported for now"]()

    var Q = Matrix(N, d)
    var K = Matrix(N, d)
    var V = Matrix(N, d)
    var res = Matrix(N, d)

    Q.fill(1)
    K.fill(1)
    # K[0, 0] = 2
    V.fill(3)
    res.zero()

    with Runtime() as rt:

        attention(res, Q, K, V, rt)
        res.dump()