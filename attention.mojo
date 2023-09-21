import math
from math import exp
from algorithm import vectorize, parallelize
from sys.info import simdwidthof
from runtime.llcl import Runtime

from matrix import Matrix, matmul, matmul_transposed


alias nelts = simdwidthof[DType.float32]()


@value
struct AttentionParams:
    var seq_len: Int
    var num_heads: Int



# fn attention_naive(inout out: Matrix, Q: Matrix, K: Matrix, V: Matrix):
#     # Same as function below but without runtime 
#     # TODO: come up with a better way to do this
#     let N = Q.rows
#     let d = Q.cols
#     var tmp = Matrix(N, N)
#     tmp.zero()
#     matmul_transposed(Q, K, tmp)
#     # softmax[1](tmp, tmp) 
#     softmax_naive(tmp, tmp) 
#     matmul(tmp, V, out)


fn attention_naive(inout out: Matrix, Q: Matrix, K: Matrix, V: Matrix, rt: Runtime):
    # Calculate "out = softmax(Q * K^T) * V"
    # 
    # Args: 
    #   out: Matrix (N, d)
    #   Q: Matrix (N, d)
    #   K: Matrix (N, d)
    #   V: Matrix (N, d)
    let N = Q.rows
    let d = Q.cols
    var tmp = Matrix(N, N)
    tmp.zero()
    matmul_transposed(Q, K, tmp)
    # softmax[1](tmp, tmp) 
    softmax_naive(tmp, tmp, rt) 
    matmul(tmp, V, out)
     

fn softmax(inout out: Matrix, inp: Matrix, rt: Runtime):

    var max_val: Float32 = -1e9

    @parameter
    fn _row_max[nelts_: Int](i: Int):
        let m = inp.data.simd_load[nelts_](i).reduce_max()
        if m > max_val:
            max_val = m

    vectorize[nelts, _row_max](inp.cols)
    # TODO
    
fn softmax_naive(inout out: Matrix, inp: Matrix, rt: Runtime):
    @parameter
    fn softmax_(y: Int):
        # TODO: CONTINUE HERE - vectorize/parallelize these inner loops
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


# @parameter
# fn parallelize_across_rows(row_fn: fn(Int) -> None, rows: Int, rt: Runtime):
#     parallelize[row_fn](rt, rows)


fn my_exp(m: Matrix, rt: Runtime):
    @parameter
    fn calc_row(y: Int):
        @parameter
        fn v_exp[nelts: Int](x: Int):
            m.store[nelts](y, x, exp(m.load[nelts](y, x)))

        vectorize[nelts, v_exp](m.cols)
    parallelize[calc_row](rt, m.rows)
    

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
    # K.fill(1)
    # K[0, 0] = 2
    # V.fill(3)
    # res.zero()

    with Runtime() as rt:

        # attention_naive(res, Q, K, V)
        # res.dump()

        my_exp(Q, rt)
        Q.dump()