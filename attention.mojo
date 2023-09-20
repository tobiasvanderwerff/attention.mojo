from matrix import Matrix, matmul, matmul_transposed
import math
from math import exp
from algorithm import vectorize


@value
struct AttentionParams:
    var seq_len: Int
    var num_heads: Int



fn attention_naive(inout out: Matrix, Q: Matrix, K: Matrix, V: Matrix):
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
    softmax_naive(tmp, tmp) 
    matmul(tmp, V, out)
     

fn softmax[nelts: Int](inout out: Matrix, inp: Matrix):

    var max_val: Float32 = -1e9

    @parameter
    fn _row_max[nelts_: Int](i: Int):
        let m = inp.data.simd_load[nelts_](i).reduce_max()
        if m > max_val:
            max_val = m

    vectorize[nelts, _row_max](inp.cols)
    # TODO
    

fn softmax_naive(inout out: Matrix, inp: Matrix):
    for y in range(inp.rows):
        var sum: Float32 = 0.0
        var max_val: Float32 = math.limit.neginf[DType.float32]()
        for x in range(inp.cols):
            if inp[y, x] > max_val:
                max_val = inp[y, x]
        for x in range(inp.cols):
            # out.store(y, x, exp(inp.load[nelts](y, x)))  # vectorized
            let e_x = exp(inp[y, x] - max_val)
            out[y, x] = e_x
            sum += e_x
        for x in range(inp.cols):
            out[y, x] /= sum

    
fn main():
    let N = 4
    let d = 2

    var Q = Matrix(N, d)
    var K = Matrix(N, d)
    var V = Matrix(N, d)
    var res = Matrix(N, d)

    Q.fill(2)
    K.fill(1)
    K[0, 0] = 2
    V.fill(3)
    res.zero()

    attention_naive(res, Q, K, V)

    res.dump()