from matrix import Matrix, matmul, matmul_transposed
from math import exp


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
    softmax_naive[1](tmp, tmp) 
    matmul(tmp, V, out)
     
fn softmax_naive[nelts: Int](inout out: Matrix, inp: Matrix):
    for y in range(inp.rows):
        var sum: Float32 = 0.0
        for x in range(inp.cols):
            # out.store(y, x, exp(inp.load[nelts](y, x)))  # vectorized
            let xp = exp(inp[y, x])
            out[y, x] = xp
            sum += xp
        for x in range(inp.cols):
            out[y, x] /= sum
    
fn main():
    let N = 4
    let d = 2

    var Q = Matrix(N, d)
    var K = Matrix(N, d)
    var V = Matrix(N, d)
    var res = Matrix(N, d)

    Q.fill(1)
    K.fill(1)
    V.fill(1)
    res.zero()

    attention_naive(res, Q, K, V)

    res.dump()