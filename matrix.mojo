from sys.info import simdwidthof
from random import rand
from memory.memory import memset_zero
from algorithm import vectorize, parallelize
from runtime.llcl import Runtime

# alias f32 = DType.float32

alias nelts = simdwidthof[DType.float32]()

# TODO: try out the built-in Tensor type instead of a custom Matrix struct

# TODO: use generic FP datatype
struct Matrix:
    var rows: Int
    var cols: Int
    var data: DTypePointer[DType.float32]

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType.float32].alloc(rows*cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn __getitem__(self, i: Int) -> Float32:
        return self.data.simd_load[1](i)

    @always_inline
    fn __setitem__(inout self, y: Int, x: Int, val: Float32):
        self.store[1](y, x, val)

    @always_inline
    fn __setitem__(inout self, i: Int, val: Float32):
        self.data.simd_store[1](i, val)

    @always_inline
    fn zero(inout self):
        memset_zero(self.data, self.size()) 

    @always_inline
    fn fill(inout self, val: Float32):
        for i in range(self.size()):
            self.data.simd_store[1](i, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    @always_inline
    fn size(self) -> Int:
        return self.rows * self.cols

    fn dump(self):
        print_no_newline("[")
        for y in range(self.rows):
            print_no_newline("[")
            for x in range(self.cols):
                print_no_newline(self[y, x], "")

            if y == self.rows - 1:
                print_no_newline("]")
            else:
                print("],")
        print("]")
    
    fn __del__(owned self):
        self.data.free()


fn matmul(inout C: Matrix, A: Matrix, B: Matrix):
    # C = A @ B
    for m in range (A.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


fn matmul(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    matmul_parallelized(C, A, B, rt)


fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n))
            vectorize[nelts, dot](C.cols)
    parallelize[calc_row](rt, C.rows) 


fn matmul_transposed(inout C: Matrix, A: Matrix, B: Matrix):
    # TODO: for some reason this doesn't work if I parallelize it in the same
    # way as done for the "matmul" function. Try to figure out why.
    # C = A @ B.T
    for m in range (A.rows):
        for n in range(B.rows):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[n, k]


fn matmul_transposed(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    matmul_parallelized_transposed(C, A, B, rt)


fn matmul_parallelized_transposed(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    # C = A * B^T
    # TODO: this implementation is slow.
    @parameter
    fn calc_row(m: Int):
        for n in range(C.cols):
            @parameter
            fn dot[nelts: Int](k: Int):
                C.store[1](m, n, C.load[1](m, n) 
                           + (A.load[nelts](m, k) * B.load[nelts](n, k)).reduce_add())
            vectorize[nelts, dot](A.cols)
    parallelize[calc_row](rt, C.rows) 


fn transpose(out_m: Matrix, in_m: Matrix, rt: Runtime):
    # B = A^T
    @parameter
    fn row_fn(m: Int):
        @parameter 
        fn col_fn[nelts: Int](n: Int):
            out_m.store[nelts](n, m, in_m.load[nelts](m, n))
        vectorize[nelts, col_fn](in_m.cols)
    parallelize[row_fn](rt, in_m.rows)
