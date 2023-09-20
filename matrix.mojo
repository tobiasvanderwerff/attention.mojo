from sys.info import simdwidthof
from random import rand
from memory.memory import memset_zero

# alias f32 = DType.float32

# TODO: use generic FP datatype
struct Matrix:
    var rows: Int
    var cols: Int
    var data: DTypePointer[DType.float32]

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[DType.float32].alloc(2*rows*cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.data.simd_load[1](y * self.cols + x)

    @always_inline
    fn __setitem__(inout self, y: Int, x: Int, val: Float32):
        self.data.simd_store[1](y * self.cols + x, val)

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
        print("[")
        for y in range(self.rows):
            print_no_newline("  [")
            for x in range(self.cols):
                print_no_newline(self[y, x], ",")
            print("],")
        print("]")
    
    fn __del__(owned self):
        self.data.free()


fn matmul(A: Matrix, B: Matrix, inout C: Matrix):
    # C = A @ B
    C.zero()
    for m in range (A.rows):
        for n in range(B.cols):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[k, n]

fn matmul_transposed(A: Matrix, B: Matrix, inout C: Matrix):
    # C = A @ B.T
    C.zero()
    for m in range (A.rows):
        for n in range(B.rows):
            for k in range(A.cols):
                C[m, n] += A[m, k] * B[n, k]


fn main() raises:
    var m = Matrix(10, 10)
    m.fill(1)
    # print_matrix(m)
    print("Element (1, 1) of matrix:", m[1, 1])

    var C = Matrix(m.rows, m.cols)
    matmul(m, m, C)
