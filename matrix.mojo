alias f32 = Float32

struct Matrix[T: AnyType]:
    var rows: Int
    var cols: Int
    var data: Pointer[T]

    fn __init__(inout self, rows: Int, cols: Int, value: T):
        self.rows = rows
        self.cols = cols
        self.data = Pointer[T].alloc(2*rows*cols)
        for i in range(rows*cols):
            self.data.store(i, value)

    @always_inline
    fn __getitem__(self, i: Int) -> T:
        return self.data.load(i)
    
    fn __del__(owned self):
        self.data.free()


def print_matrix[T: AnyType](borrowed m: Matrix[T]) -> None:  # "borrowed" enables immutable referencing
    print("Matrix with", m.rows, "rows and", m.cols, "columns.")


fn main() raises:
    let m = Matrix[f32](2, 3, 3.14)
    print_matrix(m)
    print("Element 0 of matrix:", m[0])