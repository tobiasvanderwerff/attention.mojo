struct Pair:
    var v1: Int
    var v2: Int

    fn __init__(inout self, v1: Int, v2: Int):
        self.v1 = v1
        self.v2 = v2

    fn __init__(inout self, v: Int):
        self.v1 = v
        self.v2 = v

    fn __iadd__(inout self, x: Int):
        self.v1 = self.v1 + x
        self.v2 = self.v2 + x
    
    fn dump(self):
        print_no_newline("(")
        print_no_newline(self.v1, self.v2)
        print(")")

def no_types(a, b):
    c = b
    if c != b:
        print("no Ts")

fn main() raises:
    var p = Pair(1, 4)
    p += 2
    p.dump()
    no_types(p.v1, p.v2)

