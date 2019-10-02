class A:
    def run(self, x):
        print("A, run called")
        x = self.do_nothing(x)
        print(x)

    def do_nothing(self, x):
        print("B, do called")
        return x

class B(A):
    def run(self, x):
        print("B, run called")
        x += 1
        super().run(x)

    def do_nothing(self, x):
        print("B, do called")
        x += 1
        return super().do_nothing(x)

if __name__ == "__main__":
    b = B()
    b.run(1)