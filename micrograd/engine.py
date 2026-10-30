import math


class Value:
    def __init__(self, data, label="", _children=(), _op='', grad=0.0):
        self.data = data
        self.label = label
        self._backword = lambda: None
        self._children = _children
        self._op = _op
        self.grad = grad

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backword():
            self.grad += out.grad
            other.grad += out.grad

        out._backword = _backword
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, _children=(self, other), _op='-')

        def _backword():
            self.grad += out.grad
            other.grad += -out.grad

        out._backword = _backword
        return out

    def __rsub__(self, other):
        return Value(other) - self

    def __truediv__(self, other):
        return self * other**-1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backword():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backword = _backword
        return out

    def __rmul__(self, other):
        return self * other

    def exp(self):
        t = math.exp(self.data)
        out = Value(t, _children=(self, ), _op="exp")

        def _backword():
            self.grad += out.grad * t
        out._backword = _backword
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, _children=(self, other), _op="**")

        def _backword():
            self.grad += out.grad * \
                (other.data * self.data ** (other.data - 1))
            if self.data > 0:
                other.grad += out.grad * \
                    (self.data ** other.data * math.log(self.data))
        out._backword = _backword
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, "", (self, ), "tanh")

        def _backword():
            self.grad += (1 - t**2) * out.grad
        out._backword = _backword
        return out

    def backword(self):
        nodes = []
        visited = set()

        def topo(root: Value):
            if root in visited:
                return
            visited.add(root)
            for neighbor in root._children:
                topo(neighbor)

            nodes.append(root)

        topo(self)
        self.grad = 1.0
        for node in reversed(nodes):
            node._backword()
