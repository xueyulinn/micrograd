# micrograd

## Overview
A project implementing an automatic differentiation engine with an interface similar to PyTorch's. 

### How it works
Each number is wrapped in a `Value` object that defines the derivatives of common mathematical operations, and all `Value`s form a DAG. During backpropagation, it sets the last value's gradient to 1, then works backwards applying the chain rule.

## Example usage
```python
from micrograd.engine import Value

x = Value(2.0)
y = x**2 + 3 * x
y.backward()
print(x.grad)  # 7.0
```
More examples: [demo.ipynb](demo.ipynb)

## Tests
```bash
python -m unittest
```

## Reference
- [Neural Networks: Zero to Hero by Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Original project](https://github.com/karpathy/micrograd)