import unittest
from micrograd.engine import Value
import torch


class TestValue(unittest.TestCase):
    def test_sanity_check(self):
        x = Value(2.0)
        y = Value(3.0)
        l = x**2 + y**2 + 10
        l.backword()
        xmg, ymg = x.grad, y.grad

        x = torch.tensor(2.0).requires_grad_()
        y = torch.tensor(3.0).requires_grad_()
        l = x**2 + y**2 + 10
        l.backward()

        self.assertEqual(xmg, x.grad.item())
        self.assertEqual(ymg, y.grad.item())
