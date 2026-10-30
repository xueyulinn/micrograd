import random
from .engine import Value


class Neuron:
    """
    A single neuron receives nint inputs and gives a single output
    """

    def __init__(self, nint):
        """
        Args:
          nint: number of inputs for this neuron
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nint)]
        self.b = Value(random.uniform(-1, 1))
        self.params = [w for w in self.w] + [self.b]

    def __call__(self, features):
        return sum((w * f for w, f in zip(self.w, features)), self.b)


class Layer:
    """
    A layer consists of nout neurons that receive nint inputs
    """

    def __init__(self, nint, nout):
        """
          Args:
            nint: number of inputs for one neuron
            nout: number of neurons for this layer
        """
        self.neurons = [Neuron(nint) for _ in range(nout)]
        self.params = [p for n in self.neurons for p in n.params]

    def __call__(self, features):
        outs = [neuron(features) for neuron in self.neurons]
        return outs if len(outs) > 1 else outs[0]


class MLP:
    """
    Multiple layer perceptron consists of nouts layers
    """

    def __init__(self, nint, nouts):
        """
        Args:
          nint: number of input features
          nout: number of layers
        """
        self.outs = [nint] + nouts
        self.layers = [Layer(self.outs[i], self.outs[i + 1])
                       for i in range(len(nouts))]
        self.params = [
            p for l in self.layers for n in l.neurons for p in n.params]

    def __call__(self, features):
        """
          Forward pass to get a final output
        """
        x = features
        for layer in self.layers:
            x = layer(x)
        return x
