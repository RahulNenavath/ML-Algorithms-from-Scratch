# requirements:
# - Goal: a computation graph of neurons: forward propagate (done), backward propage, find gradients
# - each neuron is a computation model - find the dot product of inputs & weights + bias (done)
# - each neuron will get a input data point (X: [x1, x2, x3]). this will be a custon data type (done)
# - implement derivative calculation using `first principles of derivative` for all the operations

import math
import numpy as np
from typing import Any

class Datapoint:
    def __init__(self, value: int, _prev: tuple = (), _op: str = '', label: str = '') -> None:
        self.value = value
        self._prev = _prev
        self.label = label
        self.grad = 0.0 # grad of output variable w.r.t itself
        self._backward = lambda: None
    
    def __repr__(self):
        return f"{self.label}: Value: {self.value}"
    
    def __add__(self, second_value: Any):
        if not isinstance(second_value, Datapoint):
            second_value = Datapoint(second_value)
        out = Datapoint(self.value + second_value.value, _prev=(self, second_value), _op = '+')
        def _backward():
            # grad accumilation so `+=`
            self.grad += out.grad
            second_value.grad += out.grad
        # in backprop, as we move from right to left, the output comes first. so attach the backward() to _backward of output varibale
        out._backward = _backward
        return out
    
    def __mul__(self, second_value: Any):
        if not isinstance(second_value, Datapoint):
            second_value = Datapoint(second_value)
        out = Datapoint(self.value * second_value.value, _prev=(self, second_value), _op = '*')
        def _backward():
            self.grad += second_value.value * out.grad
            second_value.grad += self.value * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self,):
        return Datapoint(-1.0) * self
    
    def __sub__(self, second_value: Any):
        if not isinstance(second_value, Datapoint):
            second_value = Datapoint(second_value)
        out = self + (-second_value)
        return out
    
    def __pow__(self, pow: int):
        if not isinstance(pow, Datapoint):
            pow = Datapoint(pow)
        out = Datapoint(math.pow(self.value, pow.value), _prev=(self,), _op='^')
        def _backward():
            self.grad += pow.value * (math.pow(self.value, (pow.value - 1))) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self, ):
        out = Datapoint(math.tanh(self.value), _prev=(self,), _op='tahH')
        def _backward():
            self.grad += (1 - out.value**2) * out.grad
        out._backward = _backward
        return out

    def get_children(self):
        return self._prev
    
    def backprop(self):
        
        # topological sort
        stack = [self]
        topological_sorted = []
        visited = set()
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                topological_sorted.append(node)
                for prev_node in node.get_children():
                    stack.append(prev_node)
        
        self.grad = 1.0
        for node in topological_sorted:
            node._backward()
    
# neuron class: tanh(x1*w1, x2*w2, x3*w3 + 1)
class Neuron:
    def __init__(self, num_inputs: int) -> None:
        self.weights = [Datapoint(np.random.uniform(-1.0, 1.0)) for _ in range(num_inputs)]
        self.bias = Datapoint(np.random.uniform(-1.0, 1.0))
    
    def __call__(self, x: list):
        out = sum((xi*wi for xi, wi in zip(x, self.weights)), self.bias)
        return out.tanh()
    def get_parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, num_inputs: int, num_neurons: int) -> None:
        self.layer = [Neuron(num_inputs) for _ in range(num_neurons)]

    def __call__(self, x: list):
        out = [neuron(x) for neuron in self.layer]
        return out[0] if len(out) == 1 else out
    
    def get_parameters(self):
        return [p for neuron in self.layer for p in neuron.get_parameters()]
    
class MLP:
    # num_inputs: int, layer_config: [4,4,1]
    def __init__(self, num_inputs: int, layer_config: list) -> None:
        # outpusl of 1st layer are inputs of 2nd layer
        graph_config = [num_inputs] + layer_config # [3, 4, 4, 1]
        # Layer 1 -> in: 3, neurons: 4
        # Layer 2: in: 4, neurons: 4
        # Layer 3: in: 4, nuerons: 1
        self.layers = [Layer(graph_config[i], graph_config[i+1]) for i in range(len(layer_config))]
        
        
    def __call__(self, x: list):
        # input -> L1 -> L2 -> L3
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_parameters(self):
        return [lp for layer in self.layers for lp in layer.get_parameters()]

if __name__ == '__main__':
    # 3 inputs
    X = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    Y = [1.0, -1.0, -1.0, 1.0]
    
    X = [[Datapoint(xii) for xii in xi] for xi in X]
    Y = [Datapoint(yi) for yi in Y]
    learning_rate = 0.01
    
    net = MLP(num_inputs=3, layer_config=[4,4,1])
    parameters = net.get_parameters()
    print(f'Number of params: {len(parameters)}')
    
    epochs = 50
    
    for e in range(epochs):
        # forward pass
        ypred = [net(x) for x in X]
        loss = sum(((yout - ygt)**2 for ygt, yout in zip(Y, ypred)), Datapoint(0.0))
        
        # backward pass
        for p in net.get_parameters():
            p.grad = 0.0
        loss.backprop()
        
        # update
        # grad always point towards the highest slope, hence move opposite to it
        for idx, p in enumerate(parameters):
            p.value += -learning_rate * p.grad
        
        print(f'E: {e} | Loss: {loss.value}')
    
    show_val = lambda x: round(float(x.value),2)
    ypred = [show_val(net(x)) for x in X]
    ytrue = [show_val(yi) for yi in Y]
    print(f'Y True: {ytrue}\nY Pred: {ypred}')
    
    # a = Datapoint(5)
    # b = Datapoint(6)
    # out = b - a; out.label = 'out'
    # print(out)
    # print(out)
    # out.backward()
    # d/dout(out) = 1 -> # gradient of output w.r.t = 1
    # operator +: d-out/d(a) & d-out/d(b) => d(a + b)d(a) = 1 + 0; => d(a + b)/d(b) = 0 + 1
    # operator *: d-out/d(a) & d-out/d(b) => d(a * b)d(a) = b => d(a * b)/d(b) = a
    # h = 0.00001, first principle of derivate -> f(a+h) - f(a) / h -> rate of chage
    # d/dh(tanh) = sec^2h = 1 - tan^2h
    # print(out)
        
