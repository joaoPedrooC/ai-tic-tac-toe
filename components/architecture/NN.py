from components.activations.Softmax import Softmax
from components.activations.ReLU import ReLU
from components.activations.SiLU import SiLU
from .Layer import Layer

import torch

class NN:
  def __init__(self, layers: list[int]):
    self.layers = []

    for i, (x_in, x_out) in enumerate(zip(layers[:-1], layers[1:])):
      self.layers.extend([
          Layer(x_in, x_out, 1 if i != (len(layers) - 2) else 2),
          SiLU() if i != (len(layers) - 2) else Softmax()
      ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, grad: torch.Tensor) -> None:
    for layer in reversed(self.layers):
      grad = layer.backward(grad)

  def step(self, lr: float) -> None:
    for module in self.layers:
      if hasattr(module, "step"):
        module.step(lr)