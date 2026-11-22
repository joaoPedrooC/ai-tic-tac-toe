import torch

class SiLU:
  def __init__(self):
    self.input = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    self.input = x

    sigmoid = 1 / (1 + torch.exp(-x))
    return x * sigmoid

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    sigmoid = 1 / (1 + torch.exp(-self.input))
    return grad * (sigmoid * (1 + self.input * (1 - sigmoid)))