import torch

class ReLU:
  def __init__(self):
    self.input = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    self.input = x
    return x * (x > 0).to(torch.int8)

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    return grad * (self.input > 0).to(torch.int8)