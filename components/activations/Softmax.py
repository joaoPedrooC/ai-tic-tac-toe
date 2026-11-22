import torch

class Softmax():
  def __init__(self):
    self.output = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=1, keepdim=True).values
    exp = torch.exp(x)
    denom = exp.sum()
    self.output = exp / denom
    return self.output

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    probs = self.output
    dot = (grad * probs).sum(dim=1, keepdim=True)
    return probs * (grad - dot)