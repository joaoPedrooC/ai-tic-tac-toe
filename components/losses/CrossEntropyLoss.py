import torch

class CrossEntropyLoss:
  def __init__(self):
    self.input = None
    self.y = None

  def forward(self, probs: torch.Tensor, y: list[int]) -> float:
    """
      Here we consider all the possible moves as targets and in order not to cause
      conflict (each move counts as a target, so we try to maximize all of them separately),
      we gather the classes' probabilities.
    """
    self.input = probs
    self.y = y

    loss = probs[0, y].sum()
    return -torch.log(loss)

  def backward(self):
    grad = torch.zeros_like(self.input)
    s = self.input[0, self.y].sum()

    grad[0, self.y] = - 1 / s
    return grad