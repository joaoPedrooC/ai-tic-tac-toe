import torch

class Layer:
  def __init__(self, x_in: int, x_out: int, init_type: int):
    self.weights = torch.randn((x_in, x_out))
    # self.weights = torch.randn(x_in, x_out) * (2 / x_in) ** 0.5 if init_type == 1 else torch.randn(x_in, x_out) * (2 / (x_in + x_out)) ** 0.5
    self.bias = torch.zeros(1, x_out)
    self.input = None
    self.w_grad = None
    self.b_grad = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    self.input = x
    return x.matmul(self.weights) + self.bias

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    """
      grad is either the gradient from the loss w.r.t. the output of the layer
      or the gradient from the next layer's output w.r.t the input of the next layer
      As the operation is xW + b, I should compute the derivatives w.r.t. the bias (1 * grad),
      w.r.t. the weights (x * grad) and w.r.t. the input (W * grad)
    """
    self.w_grad = self.input.T.matmul(grad)
    self.b_grad = grad

    return grad.matmul(self.weights.T)

  def step(self, lr: float) -> None:
    self.weights -= self.w_grad * lr
    self.bias -= self.b_grad * lr

    self.w_grad = None
    self.b_grad = None
    self.input = None