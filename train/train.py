from components.losses.CrossEntropyLoss import CrossEntropyLoss

from dataset.generate_dataset import generate_dataset
from dataset.Dataset import Dataset

from .GeneticAlgorithm import genetic_algorithm
from test.play import play
from .eval import eval

from components.architecture.NN import NN

import torch

def train(
    model: NN,
    train_set: torch.utils.data.DataLoader,
    eval_set: torch.utils.data.DataLoader | None,
    enable_alg_gen: bool = False,
    alg_gen_every_n_epoch: int = 5,
    max_epochs: int =  250,
    lr: float = 1e-2
) -> None:
  for epoch in range(max_epochs):
    print(f"{'-'*25}     Epoch [{epoch + 1}]     {'-'*25}")
    for step, (features, tgt) in enumerate(train_set):
      features = torch.tensor(features).unsqueeze(0).to(torch.float32)

      out = model.forward(features)
      loss = CrossEntropyLoss()
      fwd_loss = loss.forward(out, tgt)

      if step % 200 == 0:
        print(f"Step {step} | loss {fwd_loss.item():.6f}")

      loss_grad = loss.backward()
      model.backward(loss_grad)
      model.step(lr=lr)

    _, acc = eval(model, eval_set, "accuracy")
    print(f"[EPOCH EVAL] | Epoch [{epoch + 1}]: Accuracy {acc}")

    if acc >= 0.99:
      return

    if enable_alg_gen and ((epoch + 1) % alg_gen_every_n_epoch == 0):
      model = genetic_algorithm(model, eval_set)

model = NN(
    [9, 32, 9]
)

dataset_train, dataset_eval = generate_dataset()

train_set = torch.utils.data.DataLoader(
    Dataset(dataset_train), shuffle=True
)
eval_set = torch.utils.data.DataLoader(
    Dataset(dataset_eval)
)

train(
    model=model,
    train_set=train_set,
    eval_set=eval_set,
    enable_alg_gen=True,
    alg_gen_every_n_epoch=5,
    max_epochs=100,
    lr=1e-2
)

play(model)