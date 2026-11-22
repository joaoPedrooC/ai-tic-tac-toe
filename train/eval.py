import torch

def eval(
    model: NN,
    eval_set: torch.utils.data.DataLoader,
    method: str = "accuracy"
) -> tuple[NN, float]:
  correct_classified = 0
  total_classifications = 0
  for step, (features, tgt) in enumerate(eval_set):
    features = torch.tensor(features).unsqueeze(0).to(torch.float32)
    out = model.forward(features)

    selected = out.argmax(dim=-1)
    if selected in tgt:
      correct_classified += 1
    total_classifications += 1
  return model, correct_classified / total_classifications
