from joblib import Parallel, delayed

from .eval import eval

import torch
import copy

def layer_norm(model: NN) -> NN:
  for layer_idx, layer in enumerate(model.layers):
    if not hasattr(layer, "weights"):
      continue
    mean = layer.weights.mean(dim=1, keepdim=True)
    std = ((layer.weights - mean) ** 2).mean(dim=1, keepdim=True)

    new_w = (layer.weights - mean) / torch.sqrt(std + 1e-8)
    model.layers[layer_idx].weights = layer.weights + (layer.weights * new_w)
  return model

def std(model: NN) -> NN:
  for layer_idx, layer in enumerate(model.layers):
    if not hasattr(layer, "weights"):
      continue
    mean = layer.weights.mean(dim=1, keepdim=True)
    noise = (layer.weights - mean) ** 2
    noise = torch.randn_like(layer.weights) * torch.sqrt(noise / noise.shape[1])

    model.layers[layer_idx].weights = layer.weights + noise
  return model

def rnd(model: NN) -> NN:
  for layer_idx, layer in enumerate(model.layers):
    if not hasattr(layer, "weights"):
      continue
    model.layers[layer_idx].weights = layer.weights + (torch.randn_like(layer.weights) - 0.5)
  return model

def mutate(model: NN, mutation_type: int) -> NN:
  mutations = {
      1: layer_norm,
      2: std,
      3: rnd
  }

  if mutation_type in mutations:
    return mutations[mutation_type](model)

  return model

def genetic_algorithm(
    model: NN,
    eval_set: torch.utils.data.DataLoader
) -> NN:
  """
    Objective:
    clone the principal neural network three times, apply mutation on the clones and keep the best.
    Four networks will be evaluated:
      1. Original network (w);
      2. w + w * LayerNorm(w);
      3. w + w * std(w);
      4. w + random [-0.5, 0.5] (absolute random variation)
  """
  networks_accs = Parallel(n_workers=4)(delayed(eval)(mutate(copy.deepcopy(model), i), eval_set, "accuracy") for i in range(4))
  networks = [x[0] for x in networks_accs]
  accs = [x[1] for x in networks_accs]

  best_idx = 0
  for idx, i in enumerate(accs[1:]):
    if accs[best_idx] < i:
      best_idx = idx

  print_msg = f"""
***********************************************************
***********************************************************
*******       RESULTADO DO ALGORITMO GENÉTICO       *******
*******       0. Original: {accs[0]:.6f}                 *******
*******       1. LayerNorm: {accs[1]:.6f}                *******
*******       2. Standard Deviation {accs[2]:.6f}        *******
*******       3. Absolute variation {accs[3]:.6f}        *******
***********************************************************
***********************************************************

MELHOR MODELO (SELECIONADO PARA CONTINUAR O TREINAMENTO COM BACKPROPAGATION):
{'Original' if best_idx == 0 else 'LayerNorm' if best_idx == 1 else 'StandardDeviation' if best_idx == 2 else 'AbsoluteVariation'}
  """
  print(print_msg)

  return networks[best_idx]