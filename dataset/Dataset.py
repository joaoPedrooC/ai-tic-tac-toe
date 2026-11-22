import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, df: pl.DataFrame):
    super().__init__()
    self.df = df
    self.in_cols = [x for x in df.columns if 'board' in x]
    self.tgt_col = "best_moves"

  def __len__(self):
    return self.df.height

  def __getitem__(self, index):
    row = self.df.select(self.in_cols + [self.tgt_col]).row(index)
    return row[:-1], row[-1] # features, tgt