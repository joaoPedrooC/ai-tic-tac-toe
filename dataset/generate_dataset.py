import polars as pl

"""
Geração do dataset.
Contém:
Combinação de todas possíveis jogadas usando o algorítmo minimax (min para adversário e max para rede);
Verificação de vitória;
Combinação de possíveis jogadas com targets.
"""

EMPTY_CELL = 0
X = 1
O = -1

WIN_CONDITIONS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)
]

def check_winner(game_board: list[int]) -> str | None:
  for a, b, c in WIN_CONDITIONS:
    if game_board[a] != EMPTY_CELL and (game_board[a] == game_board[b] == game_board[c]):
      return game_board[a]
  return None

def game_ended(game_board: list[int]) -> bool:
  return check_winner(game_board) is not None or EMPTY_CELL not in game_board

def possible_moves(game_board: list[int]) -> list[int]:
  if game_ended(game_board):
    return []
  return [i for i, cell in enumerate(game_board) if cell == EMPTY_CELL]

def minimax(game_board: list[int], player_type: int) -> list[int]:
  result = check_winner(game_board)

  if result is not None: # X or O have won the game
    return result

  if EMPTY_CELL not in game_board: # it's a tie
    return 0

  scores = []

  for action in possible_moves(game_board):
    new_board = list(game_board)
    new_board[action] = player_type

    scores.append(minimax(tuple(new_board), -player_type))

  return max(scores) if player_type == X else min(scores)

def chose_best_moves(game_board: list[int], player_type: int) -> list[int]:
  if game_ended(game_board):
    return []

  scored_moves = []

  for action in possible_moves(game_board):
    new_board = list(game_board)
    new_board[action] = player_type

    score = minimax(tuple(new_board), -player_type)
    scored_moves.append((action, score))

  best_score = (
      max(score for _, score in scored_moves)
      if player_type == X else
      min(score for _, score in scored_moves)
  )

  return [
      action
      for action, score in scored_moves
      if score == best_score
  ]

def generate_dataset():
  visited = set()
  rows = []

  def explore(game_board: list[int]):
    if game_board in visited:
      return

    visited.add(game_board)

    curr_player = X if game_board.count(X) == game_board.count(O) else O

    result = check_winner(game_board)
    has_ended = game_ended(game_board)

    row = {
        f"board_{i}": game_board[i]
        for i in range(len(game_board))
    }

    row.update({
        "player": curr_player,
        "possible_moves": possible_moves(game_board),
        "best_moves": chose_best_moves(game_board, curr_player),
        "winner": result if result is not None else (0 if has_ended else ""),
        "game_ended": has_ended
    })

    rows.append(row)

    for action in possible_moves(game_board):
      new_board = list(game_board)
      new_board[action] = curr_player

      explore(tuple(new_board))

  initial_board = (EMPTY_CELL, ) * 9
  explore(initial_board)

  dataset = pl.DataFrame(rows)
  dataset = dataset.filter(
      (~pl.col("game_ended")) & (pl.col("best_moves").len() > 0) & (pl.col("player") == -1)
  )
  dataset_eval = dataset.sample(
      dataset.height * 25 // 100
  )

  dataset_train = dataset.join(
      dataset_eval, on=dataset.columns, how="anti"
  )

  # Melhora na visualização - exibe todas as linhas e colunas ao invés de truncar a tabela
  pl.Config.set_tbl_cols(-1)
  pl.Config.set_tbl_rows(-1)
  pl.Config.set_tbl_width_chars(-1)

  print("-"*100)
  print("Geração do dataset concluída!")
  print(f"{dataset.shape=}")
  print(f"{dataset_train.shape=}")
  print(f"{dataset_eval.shape=}")
  print(f"Sample para visualização:\n{dataset_train[:10]}")
  print("-"*100)
  return dataset_train, dataset_eval
