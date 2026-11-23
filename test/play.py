import torch

EMPTY_CELL = 0
X = 1
O = -1

WIN_CONDITIONS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)
]

def show_board(board):
    symbols = {
        1: " X ",
        -1: " O ",
        0: "   "
    }

    print("\n")
    for i in range(3):
        row = board[i * 3:i * 3 + 3]

        print(" | ".join(symbols[x] for x in row))

        if i < 2:
            print("-" * 11)

def check_winner(game_board: list[int]) -> str | None:
  for a, b, c in WIN_CONDITIONS:
    if game_board[a] != EMPTY_CELL and (game_board[a] == game_board[b] == game_board[c]):
      return game_board[a]
  return None

def play(model):
    board = [0] * 9

    model.eval() if hasattr(model, "eval") else None
    print("Você é X (1). A rede é O (-1).")
    show_board(board)

    while True:
        print("\nÍndices:")
        print(" 0 | 1 | 2 ")
        print("---+---+---")
        print(" 3 | 4 | 5 ")
        print("---+---+---")
        print(" 6 | 7 | 8 ")
        print()

        try:
            move = int(input("Escolha uma posição (0-8): "))
        except ValueError:
            print("Digite um número válido.")
            continue

        if move < 0 or move > 8:
            print("Posição inválida.")
            continue

        if board[move] != 0:
            print("Essa posição já está ocupada.")
            continue

        board[move] = 1
        show_board(board)

        winner = check_winner(board)

        if winner == 1:
            print("Você venceu!")
            break

        if 0 not in board:
            print("Empate!")
            break

        # Jogada da rede
        features = torch.tensor(
            board, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            probs = model.forward(features)

        # Impede a rede de escolher posições ocupadas.
        scores = probs.squeeze(0).clone()

        # for i in range(9):
        #     if board[i] != 0:
        #         scores[i] = -float("inf")

        # Três melhores posições disponíveis.
        top_values, top_indices = torch.topk(
            scores, k=min(3, board.count(0))
        )

        print("Top 3 opções da rede:")

        for rank, (idx, value) in enumerate(
            zip(top_indices.tolist(), top_values.tolist()), 1
        ):
            print(
                f"{rank}º - Posição {idx} | "
                f"Probabilidade: {value:.6f}"
            )

        # A rede joga sua melhor opção.
        ai_move = top_indices[0].item()
        board[ai_move] = -1

        print(f"\nA rede escolheu a posição {ai_move}.")

        show_board(board)

        winner = check_winner(board)

        if winner == -1:
            print("A rede venceu!")
            break

        if 0 not in board:
            print("Empate!")
            break

    print("Fim de jogo!")
