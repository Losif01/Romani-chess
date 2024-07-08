import chess
import numpy as np
import chess.engine
import random
# Load the Q-table
q_table = np.load("q_table.npy", allow_pickle=True).item()

# Initialize the chess engine (Stockfish)
engine_path = r"C:\Users\losif\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # Replace with your Stockfish path
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Function to let human play against AI
def play_human_vs_ai():
    board = chess.Board()

    while not board.is_game_over():
        print(board)  # Print the board for the human player
        if board.turn == chess.WHITE:
            human_move = input("Your move (e.g., e2e4): ")
            if not chess.Move.from_uci(human_move) in board.legal_moves:
                print("Invalid move. Try again.")
                continue
            board.push(chess.Move.from_uci(human_move))
        else:
            # AI's turn
            state = board.fen()
            legal_moves = list(board.legal_moves)
            if state in q_table:
                move_idx = np.argmax(q_table[state])
                ai_move = legal_moves[move_idx]
            else:
                ai_move = random.choice(legal_moves)
                print(f"Avoided random move: {ai_move}")
            board.push(ai_move)

        # Print the board after AI move
        print(board)
        print("\n\n\n\n\n\n")

    # Game ended
    print("Game over!")
    print(f"Result: {board.result()}")

# Play human vs AI
if __name__ == "__main__":
    play_human_vs_ai()

# Close the engine
engine.quit()
