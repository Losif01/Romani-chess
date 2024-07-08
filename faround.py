

import chess
import chess.engine
import chess.pgn
import numpy as np
import random
import os

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-table
q_table = {}

def get_state(board):
    return board.fen()

def choose_action(board, epsilon):
    legal_moves = list(board.legal_moves)
    if random.uniform(0, 1) < epsilon:
        return random.choice(legal_moves)
    else:
        state = get_state(board)
        if state not in q_table:
            q_table[state] = np.zeros(len(legal_moves))
        move_indices = {move: i for i, move in enumerate(legal_moves)}
        best_move_idx = np.argmax(q_table[state])
        return legal_moves[best_move_idx]

def update_q_table(state, action, reward, next_state):
    legal_moves = list(chess.Board(state).legal_moves)
    move_indices = {move: i for i, move in enumerate(legal_moves)}
    action_idx = move_indices[action]

    if state not in q_table:
        q_table[state] = np.zeros(len(legal_moves))
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(list(chess.Board(next_state).legal_moves)))

    old_value = q_table[state][action_idx]
    next_max = np.max(q_table[next_state])

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
    q_table[state][action_idx] = new_value

def play_game(engine):
    board = chess.Board()
    state = get_state(board)
    moves_played = 0
    while not board.is_game_over():
        action = choose_action(board, epsilon)
        board.push(action)
        next_state = get_state(board)
        reward = get_reward(board, engine)
        update_q_table(state, action, reward, next_state)
        state = next_state
        moves_played += 1
    return board.result(), moves_played

def get_reward(board, engine):
    if board.is_checkmate():
        return 1 if board.turn == chess.BLACK else -1
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0
    else:
        info = engine.analyse(board, chess.engine.Limit(time=0.2))
        return info["score"].relative.score(mate_score=1000) / 1000

def evaluate_game(pgn, engine):
    game = chess.pgn.read_game(pgn)
    board = game.board()
    white_score = 0
    black_score = 0
    move_count = 0

    for move in game.mainline_moves():
        board.push(move)
        move_count += 1
        info = engine.analyse(board, chess.engine.Limit(time=0.2))
        score = info["score"].relative.score(mate_score=1000) / 1000
        if board.turn == chess.WHITE:
            black_score += score
        else:
            white_score += score

    white_accuracy = white_score / move_count
    black_accuracy = black_score / move_count

    return white_accuracy, black_accuracy

# Load the Stockfish engine
engine_path = r"there was a path here pointing to the stockfish chess engine used in the reward function"
# Print the engine path to ensure it is correct
print(f"Engine path: {engine_path}")

# Check if the file exists
if not os.path.isfile(engine_path):
    print("The specified path does not point to a valid file.")
else:
    print("The specified path points to a valid file.")

# Load the Stockfish engine using chess.engine
try:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
except Exception as e:
    print(f"Failed to load Stockfish engine: {e}")
    engine = None

# Training the agent
if engine:
    for episode in range(100):
        result, moves_played = play_game(engine)
        print(f"Episode {episode + 1}: Game result: {result}, Moves played: {moves_played}")

    # Save the Q-table
    np.save("q_table.npy", q_table)

    # Close the chess engine
    engine.quit()
    print("Training completed and Q-table saved.")
else:
    print("Failed to load the Stockfish engine. Training aborted.")

# Example usage of the evaluation function
if engine:
    with open("game.pgn") as pgn:
        white_accuracy, black_accuracy = evaluate_game(pgn, engine)
        print(f"White accuracy: {white_accuracy:.2f}")
        print(f"Black accuracy: {black_accuracy:.2f}")
else:
    print("Failed to load the Stockfish engine. Evaluation aborted.")
