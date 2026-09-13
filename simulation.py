"""Brute-force simulation of Snakes and Ladders.

Plays the game by actually rolling a die, and tallies the transitions it
observes into an empirical transition matrix.  That matrix is the check on
the analytic one built in model.py: the two should agree to within sampling
noise on every reachable square.

Run as a script:  python simulation.py
"""

import matplotlib.pyplot as plt
import numpy as np

import model
from model import N_SQUARES, SEED, WIN, game_board, roll_die

NUM_GAMES = 1000


def simulate_game(counts):
    """Play one game, tallying observed transitions into `counts`.

    Returns the squares visited in order, one entry per turn.  A turn where
    the roll would overshoot square 100 counts as a turn in which the player
    stayed put -- so it is recorded as a self-transition, exactly as the
    analytic matrix models it.
    """
    current_square = 0
    visited = []

    while current_square < WIN:
        squares_to_move = roll_die()
        previous_square = current_square

        # Overshooting the final square is not a legal move, so a roll that is
        # too large leaves the player exactly where they are.
        if current_square + squares_to_move <= WIN:
            current_square += squares_to_move
            # A snake or ladder moves the player on again.
            if current_square in game_board:
                current_square = game_board[current_square]

        counts[previous_square][current_square] += 1
        visited.append(current_square)

    return visited


def empirical_matrix(counts):
    """Row-normalise observed transition counts into a probability matrix."""
    T = np.asarray(counts, dtype=float)
    row_sums = T.sum(axis=1, keepdims=True)
    return np.divide(T, row_sums, out=np.zeros_like(T), where=row_sums != 0)


def main():
    np.random.seed(SEED)
    counts = np.zeros((N_SQUARES, N_SQUARES), dtype=np.int64)

    position_sum = {}
    turn_freq = {}
    visit_count = {}
    turns = []
    for _ in range(NUM_GAMES):
        visited = simulate_game(counts)
        turns.append(len(visited))
        for turn, square in enumerate(visited, start=1):
            visit_count[square] = visit_count.get(square, 0) + 1
            position_sum[turn] = position_sum.get(turn, 0) + square
            turn_freq[turn] = turn_freq.get(turn, 0) + 1

    print(np.mean(turns))

    average_position = [
        position_sum[turn] / turn_freq[turn] for turn in sorted(position_sum)
    ]

    # Probability of being on a given square on a randomly chosen turn.
    total_visits = sum(visit_count.values())
    squares = sorted(visit_count)
    visit_prob = [visit_count[s] / total_visits for s in squares]

    plt.title("The number of turns to win the game")
    plt.xlabel("Simulation Index(0-based)")
    plt.ylabel("Number of turns game lasted for")
    plt.plot(turns)
    plt.show()

    plt.title("Average position by turn number")
    plt.xlabel("Number of turns")
    plt.ylabel("Average position")
    plt.plot(average_position)
    plt.show()

    plt.title("How often each square is occupied")
    plt.xlabel("Box number")
    plt.ylabel("Probability of occupying box on a given turn")
    plt.bar(squares, visit_prob)
    plt.show()

    # Compare the simulated matrix against the analytic one from model.py.
    simulated = empirical_matrix(counts)
    np.savetxt("simulated_matrix.csv", simulated, delimiter=",")

    analytic = model.transition_matrix()
    visited_rows = [i for i in range(N_SQUARES) if counts[i].sum() > 0]
    worst = max(np.max(np.abs(simulated[i] - analytic[i])) for i in visited_rows)
    print("Simulated transition matrix written to simulated_matrix.csv")
    print("Rows observed at least once: ", len(visited_rows), "of", N_SQUARES)
    print("Largest disagreement with the analytic matrix: ", round(worst, 4))


if __name__ == "__main__":
    main()
