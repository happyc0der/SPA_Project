"""Markov-chain model of Snakes and Ladders.

Builds the exact transition matrix for the board below, then uses it to
answer the questions the project set out to answer: how long a game takes,
how the position distribution spreads out over time, and after how many
turns the chain has effectively settled (its limiting distribution).

Squares are numbered 0-100, where 0 is "off the board" (the starting
position) and 100 is the winning square.  A roll that would take the player
past 100 is not played: the player stays where they are.

Run as a script to reproduce every figure:  python model.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Seeded in main() rather than here: every random draw in this file goes
# through np.random, so seeding at import time would silently reset the global
# generator of anything that imports this module.
SEED = 42

N_SQUARES = 101  # squares 0..100 inclusive
WIN = 100
DEFAULT_ROLL_HIGH = 6  # number of faces on the die
N_GAMES = 1000
MAX_TURNS = 100_000  # a game longer than this means the board cannot be won
MIN_GAMES_PER_TURN = 30  # stop the average-position curve once samples run thin
COMPLETION_STEPS = 300

# The snakes and ladders: a key is the square landed on, the value is the
# square the player is moved to.  Values above the key are ladders, below
# are snakes.
game_board = {
    1: 38,
    4: 14,
    9: 31,
    21: 42,
    28: 84,
    36: 44,
    51: 67,
    71: 91,
    80: 100,
    16: 6,
    48: 26,
    49: 11,
    56: 53,
    62: 19,
    64: 60,
    87: 24,
    93: 73,
    95: 75,
    98: 78,
}


def matrix_exp(A, n):
    """Return A raised to the n-th power (n == 0 gives the identity)."""
    return np.linalg.matrix_power(A, n)


def roll_die(roll_high=DEFAULT_ROLL_HIGH):
    """Return a uniform roll in 1..roll_high."""
    return np.random.randint(1, roll_high + 1)


def transition_matrix(roll=DEFAULT_ROLL_HIGH):
    """Build the one-step transition matrix for a die with `roll` faces."""
    if roll < 1:
        raise ValueError(f"a die needs at least one face, got {roll}")
    T = np.zeros((N_SQUARES, N_SQUARES))

    for i in range(N_SQUARES):
        for j in range(1, roll + 1):
            new_square = i + j
            # Overshooting the final square is not a legal move: stay put.
            if new_square > WIN:
                new_square = i
            # A snake or ladder moves the player on again.
            elif new_square in game_board:
                new_square = game_board[new_square]

            T[i, new_square] += 1 / roll

    # The winning square is absorbing.
    T[WIN] = 0.0
    T[WIN, WIN] = 1.0
    return T


def simulate_game(T, current_state=0):
    """Play one game by sampling from `T`; return the squares visited in order.

    Raises if the game runs past MAX_TURNS.  Not every die makes this board
    winnable -- a 1-sided die traps the player in the loop 26..47 -> 48 -> 26,
    so square 100 is never reached and the loop would otherwise never end.
    """
    visited = []
    while current_state < WIN:
        current_state = np.random.choice(N_SQUARES, p=T[current_state])
        visited.append(current_state)
        if len(visited) > MAX_TURNS:
            raise RuntimeError(
                f"no win after {MAX_TURNS} turns from square {current_state}; "
                "this board and die cannot reach square 100"
            )
    return visited


def average_position_by_turn(position_sum, turn_freq, min_games=MIN_GAMES_PER_TURN):
    """Mean square of the games still in play, turn by turn.

    Truncated at the last turn backed by at least `min_games` games.  Past that
    point the average is drawn from a handful of unusually long games and is
    noise rather than signal -- over 1000 games the tail was 11% single-game
    points.  `turn_freq` counts games lasting at least that many turns, so it
    only ever decreases and the first thin turn ends the curve.
    """
    curve = []
    for turn in sorted(position_sum):
        if turn_freq[turn] < min_games:
            break
        curve.append(position_sum[turn] / turn_freq[turn])
    return curve


def reachable_squares(T, start=0):
    """Squares that can actually be occupied at the end of a turn, from `start`.

    Every snake and ladder head is excluded: landing on one moves the player
    straight on, so no head is ever the square a turn ends on.
    """
    reachable = {start}
    frontier = [start]
    while frontier:
        square = frontier.pop()
        for successor in np.nonzero(T[square])[0]:
            successor = int(successor)
            if successor not in reachable:
                reachable.add(successor)
                frontier.append(successor)
    return sorted(reachable)


def expected_turns(T, start=0):
    """Exact expected number of turns to reach square 100, from the chain itself.

    Drops the absorbing square to leave the substochastic matrix Q, then sums a
    row of the fundamental matrix (I - Q)^-1.  Rows for unreachable snake and
    ladder heads are kept: they are ordinary stochastic rows that all lead to
    100 eventually, so I - Q stays well conditioned, and their presence does
    not affect the answer for a reachable `start`.
    """
    if not 0 <= start <= WIN:
        raise ValueError(f"start must be a square in 0..{WIN}, got {start}")
    if start == WIN:
        return 0.0
    if WIN not in reachable_squares(T, start):
        raise ValueError(
            f"square {WIN} is unreachable from square {start} with this die, "
            "so the expected number of turns is infinite"
        )
    Q = np.delete(np.delete(T, WIN, axis=0), WIN, axis=1)
    fundamental = np.linalg.inv(np.eye(len(Q)) - Q)
    return fundamental.sum(axis=1)[start]


def sharing_distribution_plot(T, n):
    """Return the position distribution after `n` turns, starting from square 0."""
    initial_distribution = np.zeros(N_SQUARES)
    initial_distribution[0] = 1.0
    return initial_distribution @ matrix_exp(T, n)


def sharing_distribution(T, n):
    """Plot the position distribution after `n` turns and save it to share.csv."""
    b = sharing_distribution_plot(T, n)
    np.savetxt("share.csv", b, delimiter=",")
    plt.title("Initial Distribution after " + str(n) + " transitions")
    plt.bar(range(N_SQUARES), b)
    plt.show()
    return b


def limiting_distribution(M, n, epsilon):
    """True if M^n and M^(n-1) agree to within `epsilon` in every entry."""
    diff = matrix_exp(M, n) - matrix_exp(M, n - 1)
    return bool(np.max(np.abs(diff)) < epsilon)


def first_limiting_power(M, epsilon, max_n=600):
    """Smallest n <= max_n for which the chain has settled, else None.

    Walks the powers incrementally instead of recomputing M^n from scratch
    for every candidate n.
    """
    previous = np.eye(len(M))
    current = M.copy()
    for n in range(1, max_n + 1):
        if np.max(np.abs(current - previous)) < epsilon:
            return n
        previous, current = current, current @ M
    return None


def completion_curve(roll, steps=COMPLETION_STEPS):
    """Percentage of games finished after 0, 1, ... steps-1 turns."""
    T = transition_matrix(roll)
    distribution = np.zeros(N_SQUARES)
    distribution[0] = 1.0
    finished = []
    for _ in range(steps):
        finished.append(distribution[WIN] * 100)
        distribution = distribution @ T
    return finished


def main():
    np.random.seed(SEED)
    mat = transition_matrix(DEFAULT_ROLL_HIGH)

    row_sums = mat.sum(axis=1)
    print("The sum of all rows is 1: ", bool(np.allclose(row_sums, 1.0)))

    plt.matshow(mat)
    plt.title("Transition Probability Matrix")
    plt.show()
    # Save every entry of the transition matrix to matrix.csv
    np.savetxt("matrix.csv", mat, delimiter=",")

    # Simulate games by sampling from the transition matrix.
    position_sum = {}
    turn_freq = {}
    visit_count = {}
    turns = []
    for _ in range(N_GAMES):
        visited = simulate_game(mat)
        turns.append(len(visited))
        for turn, square in enumerate(visited, start=1):
            visit_count[square] = visit_count.get(square, 0) + 1
            position_sum[turn] = position_sum.get(turn, 0) + square
            turn_freq[turn] = turn_freq.get(turn, 0) + 1

    print(np.mean(turns))
    print("Exact expected number of turns: ", round(expected_turns(mat), 4))

    average_position = average_position_by_turn(position_sum, turn_freq)

    # Probability of being on a given square on a randomly chosen turn.
    total_visits = sum(visit_count.values())
    squares = sorted(visit_count)
    visit_prob = [visit_count[s] / total_visits for s in squares]

    plt.title("The number of turns to win the game")
    plt.xlabel("Index of the Simulation(0-based)")
    plt.ylabel("Number of turns simulation lasted for")
    plt.plot(turns)
    plt.show()

    plt.title("Average position while the game is still running")
    plt.xlabel("Number of turns")
    plt.ylabel("Mean square of games still in play")
    plt.plot(average_position)
    plt.show()

    plt.title("How often each square is occupied")
    plt.xlabel("Box number")
    plt.ylabel("Probability of occupying box on a given turn")
    plt.bar(squares, visit_prob)
    plt.show()

    plt.title("Game completion time")
    plt.xlabel("Number of turns")
    plt.ylabel("% of game completed")
    rolls = range(6, 9)
    for roll in rolls:
        plt.plot(np.arange(COMPLETION_STEPS), completion_curve(roll))
    plt.legend(["Max die roll= " + str(r) for r in rolls])
    plt.show()

    # Feel free to change the input value here to see the distribution at
    # other points in time.
    sharing_distribution(mat, 10)

    epsilon = 0.0001
    n = first_limiting_power(mat, epsilon)
    if n is None:
        print("No limiting distribution within the search range for epsilon =", epsilon)
    else:
        print(n)


if __name__ == "__main__":
    main()
