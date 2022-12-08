import numpy as np



# Setup of the snakes and ladders
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
    98: 78
}

def matrix_exp(A, n):
    # Compute the exponent of the matrix using numpy's matrix power function
    return np.linalg.matrix_power(A, n)

def roll_die():
    return np.random.randint(1, 7)

# Define a transition probability matrix for the above game board
def transition_matrix():
    # Initialize the transition matrix
    T = np.zeros((101, 101))

    # Loop through the squares on the board
    for i in range(0, 101):
        # Loop through the possible die rolls
        for j in range(1, 7):
            # Determine the new square
            new_square = i + j
            # Check if the new square is on a ladder or snake
            if new_square in game_board:
                new_square = game_board[new_square]

            if new_square>100:
                new_square = i

            # Update the transition matrix
            T[i, new_square] += 1 / 6

    # Update the transition matrix for the last square
    T[100, 100] = 1

    return T

mat = transition_matrix()
flag = True
# No need to check the last row we manually set it to 1
for i in range(100):
    if sum(mat[i]) != 0.9999999999999999:
        flag = False
print("The sum of all rows is 1: ", flag)
import matplotlib.pyplot as plt
plt.matshow(mat)
plt.show()
# use the above board to simulate a game of snakes and ladders

def simulate_game(T,initial_distribution):
    # Use matrix alegbra, to simulate the game
    # We start with the initial distribution
    # Then we multiply it by the transition matrix
    pass





initial_distribution = np.zeros(101)
initial_distribution[0] = 1