# Import the necessary libraries
import numpy as np

# Define a function to simulate the roll of a die
def roll_die():
    return np.random.randint(1, 7)

# Define a dictionary to represent the game board,
# where the keys are the squares on the board and the
# values are the corresponding positions after a ladder
# or snake is landed on
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

# Define a function to simulate a single game of Snakes and Ladders

def simulate_game():
    # Start the player at square 1
    current_square = 0
    turns = 0

    # Simulate the game until the player reaches the end of the board
    while current_square < 100:
        # Roll the die to determine the number of squares to move
        squares_to_move = roll_die()
        turns += 1

        # Move the player to the new square
        if (current_square  + squares_to_move) > 100:
            current_square = current_square
            continue
        else:
            current_square += squares_to_move
        # Check if the new square is on a ladder or snake
        # and move the player to the corresponding square
        if current_square in game_board:
            current_square = game_board[current_square]

        # print(current_square)

    # Return the number of turns it took for the player to reach the end of the board
    return turns

# Simulate a large number of games to estimate the expected number of turns
num_games = 10000
turns = []
for i in range(num_games):

    temp  =simulate_game()
    # print(i,temp)
    turns.append(temp)

# Print the average number of turns
print(np.mean(turns))