import numpy as np
import random
import matplotlib.pyplot as plt
roll_high = 6 # set to number of faces on the die.
random.seed(42)
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
    temp = A.copy()
    for i in range(1,n):
        temp = temp @ A
    return temp

def roll_die(roll_high):
    return np.random.randint(1, roll_high+1)


# Define a transition probability matrix for the above game board
def transition_matrix(roll):
    # Initialize the transition matrix
    T = np.zeros((101, 101))

    # Loop through the squares on the board
    for i in range(0, 101):
        # Loop through the possible die rolls
        for j in range(1, roll+1):
            # Determine the new square
            new_square = i + j
            # Check if the new square is on a ladder or snake
            if new_square in game_board:
                new_square = game_board[new_square]

            if new_square > 100:
                new_square = i

            # Update the transition matrix
            T[i, new_square] += 1 / roll
    # Update the transition matrix for the last square
    T[100, 100] = 1
    return T


mat = transition_matrix(6)
flag = True
# No need to check the last row we manually set it to 1
for i in range(100):
    if sum(mat[i]) != 0.9999999999999999:
        flag = False
        break

print("The sum of all rows is 1: ", flag)



plt.matshow(mat)
plt.title("Transition Probability Matrix")
plt.show()
# Write all the values of the transition probability matrix mat into a csv file called "transition_matrix.csv"
np.savetxt("matrix.csv", mat, delimiter=",")
# use the above board to simulate a game of snakes and ladders
hash = {}
freq = {}
boxmap = {}


def simulate_game(T, current_state=0):
    # Use matrix alegbra, to simulate the game
    # We start with the initial distribution
    # Then we multiply it by the transition matrix
    turns = 0
    while current_state < 100:
        current_state = np.random.choice(range(101), p=T[current_state])
        # print(current_state)
        turns += 1
        if current_state in boxmap.keys():
            boxmap[current_state] = boxmap[current_state] + 1
        else:
            boxmap[current_state] = 1
        if turns in hash.keys():
            hash[turns] = hash[turns] + 1
            freq[turns] = freq[turns] + 1
        else:
            hash[turns] = current_state
            freq[turns] = 1
    return turns


# Simulate 1000 games
turns = []
for i in range(1000):
    turns.append(simulate_game(mat))
print(np.mean(turns))

for i in hash.keys():
    hash[i] = hash[i] / freq[i]
x = sorted(hash.keys())
temp = []

for i in x:
    temp.append(hash[i])

y = sorted(boxmap.values(), reverse=True)

box_li = []
for i in boxmap.keys():
    tup = (boxmap[i], i)
    box_li.append(tup)

sorted_box_li = sorted(box_li, reverse=True)
# print(sorted_box_li)
# Print the distribution of the boxes visited(probability of reaching box)
summy = np.sum(turns)
z = []
for i in sorted_box_li:
    z.append([i[0] / summy, i[1]])

# Plot the histogram of the number of turns
plt.title("The number of turns to win the game")
plt.xlabel("Index of the Simulation(0-based)")
plt.ylabel("Number of turns simulation lasted for")
plt.plot(turns)
plt.show()

plt.title("Distribution of the number of turns")
plt.xlabel("Number of turns")
plt.ylabel("Average position")
plt.plot(temp)
plt.show()

plt.title("Distribution of the boxes(cells on the board) visited")
plt.xlabel("Box number")
plt.ylabel("Probability of reaching box")
plt.bar([i[1] for i in z], [i[0] for i in z])
plt.show()


# For sharing particular distributions after some time
def sharing_distribution(T, n):
    mat = T.copy()
    initial_distribution = np.zeros(101)
    initial_distribution[0] = 1.
    b = initial_distribution @ matrix_exp(mat, n)
    np.savetxt("share.csv", b, delimiter=",")
    plt.title("Initial Distribution after " +  str(n) + " transitions")
    plt.bar(range(101), b)
    plt.show()

def sharing_distribution_plot(T, n):
    mat = T.copy()
    initial_distribution = np.zeros(101)
    initial_distribution[0] = 1.
    b = initial_distribution @ matrix_exp(mat, n)

    np.savetxt("share.csv", b, delimiter=",")
    return b

plt.title("Game completion time")
plt.xlabel("Number of turns")
plt.ylabel("% of game completed")
for roll in range(6, 9):
    percent_dist = [sharing_distribution_plot(transition_matrix(roll), n)[-1] * 100 for n in range(300)]
    plt.plot(np.arange(300), percent_dist)
plt.legend(["Max die roll= 6", "Max die roll = 7", "Max die roll = 8"])
plt.show()
# feel free to change the input value here see the various distributions after some time
sharing_distribution(mat, 10) # makes sense



def limiting_distribution(M, n, epsilon):
    # compute M^n-1 and M^n and find the max difference between them if the max diff is below tolerance then its
    # limiting
    diff = matrix_exp(M,n)-matrix_exp(M,n-1)
    max_diff = np.max(diff)
    if (max_diff<epsilon):
        return True

for i in range(2,100):
    if limiting_distribution(mat, i, 0.0001):
        print(i)
        break
