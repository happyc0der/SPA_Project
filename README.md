# Snakes and Ladders as a Markov Chain

A Markov-chain model of Snakes and Ladders, plus a brute-force simulation that
checks the model against actual dice rolls.

The board is a 101-state chain: squares `0`–`100`, where `0` is the starting
position off the board and `100` is the winning square. Because the player's
next square depends only on where they are now, the game is a Markov chain, and
one 101×101 transition matrix answers every question about it exactly — no
simulation needed. `simulation.py` exists to confirm the matrix is right.

## Contents

| File | What it does |
| --- | --- |
| `model.py` | Builds the transition matrix and does the analysis: expected game length, position distribution over time, completion curves, limiting distribution. |
| `simulation.py` | Plays games by rolling a die, and tallies an empirical transition matrix to compare against the analytic one. |
| `test_model.py` | Tests pinning down the properties both files rely on. |

## Running it

Needs Python 3.10 or newer.

```bash
pip install -r requirements.txt
python model.py       # the analysis and its figures
python simulation.py  # the brute-force check
pytest                # 47 tests
```

Each script opens its figures one at a time; close a window to get the next.
Both seed `numpy`'s generator inside `main()`, so repeated runs give identical
numbers while importing either module leaves your own random state alone.

Generated files (all git-ignored): `matrix.csv` (the transition matrix),
`share.csv` (the position distribution after 10 turns), `simulated_matrix.csv`
(the empirical matrix).

## The board

Nineteen snakes and ladders, as `landed-on square: square moved to` in
`game_board`:

```
ladders   1→38   4→14   9→31  21→42  28→84  36→44  51→67  71→91  80→100
snakes   16→6   48→26  49→11  56→53  62→19  64→60  87→24  93→73  95→75  98→78
```

**House rule:** a roll that would take the player past 100 is not played — they
stay where they are and lose the turn. So from square 99 a six-sided die
advances only on a 1, and `T[99, 99] = 5/6`.

## Results

With a fair six-sided die:

| Quantity | Value |
| --- | --- |
| Expected number of turns to win | **39.5984** |
| Fewest turns in which a win is possible | 7 (probability 0.0016) |
| Median game — half of games finished by turn | 33 |
| 90% of games finished by turn | 73 |
| 99% of games finished by turn | 130 |

The expected game length is computed exactly, from the chain rather than from
samples: with the absorbing square removed, `(I − Q)⁻¹` is the fundamental
matrix, and its row sums are the expected turns from each starting square. The
simulations agree with 39.5984 to within sampling noise.

### A bigger die helps, but only up to d15

39.60 turns on a d6, 34.85 on a d7, 31.97 on a d8 — `model.py` plots all three
completion curves together. Pushing further, the trend reverses: the shortest
game is **25.84 turns on a d15**, and a d20 is already worse than a d12
(27.22 vs 27.07). The house rule is why — the bigger the die, the more rolls
near the end of the board are too large to play, and the turn is wasted.

Going the other way is not monotone either: a **d3 is worse than a d2**
(81.18 turns against 72.01). With three faces the reachable squares line up
badly against this board's snakes.

Below that the board stops working altogether. A **d1 can never be won**: the
player walks 26, 27, … 47, lands on 48, and the snake there returns them to 26,
forever. Only 18 squares are reachable and 100 is not among them. `simulate_game`
raises rather than looping, and `expected_turns` reports the square as
unreachable instead of returning a number.

### Reading the "average position" figure

That curve is the mean square of the games **still in play** on a given turn, not
the average position of a player in a typical game. The difference is large: by
turn 80 the unconditional mean is 97, because most games have finished and are
sitting on square 100, while the games still running average 61.

It is also truncated at the last turn backed by at least 30 games. Untruncated it
ran to turn 213, where 11% of the points came from a single game — noise drawn as
signal.

### 82 of the 101 squares are reachable

Every one of the 19 snake and ladder heads is unreachable — `reachable_squares`
returns exactly the other 82. Landing on a head immediately moves the player on,
so no head is ever the square a turn *ends* on. Those rows still exist in the
matrix; they are transient states with no incoming probability.

They are harmless to keep. Each one is an ordinary stochastic row that reaches
100 eventually, so `I − Q` over all 100 non-winning squares stays invertible
and well conditioned (condition number ≈ 57), and `expected_turns` gets the same
39.5984 whether or not they are included.

### Stationary and limiting distribution

Square 100 is absorbing (`T[100, 100] = 1`), which settles both questions:

- **Stationary distribution:** unique, and it puts *all* mass on square 100.
  Every other square has probability 0. There is no interesting long-run
  distribution over the board, because the game ends.
- **Limiting distribution:** `Tⁿ` converges to that same answer. Measuring
  convergence as `max |Tⁿ − Tⁿ⁻¹| < ε`, it settles at **n = 165** for
  ε = 10⁻⁴ and **n = 279** for ε = 10⁻⁶.

The interesting behaviour is therefore all in the transient phase — how the
distribution spreads across the board before absorption — which is what the
`share.csv` and completion-time plots show.
