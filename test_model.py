"""Tests for the Snakes and Ladders Markov model.

Each test pins down one of the properties the project relies on, so a
regression in the matrix or the analysis shows up as a failure rather than as
a slightly wrong figure.
"""

import itertools

import matplotlib

matplotlib.use("Agg")  # the modules import pyplot; keep tests headless

import numpy as np
import pytest

import model
import simulation

ROLLS = [6, 7, 8, 10, 12]


@pytest.mark.parametrize("roll", ROLLS)
def test_rows_are_probability_distributions(roll):
    T = model.transition_matrix(roll)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert (T >= 0).all()


@pytest.mark.parametrize("roll", ROLLS)
def test_win_square_is_absorbing(roll):
    T = model.transition_matrix(roll)
    assert T[model.WIN, model.WIN] == 1.0
    assert T[model.WIN].sum() == 1.0


def test_no_probability_mass_on_a_snake_or_ladder_head():
    """Landing on a head always moves the player on, so no head is ever occupied."""
    T = model.transition_matrix()
    for head in model.game_board:
        assert T[:, head].sum() == pytest.approx(T[head, head]), (
            f"square {head} is reachable but is a snake/ladder head"
        )


def test_overshooting_keeps_the_player_in_place():
    """From square 99 a 6-sided die only advances on a 1; the rest stay put."""
    T = model.transition_matrix(6)
    assert T[99, 99] == pytest.approx(5 / 6)
    assert T[99, 100] == pytest.approx(1 / 6)


def test_ladder_from_80_wins_the_game():
    T = model.transition_matrix(6)
    # From 79 a roll of 1 lands on 80, whose ladder goes straight to 100.
    assert T[79, 100] >= 1 / 6


def test_matrix_exp_zero_is_the_identity():
    T = model.transition_matrix()
    assert np.array_equal(model.matrix_exp(T, 0), np.eye(model.N_SQUARES))


def test_matrix_exp_matches_repeated_multiplication():
    T = model.transition_matrix()
    expected = np.eye(model.N_SQUARES)
    for n in range(1, 12):
        expected = expected @ T
        assert np.allclose(model.matrix_exp(T, n), expected)


def test_expected_turns_matches_the_known_value():
    """39.5984 turns from square 0 with a fair 6-sided die."""
    assert model.expected_turns(model.transition_matrix(6)) == pytest.approx(
        39.5984, abs=1e-3
    )


def test_a_bigger_die_finishes_the_game_sooner():
    previous = np.inf
    for roll in (6, 8, 10, 12):
        current = model.expected_turns(model.transition_matrix(roll))
        assert current < previous
        previous = current


def test_limiting_distribution_detects_convergence():
    T = model.transition_matrix()
    epsilon = 1e-4
    n = model.first_limiting_power(T, epsilon)
    assert n is not None
    assert model.limiting_distribution(T, n, epsilon)
    assert not model.limiting_distribution(T, n - 1, epsilon)


def test_limiting_distribution_uses_absolute_difference():
    """Successive powers far apart in magnitude must not count as settled."""
    T = model.transition_matrix()
    # At n=2 the largest signed difference is smaller than the largest
    # magnitude, so a signed comparison understates how far apart the powers are.
    diff = model.matrix_exp(T, 2) - model.matrix_exp(T, 1)
    assert np.max(np.abs(diff)) > np.max(diff)
    assert not model.limiting_distribution(T, 2, np.max(diff) + 1e-12)


def test_limiting_distribution_is_absorption_at_the_win_square():
    T = model.transition_matrix()
    settled = model.matrix_exp(T, 2000)
    assert settled[0, model.WIN] == pytest.approx(1.0, abs=1e-6)


def test_completion_curve_starts_at_zero_and_increases():
    curve = model.completion_curve(6, steps=60)
    assert curve[0] == 0.0
    assert all(b >= a for a, b in itertools.pairwise(curve))
    assert curve[-1] > 50.0


def test_sharing_distribution_is_a_distribution():
    T = model.transition_matrix()
    for n in (0, 1, 10, 100):
        d = model.sharing_distribution_plot(T, n)
        assert d.sum() == pytest.approx(1.0)
        assert (d >= 0).all()
    assert model.sharing_distribution_plot(T, 0)[0] == pytest.approx(1.0)


def test_simulated_matrix_records_overshoot_self_transitions():
    """The empirical matrix must show the self-loops on squares 96-99."""
    np.random.seed(7)
    counts = np.zeros((model.N_SQUARES, model.N_SQUARES), dtype=np.int64)
    for _ in range(3000):
        simulation.simulate_game(counts)
    empirical = simulation.empirical_matrix(counts)
    for square in (96, 97, 99):
        assert empirical[square, square] > 0.0, f"no self-loop recorded for {square}"


def test_simulated_matrix_agrees_with_the_analytic_one():
    np.random.seed(11)
    counts = np.zeros((model.N_SQUARES, model.N_SQUARES), dtype=np.int64)
    for _ in range(20000):
        simulation.simulate_game(counts)
    empirical = simulation.empirical_matrix(counts)
    analytic = model.transition_matrix()
    # Only compare rows with enough observations for the estimate to mean anything.
    for i in range(model.N_SQUARES):
        if counts[i].sum() >= 500:
            assert np.max(np.abs(empirical[i] - analytic[i])) < 0.05, (
                f"row {i} disagrees with the model"
            )


def test_simulated_game_length_matches_the_exact_expectation():
    np.random.seed(3)
    counts = np.zeros((model.N_SQUARES, model.N_SQUARES), dtype=np.int64)
    lengths = [len(simulation.simulate_game(counts)) for _ in range(20000)]
    exact = model.expected_turns(model.transition_matrix())
    assert np.mean(lengths) == pytest.approx(exact, rel=0.05)


def test_sampling_from_the_matrix_matches_the_brute_force_simulation():
    T = model.transition_matrix()
    np.random.seed(5)
    sampled = [len(model.simulate_game(T)) for _ in range(5000)]
    np.random.seed(5)
    counts = np.zeros((model.N_SQUARES, model.N_SQUARES), dtype=np.int64)
    brute = [len(simulation.simulate_game(counts)) for _ in range(5000)]
    assert np.mean(sampled) == pytest.approx(np.mean(brute), rel=0.05)


def test_seeding_makes_runs_reproducible():
    T = model.transition_matrix()
    np.random.seed(99)
    first = [len(model.simulate_game(T)) for _ in range(50)]
    np.random.seed(99)
    second = [len(model.simulate_game(T)) for _ in range(50)]
    assert first == second


def test_roll_die_stays_in_range():
    np.random.seed(1)
    for roll_high in (6, 8, 20):
        values = {model.roll_die(roll_high) for _ in range(500)}
        assert min(values) >= 1
        assert max(values) <= roll_high


def test_reachable_squares_excludes_every_head():
    T = model.transition_matrix()
    reachable = model.reachable_squares(T)
    assert len(reachable) == 82
    assert set(range(model.N_SQUARES)) - set(reachable) == set(model.game_board)


def test_expected_turns_ignores_the_unreachable_rows():
    """Keeping the unreachable head rows in I - Q must not change the answer."""
    T = model.transition_matrix()
    reachable = model.reachable_squares(T)
    transient = [s for s in reachable if s != model.WIN]
    Q = T[np.ix_(transient, transient)]
    restricted = np.linalg.inv(np.eye(len(transient)) - Q).sum(axis=1)[
        transient.index(0)
    ]
    assert model.expected_turns(T) == pytest.approx(restricted, abs=1e-9)


def test_i_minus_q_is_well_conditioned():
    T = model.transition_matrix()
    Q = np.delete(np.delete(T, model.WIN, axis=0), model.WIN, axis=1)
    assert np.linalg.cond(np.eye(len(Q)) - Q) < 100


def test_the_shortest_game_is_on_a_d15():
    """A bigger die helps only up to a point: overshoot wastes turns beyond d15."""
    lengths = {
        r: model.expected_turns(model.transition_matrix(r)) for r in range(2, 25)
    }
    assert min(lengths, key=lengths.get) == 15
    assert lengths[20] > lengths[12]
    assert lengths[3] > lengths[2]


def test_expected_turns_from_the_winning_square_is_zero():
    assert model.expected_turns(model.transition_matrix(), model.WIN) == 0.0


def test_both_files_roll_the_same_die():
    assert simulation.roll_die is model.roll_die


def test_importing_the_modules_does_not_reseed_the_global_generator():
    """A module that seeds at import time silently resets its caller's RNG."""
    import importlib

    np.random.seed(1234)
    expected = np.random.rand()
    np.random.seed(1234)
    importlib.reload(model)
    importlib.reload(simulation)
    assert np.random.rand() == expected


def test_transition_matrix_rejects_a_die_with_no_faces():
    for roll in (0, -3):
        with pytest.raises(ValueError, match="at least one face"):
            model.transition_matrix(roll)


def test_expected_turns_rejects_an_out_of_range_square():
    T = model.transition_matrix()
    for start in (-1, 101):
        with pytest.raises(ValueError, match="must be a square"):
            model.expected_turns(T, start)


def test_a_one_sided_die_still_builds_a_valid_chain():
    T = model.transition_matrix(1)
    assert np.allclose(T.sum(axis=1), 1.0)
