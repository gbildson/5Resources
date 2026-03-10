from catan_rl.bots import RandomLegalAgent
from catan_rl.eval import play_match, tournament


def test_truncation_assigns_winner_by_highest_vp_after_one_step():
    agents = [RandomLegalAgent(seed=1) for _ in range(4)]
    res = play_match(agents, seed=42, max_steps=1)
    assert res.truncated
    assert res.winners == [0]


def test_truncation_tie_splits_win_credit():
    agents = [RandomLegalAgent(seed=2) for _ in range(4)]
    stats = tournament(agents, num_games=1, base_seed=99, max_steps=0)
    assert stats["truncated_games"] == 1
    assert stats["terminal_games"] == 0
    assert stats["wins"] == [0.25, 0.25, 0.25, 0.25]
    assert stats["win_rates"] == [0.25, 0.25, 0.25, 0.25]
    assert stats["strict_win_rates"] == [0.0, 0.0, 0.0, 0.0]
