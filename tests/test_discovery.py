import pytest

from nufield.discovery.objectives import CandidateScore
from nufield.discovery.search import SearchCandidate, candidate_grid


def test_candidate_search_grid():
    result = candidate_grid([0.5, 1.0], [16, 32])
    assert len(result) == 4


def test_unsafe_candidate_rejected():
    with pytest.raises(ValueError):
        SearchCandidate(100, 32).validate()


def test_candidate_score_gates():
    score = CandidateScore(0.1, 0.1, 0.02, 5.0, 100.0, True)
    assert score.admissible()
