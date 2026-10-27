from lifeos.retrieval import reciprocal_rank_fusion


def test_rrf_scores_and_order():
    fused = reciprocal_rank_fusion({"v": ["a", "b", "c"], "g": ["c", "a"]}, k=60)
    ids = [d for d, _, _ in fused]
    assert ids == ["a", "c", "b"]
    score_a = dict((d, s) for d, s, _ in fused)["a"]
    assert abs(score_a - (1 / 61 + 1 / 62)) < 1e-12


def test_rrf_reports_ranks_and_handles_single_source():
    fused = reciprocal_rank_fusion({"v": ["x", "y"]})
    assert fused[0][2] == {"v": 1}
    assert reciprocal_rank_fusion({}) == []


def test_rrf_tie_break_is_deterministic():
    assert [d for d, _, _ in reciprocal_rank_fusion({"v": ["b"], "g": ["a"]})] == ["a", "b"]
