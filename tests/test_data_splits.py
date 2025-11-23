"""Tests for leave-one-out splitting and windowing (anti-leakage protocol)."""

from recommender.data.amazon import synthetic_interactions
from recommender.data.splits import leave_one_out, train_windows


def test_leave_one_out_targets_are_chronological():
    # Out-of-order timestamps must be sorted before splitting.
    interactions = {0: [(10, 3.0), (11, 1.0), (12, 2.0), (13, 4.0)]}
    splits = leave_one_out(interactions, min_seq_len=3)
    s = splits[0]
    # Sorted by ts: [11, 12, 10, 13]; test=last, valid=2nd-last.
    assert s.test_target == 13
    assert s.valid_target == 10
    assert s.train_seq == [11, 12]
    assert s.test_seq == [11, 12, 10]


def test_no_future_leakage():
    # The test history must be a strict prefix that excludes the test target.
    interactions = {0: [(i, float(i)) for i in range(1, 11)]}
    s = leave_one_out(interactions, min_seq_len=3)[0]
    assert s.test_target == 10
    assert 10 not in s.test_seq
    assert s.valid_target not in s.valid_seq
    # Validation history must not contain the validation or test targets.
    assert s.valid_target == 9
    assert 9 not in s.valid_seq and 10 not in s.valid_seq


def test_short_users_dropped():
    interactions = {0: [(1, 1.0), (2, 2.0)], 1: [(i, float(i)) for i in range(5)]}
    splits = leave_one_out(interactions, min_seq_len=3)
    assert 0 not in splits and 1 in splits


def test_train_windows_shifted_targets():
    inputs, targets = train_windows([1, 2, 3, 4], max_len=10)
    assert inputs == [1, 2, 3]
    assert targets == [2, 3, 4]


def test_train_windows_truncates():
    seq = list(range(1, 21))
    inputs, targets = train_windows(seq, max_len=5)
    assert len(inputs) == 5 and len(targets) == 5
    assert targets[-1] == 20


def test_synthetic_dataset_consistent():
    ds = synthetic_interactions(num_users=20, num_items=30, seed=0)
    assert ds.num_users == 20
    # All item ids fall in [1, num_items].
    for events in ds.interactions.values():
        for item, _ in events:
            assert 1 <= item <= ds.num_items
