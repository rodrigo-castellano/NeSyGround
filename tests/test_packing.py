"""Tests for grounder.packing: compact_atoms, pack_combined, pack_fact_rule."""

import torch
import pytest
from grounder.packing import compact_atoms, pack_combined, pack_fact_rule

PAD = 99


class TestCompactAtoms:
    def test_left_align_simple(self):
        # [pad, A, pad, B] → [A, B, pad, pad]
        states = torch.tensor([[
            [PAD, PAD, PAD],
            [1, 2, 3],
            [PAD, PAD, PAD],
            [4, 5, 6],
        ]])  # [1, 4, 3]
        result = compact_atoms(states, PAD)
        assert result[0, 0].tolist() == [1, 2, 3]
        assert result[0, 1].tolist() == [4, 5, 6]
        assert result[0, 2, 0].item() == PAD
        assert result[0, 3, 0].item() == PAD

    def test_no_gaps(self):
        states = torch.tensor([[
            [1, 2, 3],
            [4, 5, 6],
            [PAD, PAD, PAD],
        ]])
        result = compact_atoms(states, PAD)
        assert result[0, 0].tolist() == [1, 2, 3]
        assert result[0, 1].tolist() == [4, 5, 6]

    def test_all_padding(self):
        states = torch.full((1, 3, 3), PAD, dtype=torch.long)
        result = compact_atoms(states, PAD)
        assert (result == PAD).all()

    def test_3d_batch(self):
        # [B, S, M, 3]
        states = torch.full((2, 3, 4, 3), PAD, dtype=torch.long)
        states[0, 0, 2] = torch.tensor([1, 2, 3])
        states[1, 1, 3] = torch.tensor([4, 5, 6])
        result = compact_atoms(states, PAD)
        assert result[0, 0, 0].tolist() == [1, 2, 3]
        assert result[1, 1, 0].tolist() == [4, 5, 6]


class TestPackCombined:
    def test_compact_valid(self):
        B, K_total, M = 1, 4, 2
        states = torch.full((B, K_total, M, 3), PAD, dtype=torch.long)
        states[0, 1] = torch.tensor([[1, 2, 3], [4, 5, 6]])
        states[0, 3] = torch.tensor([[7, 8, 9], [10, 11, 12]])
        success = torch.tensor([[False, True, False, True]])
        derived, counts = pack_combined(states, success, K=3, M=M, padding_idx=PAD)
        assert counts[0].item() == 2
        assert derived.shape == (1, 3, 2, 3)
        # Valid entries should be in positions 0 and 1
        assert derived[0, 0, 0].tolist() == [1, 2, 3]
        assert derived[0, 1, 0].tolist() == [7, 8, 9]

    def test_cap_at_K(self):
        B, K_total, M = 1, 5, 1
        states = torch.ones(B, K_total, M, 3, dtype=torch.long)
        success = torch.ones(B, K_total, dtype=torch.bool)
        derived, counts = pack_combined(states, success, K=3, M=M, padding_idx=PAD)
        assert counts[0].item() == 3
        assert derived.shape == (1, 3, 1, 3)


class TestPackFactRule:
    def test_fact_and_rule(self):
        B, M, G = 1, 2, 3
        fact_gbody = torch.ones(B, 2, M, 3, dtype=torch.long)
        fact_goals = torch.full((B, 2, G, 3), PAD, dtype=torch.long)
        fact_valid = torch.tensor([[True, False]])
        fact_ridx = torch.tensor([[0, 0]])

        rule_gbody = torch.ones(B, 2, M, 3, dtype=torch.long) * 2
        rule_goals = torch.full((B, 2, G, 3), PAD, dtype=torch.long)
        rule_valid = torch.tensor([[True, False]])
        rule_ridx = torch.tensor([[1, 0]])

        gbody, goals, ridx, valid = pack_fact_rule(
            fact_gbody, fact_goals, fact_valid, fact_ridx,
            rule_gbody, rule_goals, rule_valid, rule_ridx,
            S=4, pad_idx=PAD)

        assert valid.shape == (1, 4)
        assert valid[0, 0].item() is True  # fact
        assert valid[0, 1].item() is True  # rule
        assert valid[0, 2].item() is False
        assert ridx[0, 0].item() == 0  # fact ridx
        assert ridx[0, 1].item() == 1  # rule ridx
