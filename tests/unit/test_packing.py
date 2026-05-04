"""Tests for grounder.bc.common: compact_atoms, pack_states."""

import torch
import pytest
from grounder.bc.common import compact_atoms, pack_states

PAD = 99


class TestCompactAtoms:
    def test_left_align_simple(self):
        # [pad, A, pad, B] -> [A, B, pad, pad]
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


class TestPackStates:
    """Tests for pack_states: flatten fact/rule resolution children into S_out states."""

    def test_single_fact_child(self):
        """One valid fact child should appear in output."""
        B, S_in, K_f, K_r, G, M_work = 1, 1, 2, 1, 3, 2
        S_out = 4

        # Fact resolution: 1 valid fact child out of K_f=2
        fact_goals = torch.full((B, S_in, K_f, G, 3), PAD, dtype=torch.long)
        fact_gbody = torch.full((B, S_in, K_f, M_work, 3), PAD, dtype=torch.long)
        fact_gbody[0, 0, 0] = torch.tensor([[1, 2, 3], [4, 5, 6]])
        fact_success = torch.tensor([[[True, False]]])
        fact_subs = torch.full((B, S_in, K_f, 2, 2), PAD, dtype=torch.long)

        # Rule resolution: no valid children
        rule_goals = torch.full((B, S_in, K_r, G, 3), PAD, dtype=torch.long)
        rule_gbody = torch.full((B, S_in, K_r, M_work, 3), PAD, dtype=torch.long)
        rule_success = torch.zeros(B, S_in, K_r, dtype=torch.bool)
        sub_rule_idx = torch.zeros(B, S_in, K_r, dtype=torch.long)
        rule_subs = torch.full((B, S_in, K_r, 2, 2), PAD, dtype=torch.long)

        # Parent state: already has a rule index (not first resolution)
        top_ridx = torch.tensor([[0]], dtype=torch.long)
        grounding_body = torch.ones(B, S_in, M_work, 3, dtype=torch.long)
        body_count = torch.tensor([[2]], dtype=torch.long)  # 2 active body atoms

        packed = pack_states(
            fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody, rule_success, sub_rule_idx,
            fact_subs, rule_subs,
            top_ridx, grounding_body, body_count,
            S_out, PAD,
        )

        assert packed.state_valid.shape == (B, S_out)
        assert packed.state_valid[0, 0].item() is True
        assert packed.state_valid[0, 1].item() is False

    def test_fact_and_rule_children(self):
        """Both fact and rule children should be packed together."""
        B, S_in, K_f, K_r, G, M_work = 1, 1, 1, 1, 3, 2
        S_out = 4

        # 1 valid fact
        fact_goals = torch.full((B, S_in, K_f, G, 3), PAD, dtype=torch.long)
        fact_gbody = torch.ones(B, S_in, K_f, M_work, 3, dtype=torch.long)
        fact_success = torch.tensor([[[True]]])
        fact_subs = torch.full((B, S_in, K_f, 2, 2), PAD, dtype=torch.long)

        # 1 valid rule
        rule_goals = torch.full((B, S_in, K_r, G, 3), PAD, dtype=torch.long)
        rule_goals[0, 0, 0, 0] = torch.tensor([1, 2, 3])  # remaining goal
        rule_gbody = torch.ones(B, S_in, K_r, M_work, 3, dtype=torch.long) * 2
        rule_success = torch.tensor([[[True]]])
        sub_rule_idx = torch.tensor([[[1]]])
        rule_subs = torch.full((B, S_in, K_r, 2, 2), PAD, dtype=torch.long)

        # First resolution (top_ridx == -1)
        top_ridx = torch.tensor([[-1]], dtype=torch.long)
        grounding_body = torch.full((B, S_in, M_work, 3), PAD, dtype=torch.long)
        body_count = torch.tensor([[0]], dtype=torch.long)  # no body atoms yet

        packed = pack_states(
            fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody, rule_success, sub_rule_idx,
            fact_subs, rule_subs,
            top_ridx, grounding_body, body_count,
            S_out, PAD,
        )

        assert packed.state_valid.shape == (B, S_out)
        # Fact child skipped at first resolution (body_count == 0),
        # rule child should be valid
        n_valid = packed.state_valid[0].sum().item()
        assert n_valid >= 1, f"Expected at least 1 valid child, got {n_valid}"

    def test_output_capped_at_S_out(self):
        """More valid children than S_out should be capped."""
        B, S_in, K_f, K_r, G, M_work = 1, 2, 2, 2, 3, 1
        S_out = 3  # cap at 3

        # All children valid
        fact_goals = torch.full((B, S_in, K_f, G, 3), PAD, dtype=torch.long)
        fact_gbody = torch.ones(B, S_in, K_f, M_work, 3, dtype=torch.long)
        fact_success = torch.ones(B, S_in, K_f, dtype=torch.bool)
        fact_subs = torch.full((B, S_in, K_f, 2, 2), PAD, dtype=torch.long)

        rule_goals = torch.full((B, S_in, K_r, G, 3), PAD, dtype=torch.long)
        rule_gbody = torch.ones(B, S_in, K_r, M_work, 3, dtype=torch.long)
        rule_success = torch.ones(B, S_in, K_r, dtype=torch.bool)
        sub_rule_idx = torch.ones(B, S_in, K_r, dtype=torch.long)
        rule_subs = torch.full((B, S_in, K_r, 2, 2), PAD, dtype=torch.long)

        # Not first resolution
        top_ridx = torch.tensor([[0, 1]], dtype=torch.long)
        grounding_body = torch.ones(B, S_in, M_work, 3, dtype=torch.long)
        body_count = torch.tensor([[1, 1]], dtype=torch.long)

        packed = pack_states(
            fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody, rule_success, sub_rule_idx,
            fact_subs, rule_subs,
            top_ridx, grounding_body, body_count,
            S_out, PAD,
        )

        assert packed.grounding_body.shape == (B, S_out, M_work, 3)
        assert packed.proof_goals.shape == (B, S_out, G, 3)
        assert packed.state_valid.shape == (B, S_out)
        assert packed.state_valid[0].sum().item() <= S_out

    def test_no_valid_children(self):
        """No valid children should produce all-False validity."""
        B, S_in, K_f, K_r, G, M_work = 1, 1, 1, 1, 2, 1
        S_out = 4

        fact_goals = torch.full((B, S_in, K_f, G, 3), PAD, dtype=torch.long)
        fact_gbody = torch.full((B, S_in, K_f, M_work, 3), PAD, dtype=torch.long)
        fact_success = torch.zeros(B, S_in, K_f, dtype=torch.bool)
        fact_subs = torch.full((B, S_in, K_f, 2, 2), PAD, dtype=torch.long)

        rule_goals = torch.full((B, S_in, K_r, G, 3), PAD, dtype=torch.long)
        rule_gbody = torch.full((B, S_in, K_r, M_work, 3), PAD, dtype=torch.long)
        rule_success = torch.zeros(B, S_in, K_r, dtype=torch.bool)
        sub_rule_idx = torch.zeros(B, S_in, K_r, dtype=torch.long)
        rule_subs = torch.full((B, S_in, K_r, 2, 2), PAD, dtype=torch.long)

        top_ridx = torch.tensor([[0]], dtype=torch.long)
        grounding_body = torch.ones(B, S_in, M_work, 3, dtype=torch.long)
        body_count = torch.tensor([[1]], dtype=torch.long)

        packed = pack_states(
            fact_goals, fact_gbody, fact_success,
            rule_goals, rule_gbody, rule_success, sub_rule_idx,
            fact_subs, rule_subs,
            top_ridx, grounding_body, body_count,
            S_out, PAD,
        )

        assert packed.state_valid.sum().item() == 0
