"""Tests for grounder.primitives: unify_one_to_one + apply_substitutions."""

import torch
import pytest
from grounder.primitives import apply_substitutions, unify_one_to_one

PAD = 99
C_NO = 5  # constants 0..5, vars start at 6


class TestUnifyOneToOne:
    """Test pairwise unification."""

    def test_identical_ground_atoms(self):
        q = torch.tensor([[1, 2, 3]])
        t = torch.tensor([[1, 2, 3]])
        ok, subs = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is True
        # No substitutions needed (both ground and equal)
        assert (subs[0] == PAD).all()

    def test_predicate_mismatch(self):
        q = torch.tensor([[1, 2, 3]])
        t = torch.tensor([[2, 2, 3]])
        ok, _ = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is False

    def test_constant_conflict(self):
        q = torch.tensor([[1, 2, 3]])
        t = torch.tensor([[1, 2, 4]])
        ok, _ = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is False

    def test_query_var_binds_to_constant(self):
        # q has var at arg0 (index 6 > C_NO=5)
        q = torch.tensor([[1, 6, 3]])
        t = torch.tensor([[1, 2, 3]])
        ok, subs = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is True
        # Should have sub: 6 -> 2
        subs_list = subs[0].tolist()
        assert [6, 2] in subs_list

    def test_term_var_binds_to_constant(self):
        q = torch.tensor([[1, 2, 3]])
        t = torch.tensor([[1, 6, 3]])
        ok, subs = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is True
        subs_list = subs[0].tolist()
        assert [6, 2] in subs_list

    def test_both_vars(self):
        q = torch.tensor([[1, 6, 3]])
        t = torch.tensor([[1, 7, 3]])
        ok, subs = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is True
        subs_list = subs[0].tolist()
        assert [7, 6] in subs_list  # case3: from=t_args, to=q_args

    def test_same_var_different_bindings_fail(self):
        # Same var bound to different constants → conflict
        q = torch.tensor([[1, 6, 6]])  # var 6 appears in both positions
        t = torch.tensor([[1, 2, 3]])  # different constants
        ok, _ = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is False

    def test_same_var_same_binding_succeeds(self):
        q = torch.tensor([[1, 6, 6]])
        t = torch.tensor([[1, 2, 2]])
        ok, subs = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is True

    def test_batch(self):
        q = torch.tensor([[1, 6, 3], [2, 1, 2]])
        t = torch.tensor([[1, 2, 3], [3, 1, 2]])
        ok, _ = unify_one_to_one(q, t, C_NO, PAD)
        assert ok[0].item() is True
        assert ok[1].item() is False  # pred mismatch

    def test_empty(self):
        q = torch.empty(0, 3, dtype=torch.long)
        t = torch.empty(0, 3, dtype=torch.long)
        ok, subs = unify_one_to_one(q, t, C_NO, PAD)
        assert ok.shape == (0,)
        assert subs.shape == (0, 2, 2)


class TestApplySubstitutions:
    """Test substitution application."""

    def test_single_sub(self):
        goals = torch.tensor([[[1, 6, 3], [2, 6, 4]]])  # [1, 2, 3]
        subs = torch.tensor([[[6, 2], [PAD, PAD]]])  # [1, 2, 2]
        result = apply_substitutions(goals, subs, PAD)
        assert result[0, 0, 1].item() == 2  # arg0: 6 -> 2
        assert result[0, 1, 1].item() == 2  # arg0 of atom 1: 6 -> 2
        assert result[0, 0, 0].item() == 1  # pred unchanged

    def test_two_subs(self):
        goals = torch.tensor([[[1, 6, 7]]])
        subs = torch.tensor([[[6, 2], [7, 3]]])
        result = apply_substitutions(goals, subs, PAD)
        assert result[0, 0].tolist() == [1, 2, 3]

    def test_no_match(self):
        goals = torch.tensor([[[1, 2, 3]]])
        subs = torch.tensor([[[6, 4], [PAD, PAD]]])
        result = apply_substitutions(goals, subs, PAD)
        assert result[0, 0].tolist() == [1, 2, 3]

    def test_pred_not_substituted(self):
        goals = torch.tensor([[[6, 2, 3]]])
        subs = torch.tensor([[[6, 1], [PAD, PAD]]])
        result = apply_substitutions(goals, subs, PAD)
        assert result[0, 0, 0].item() == 6  # pred stays

    def test_empty(self):
        goals = torch.empty(0, 2, 3, dtype=torch.long)
        subs = torch.empty(0, 2, 2, dtype=torch.long)
        result = apply_substitutions(goals, subs, PAD)
        assert result.shape == (0, 2, 3)
