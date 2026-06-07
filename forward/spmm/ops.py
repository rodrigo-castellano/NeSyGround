"""SpMM rule classification: op codes and per-rule descriptors.

Each compiled rule maps to one ``SpMMOp`` value plus a ``SpMMRuleDesc``
describing how to evaluate it as one or more sparse-matrix products.

Op table:

  =========== ====================================================
  ``COPY``       1-body: ``h(X,Y) :- b(X,Y)``  (direct copy)
  ``TRANSPOSE``  1-body: ``h(X,Y) :- b(Y,X)``  (matrix transpose)
  ``MATMUL``     2-body chain: sparse ``A @ B``
  ``ELEM_AND``   2-body, both bodies bind same head vars
  ``CASE_A``     2-body, body1 vars all derivable from body0
  ``EXIST_AND``  2-body, body1 has 1 head var + 1 existential
  ``MATMUL3``    3-body chain: ``A @ B @ C``
  ``UNSUPPORTED`` 4+ body or unhandled binding shape — caller falls
                  back to FCDynamic.
  =========== ====================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


__all__ = [
    "SpMMOp",
    "SpMMRuleDesc",
    "classify_rule",
    "slots_for_op",
    "slot_pred",
]


class SpMMOp(Enum):
    COPY = auto()
    TRANSPOSE = auto()
    MATMUL = auto()
    ELEM_AND = auto()
    CASE_A = auto()
    EXIST_AND = auto()
    MATMUL3 = auto()
    UNSUPPORTED = auto()


@dataclass
class SpMMRuleDesc:
    """How to compute one rule's provable atoms via SpMM."""
    op: SpMMOp
    head_pred: int
    pred_A: int = -1
    pred_B: int = -1
    pred_C: int = -1                # 3-body chain only
    transpose_A: bool = False
    transpose_B: bool = False
    transpose_C: bool = False        # 3-body chain only
    transpose_result: bool = False
    reflexive_body0: bool = False
    reflexive_body1: bool = False
    reflexive_body2: bool = False    # 3-body chain only
    b10_binding: int = -1
    b11_binding: int = -1
    b00: int = -1
    b01: int = -1
    exist_body1_head_arg: int = -1
    exist_filter_body0_arg: int = -1


def classify_rule(cr) -> SpMMRuleDesc:
    """Classify a ``RulePattern`` into a ``SpMMOp`` + descriptor.

    Uses the rule's variable names (not just binding indices) so 2-body
    rules can be categorised by their *shared variable* pattern.
    """
    head_pred = cr.head_pred_idx
    hv0 = cr.head_var0
    hv1 = cr.head_var1

    if cr.num_body == 1:
        bp = cr.body_patterns[0]
        b0, b1 = bp["arg0_binding"], bp["arg1_binding"]
        pred_A = bp["pred_idx"]
        reflexive = (b0 == b1)
        if b0 == 0 and b1 == 1:
            return SpMMRuleDesc(op=SpMMOp.COPY, head_pred=head_pred,
                                pred_A=pred_A, reflexive_body0=reflexive)
        if b0 == 1 and b1 == 0:
            return SpMMRuleDesc(op=SpMMOp.TRANSPOSE, head_pred=head_pred,
                                pred_A=pred_A, reflexive_body0=reflexive)
        return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)

    if cr.num_body == 2:
        bp0, bp1 = cr.body_patterns[0], cr.body_patterns[1]
        b00, b01 = bp0["arg0_binding"], bp0["arg1_binding"]
        b10, b11 = bp1["arg0_binding"], bp1["arg1_binding"]
        pred0, pred1 = bp0["pred_idx"], bp1["pred_idx"]
        v00, v01 = bp0["arg0_var"], bp0["arg1_var"]
        v10, v11 = bp1["arg0_var"], bp1["arg1_var"]
        reflexive0 = (v00 == v01)
        reflexive1 = (v10 == v11)
        b0_vars = {v00, v01}
        b1_vars = {v10, v11}
        shared_vars = (b0_vars & b1_vars) - {hv0, hv1}

        b1_from_b0 = b1_vars.issubset(b0_vars)
        if b1_from_b0:
            return SpMMRuleDesc(
                op=SpMMOp.CASE_A, head_pred=head_pred,
                pred_A=pred0, pred_B=pred1,
                reflexive_body0=reflexive0, reflexive_body1=reflexive1,
                b10_binding=b10, b11_binding=b11,
                b00=b00, b01=b01)

        if (b00 in (0, 1) and b01 in (0, 1) and b10 in (0, 1) and b11 in (0, 1)
                and b00 == b10 and b01 == b11):
            return SpMMRuleDesc(
                op=SpMMOp.ELEM_AND, head_pred=head_pred,
                pred_A=pred0, pred_B=pred1,
                reflexive_body0=reflexive0, reflexive_body1=reflexive1)

        b0_head_vars = b0_vars & {hv0, hv1}
        b1_head_vars = b1_vars & {hv0, hv1}
        b1_exist_vars = b1_vars - {hv0, hv1}
        if (b0_head_vars == b0_vars and len(b1_head_vars) == 1
                and len(b1_exist_vars) == 1):
            shared_head_var = next(iter(b1_head_vars))
            exist_b1_arg = 0 if v10 == shared_head_var else 1
            exist_filter_b0 = 0 if v00 == shared_head_var else 1
            if v00 == hv0 and v01 == hv1:
                t_R = False
            elif v00 == hv1 and v01 == hv0:
                t_R = True
            else:
                return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
            return SpMMRuleDesc(
                op=SpMMOp.EXIST_AND, head_pred=head_pred,
                pred_A=pred0, pred_B=pred1,
                transpose_result=t_R,
                reflexive_body0=reflexive0, reflexive_body1=reflexive1,
                exist_body1_head_arg=exist_b1_arg,
                exist_filter_body0_arg=exist_filter_b0)

        if len(shared_vars) != 1:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        shared = next(iter(shared_vars))
        shared_is_a0 = (v00 == shared)
        shared_is_a1 = (v01 == shared)
        shared_is_c0 = (v10 == shared)
        shared_is_c1 = (v11 == shared)
        if not (shared_is_a0 or shared_is_a1):
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        if not (shared_is_c0 or shared_is_c1):
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)

        t_A = shared_is_a0
        t_B = shared_is_c1
        non_shared_a_var = v01 if shared_is_a0 else v00
        non_shared_c_var = v10 if shared_is_c1 else v11
        if non_shared_a_var == hv0 and non_shared_c_var == hv1:
            t_R = False
        elif non_shared_a_var == hv1 and non_shared_c_var == hv0:
            t_R = True
        else:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        return SpMMRuleDesc(
            op=SpMMOp.MATMUL, head_pred=head_pred,
            pred_A=pred0, pred_B=pred1,
            transpose_A=t_A, transpose_B=t_B,
            transpose_result=t_R,
            reflexive_body0=reflexive0, reflexive_body1=reflexive1)

    if cr.num_body == 3:
        # 3-body chain bp0(X,Y), bp1(Y,K), bp2(K,Z) → head(X,Z).
        bp0, bp1, bp2 = cr.body_patterns[0], cr.body_patterns[1], cr.body_patterns[2]
        v00, v01 = bp0["arg0_var"], bp0["arg1_var"]
        v10, v11 = bp1["arg0_var"], bp1["arg1_var"]
        v20, v21 = bp2["arg0_var"], bp2["arg1_var"]
        pred0, pred1, pred2 = bp0["pred_idx"], bp1["pred_idx"], bp2["pred_idx"]
        b0v = {v00, v01}; b1v = {v10, v11}; b2v = {v20, v21}
        Y_set = (b0v & b1v) - {hv0, hv1}
        K_set = (b1v & b2v) - {hv0, hv1}
        if len(Y_set) != 1 or len(K_set) != 1:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        Y = next(iter(Y_set)); K = next(iter(K_set))
        if Y == K:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        if b1v != {Y, K}:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        ns_a = (b0v - {Y}).pop() if len(b0v - {Y}) == 1 else None
        ns_c = (b2v - {K}).pop() if len(b2v - {K}) == 1 else None
        if ns_a is None or ns_c is None:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        if {ns_a, ns_c} != {hv0, hv1}:
            return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)
        t_A = (v00 == Y)
        t_B = (v10 == K)
        t_C = (v20 == ns_c)
        t_R = (ns_a == hv1)
        return SpMMRuleDesc(
            op=SpMMOp.MATMUL3, head_pred=head_pred,
            pred_A=pred0, pred_B=pred1, pred_C=pred2,
            transpose_A=t_A, transpose_B=t_B, transpose_C=t_C,
            transpose_result=t_R,
            reflexive_body0=(v00 == v01),
            reflexive_body1=(v10 == v11),
            reflexive_body2=(v20 == v21))

    return SpMMRuleDesc(op=SpMMOp.UNSUPPORTED, head_pred=head_pred)


def slots_for_op(op: SpMMOp) -> int:
    """How many body slots a SpMM op consumes."""
    if op in (SpMMOp.COPY, SpMMOp.TRANSPOSE):
        return 1
    if op in (SpMMOp.MATMUL, SpMMOp.ELEM_AND,
              SpMMOp.CASE_A, SpMMOp.EXIST_AND):
        return 2
    if op == SpMMOp.MATMUL3:
        return 3
    return 0


def slot_pred(desc: SpMMRuleDesc, slot: int) -> int:
    """Predicate at the given body slot of a rule."""
    if slot == 0:
        return desc.pred_A
    if slot == 1:
        return desc.pred_B
    if slot == 2:
        return desc.pred_C
    return -1
