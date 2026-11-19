"""Bounded, deterministic placement from observed routing, not predicted output."""
import math
from numbers import Integral,Real
from .expert_residency import ExpertKey


def frequency_placement(counts,budget_bytes,stride_bytes):
    """Choose fixed-stride slots by cumulative observed frequency.

    Ties use (model,layer,expert), not dict insertion order. This deliberately
    does not claim to predict a new conversation's routing distribution.
    """
    for name,value in (('budget_bytes',budget_bytes),('stride_bytes',stride_bytes)):
        if isinstance(value,bool) or not isinstance(value,Integral) or value<(1 if name=='stride_bytes' else 0):
            raise ValueError(f'{name} must be a nonnegative integer (stride positive)')
    for key,count in counts.items():
        if not isinstance(key,ExpertKey):
            raise TypeError('Profile keys must be ExpertKey')
        if isinstance(count,bool) or not isinstance(count,Real) or not math.isfinite(count) or count<0:
            raise ValueError('Profile frequencies must be finite and nonnegative')
    candidates=sorted((k for k,v in counts.items() if v>0),
        key=lambda k:(-counts[k],k.model_id,k.layer_id,k.expert_id))
    return tuple(candidates[:int(budget_bytes//stride_bytes)])
