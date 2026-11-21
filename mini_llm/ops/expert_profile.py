"""Bounded, deterministic placement from observed routing, not predicted output."""
import math
from collections import Counter
from collections.abc import Mapping
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


def _validated_counts(counts,*,model_id=None):
    """Snapshot caller-owned routing counts without changing runtime heat."""
    if not isinstance(counts,Mapping):
        raise TypeError('Profile counts must be a mapping')
    result=dict(counts)
    for key,count in result.items():
        if not isinstance(key,ExpertKey):
            raise TypeError('Profile keys must be ExpertKey')
        if model_id is not None and key.model_id!=model_id:
            raise ValueError('Profile model identity does not match executor')
        if (not isinstance(key.model_id,str) or not key.model_id or any(
                isinstance(v,bool) or not isinstance(v,Integral) or v<0
                for v in (key.layer_id,key.expert_id))):
            raise ValueError('Invalid profile ExpertKey identity')
        if (isinstance(count,bool) or not isinstance(count,Real) or count<0 or
                (not isinstance(count,Integral) and not math.isfinite(count))):
            raise ValueError('Profile frequencies must be finite and nonnegative')
        # NumPy unsigned negation wraps instead of producing a ranking key.
        if isinstance(count,Integral):
            result[key]=int(count)
    return result


def sized_frequency_placement(counts,bundle_sizes,slot_sizes,*,resident=()):
    """Pure frequency ranking within the existing arena's class capacities.

    Sizes describe complete gate/up/down bundles, not individual projections.
    A bundle belongs only to its smallest fitting 256-aligned slot class; spare
    larger slots do not change that class's capacity. Equal counts prefer an
    existing resident, then (model,layer,expert). No byte storage is copied.
    """
    counts=_validated_counts(counts)
    if not isinstance(bundle_sizes,Mapping):
        raise TypeError('Complete bundle sizes must be a mapping')
    slots=tuple(slot_sizes)
    if not slots or any(isinstance(s,bool) or not isinstance(s,Integral) or
                        s<=0 or s%256 for s in slots):
        raise ValueError('Slot sizes must be positive 256-byte aligned integers')
    for key,size in bundle_sizes.items():
        if not isinstance(key,ExpertKey):
            raise TypeError('Bundle size keys must be ExpertKey')
        if isinstance(size,bool) or not isinstance(size,Integral) or size<=0:
            raise ValueError('Complete bundle sizes must be positive integers')
    capacities=Counter(slots)
    classes=sorted(capacities)
    candidates={s:[] for s in classes}
    for key,count in counts.items():
        if count<=0:
            continue
        if key not in bundle_sizes:
            raise ValueError(f'{key}: missing complete bundle size')
        aligned=(int(bundle_sizes[key])+255)//256*256
        size=next((s for s in classes if aligned<=s),None)
        if size is None:
            raise ValueError(f'{key}: complete bundle exceeds approved slot classes')
        candidates[size].append(key)
    resident=frozenset(resident)
    return tuple(key for size in classes for key in sorted(candidates[size],
        key=lambda k:(-counts[k],k not in resident,k.model_id,k.layer_id,k.expert_id)
    )[:capacities[size]])
