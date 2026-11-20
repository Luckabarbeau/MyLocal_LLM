"""Pure bounded admission proposals for equal-stride expert slots.

Scores must come from already observed routes. This module neither executes nor
moves weights. The caller must fence consumers, enforce transfer budgets and
publish complete bundles. Size-classed slots require a separate compatibility
filter; these global equal-slot proposals must not be applied to them blindly.
"""
import math
import numpy as np


def propose_heat_moves(heat, resident, *, limit, hysteresis=1.25, margin=1.0):
    if (not isinstance(heat,np.ndarray) or heat.ndim!=2 or
            not 1<=heat.shape[0]<=64 or not 1<=heat.shape[1]<=512 or
            heat.dtype.kind not in 'fiu' or not np.isfinite(heat).all() or (heat<0).any()):
        raise ValueError('Heat must be bounded finite nonnegative [layer,expert] scores')
    if not isinstance(resident,np.ndarray) or resident.dtype!=np.bool_ or resident.shape!=heat.shape:
        raise ValueError('Residency must be a matching boolean table')
    if isinstance(limit,bool) or not isinstance(limit,int) or not 0<=limit<=256:
        raise ValueError('Movement limit must be an integer in0..256')
    if not math.isfinite(hysteresis) or hysteresis<1 or not math.isfinite(margin) or margin<0:
        raise ValueError('Require finite hysteresis>=1 and nonnegative margin')
    if not limit: return []
    scores=heat.astype(np.float64,copy=False).reshape(-1)
    mask=resident.reshape(-1)
    candidates=np.flatnonzero(~mask)
    victims=np.flatnonzero(mask)
    candidates=candidates[np.lexsort((candidates,-scores[candidates]))]
    victims=victims[np.lexsort((victims,scores[victims]))]
    width=heat.shape[1]
    result=[]
    for hot,cold in zip(candidates[:limit],victims[:limit]):
        if scores[hot]<=hysteresis*scores[cold]+margin: break
        result.append(((int(hot)//width,int(hot)%width),(int(cold)//width,int(cold)%width)))
    return result
