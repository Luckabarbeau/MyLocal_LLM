"""Header-only deterministic size-class capacity plan; no weight payload reads."""
from collections import Counter
from numbers import Integral


def sized_slot_layout(budget_bytes,layer_bytes,experts_per_layer):
    """Use approximately equal resident fractions across checkpoint size classes.

    Routing heat still decides which experts occupy each class. Every resulting
    slot is fixed and 256-byte aligned; total physical storage never exceeds
    budget. This is a baseline layout, not a claim of optimal heat allocation.
    """
    values=(budget_bytes,experts_per_layer,*layer_bytes)
    if any(isinstance(v,bool) or not isinstance(v,Integral) or v<=0 for v in values):
        raise ValueError('Slot budgets, sizes and expert count must be positive integers')
    if not layer_bytes: raise ValueError('At least one layer is required')
    groups=Counter((int(n)+255)//256*256 for n in layer_bytes)
    limits={size:count*int(experts_per_layer) for size,count in groups.items()}
    total=sum(size*count for size,count in limits.items())
    budget=min(int(budget_bytes),total)
    slots={size:min(count,budget*count//total) for size,count in limits.items()}
    used=sum(size*count for size,count in slots.items())
    while True:
        candidates=[size for size in slots if size<=budget-used and slots[size]<limits[size]]
        if not candidates: break
        size=max(candidates,key=lambda s:(budget*limits[s]-slots[s]*total,-s))
        slots[size]+=1; used+=size
    return tuple(size for size in sorted(slots) for _ in range(slots[size]))
