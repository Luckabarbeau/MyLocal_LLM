"""SiLU activation for inference (no cache needed)."""


from mini_llm.backend import xp


def silu(x):
    """
    SiLU activation: x * sigmoid(x).
    
    Numerically stable implementation.
    """
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    result = xp.zeros_like(x)
    
    pos_x = x[pos_mask]
    if pos_x.size > 0:
        result[pos_mask] = pos_x / (1.0 + xp.exp(-pos_x))
    
    neg_x = x[neg_mask]
    if neg_x.size > 0:
        exp_neg_x = xp.exp(neg_x)
        result[neg_mask] = neg_x * exp_neg_x / (1.0 + exp_neg_x)
    
    return result
