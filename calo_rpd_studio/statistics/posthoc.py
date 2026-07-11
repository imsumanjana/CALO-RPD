"""Holm step-down p-value correction."""
def holm_correction(p_values):
    items = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(items)
    out = [0.0] * m
    running = 0.0
    for rank, (index, p_value) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p_value))
        out[index] = running
    return out
