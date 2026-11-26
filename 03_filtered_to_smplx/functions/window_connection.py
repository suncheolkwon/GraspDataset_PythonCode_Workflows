def windows_for_segment(lo, hi, size=128, overlap=16):
    if hi <= lo:
        return []
    end = min(hi, lo + size)
    out = [(lo, end, lo)]
    while end < hi:
        start = max(lo, end - overlap)
        newend = min(hi, start + size)
        out.append((start, newend, end))
        end = newend
    return out
