def duplicates(src, tgt):
    return src == tgt

def char_length(src, tgt, min = 0, max = float("inf")):
    return len(src) <= min or len(src) >= max or \
           len(tgt) <= min or len(tgt) >= max

def source_target_ratio(src, tgt, min = 0, max = float("inf")):
    ratio = len(src) / len(tgt)
    return ratio <= min or ratio >= max

def contains(src, tgt, words = []):
    for w in words:
        if w in src or w in tgt:
            return True
    return False

def digits_ratio(src, tgt, max = 0.4):
    return len([c for c in src if c.isdigit()]) / len(src) >= max or \
                len([c for c in tgt if c.isdigit()]) / len(tgt) >= max

def nonalphanum_ratio(src, tgt, max = 0.4):
    return len([c for c in src if not c.isalnum()]) / len(src) >= max or \
                len([c for c in tgt if not c.isalnum()]) / len(tgt) >= max

def digits_sum_mismatch(src, tgt):
    return sum(int(num) for num in src if num.isdecimal()) != sum(int(num) for num in tgt if num.isdecimal())