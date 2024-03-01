def excerpt(src, tgt, top_percentile, bottom_percentile):
    """
    Only add the lines between top X% and botom Y% from the dataset

    :param float top_percentile: Percentile of dataset where collection begins
    :param float bottom_percentile: Percentile of dataset where collection stops
    """
    return False # placeholder

def top(src, tgt, percent):
    """
    Only add the top X% lines from the dataset

    :param float percent: Percentage of dataset to include
    """
    return False # placeholder

def duplicates(src, tgt):
    """
    Remove lines when source is the same as target
    """
    return src == tgt

def char_length(src, tgt, min = 0, max = float("inf")):
    """
    Removes lines outside of a certain character length

    :param int min: Minimum length (inclusive)
    :param int max: Maximum length (inclusive)
    """
    return len(src) <= min or len(src) >= max or \
           len(tgt) <= min or len(tgt) >= max

def source_target_ratio(src, tgt, min = 0, max = float("inf")):
    """
    Removes lines when the ratio (len(source) / len(target)) is outside of bounds

    :param float min: Lower bound (inclusive)
    :param float max: Upper bound (inclusive)
    """ 
    ratio = len(src) / len(tgt)
    return ratio <= min or ratio >= max

def uppercase_count_mismatch(src, tgt):
    """
    Removes lines when source and target have a different number of uppercase letters
    """
    return sum(1 for ch in src if ch.isupper()) != sum(1 for ch in tgt if ch.isupper())

def contains(src, tgt, words = []):
    """
    Removes lines that contain these words

    :param list(str) words: List of words
    """
    for w in words:
        if w in src or w in tgt:
            return True
    return False

def digits_ratio(src, tgt, max = 0.4):
    """
    Removes lines when the ratio of numerical characters to the total length of the line
    is greather than max.

    :param float max: Maximum ratio (default: 0.4)
    """
    return len([c for c in src if c.isdigit()]) / len(src) >= max or \
                len([c for c in tgt if c.isdigit()]) / len(tgt) >= max

def nonalphanum_ratio(src, tgt, max = 0.4):
    """
    Removes lines when the ratio of non-alphanumeric characters to the total length of the line
    is greather than max.

    :param float max: Maximum ratio (default: 0.4)
    """
    return len([c for c in src if c != ' ' and (not c.isalnum())]) / len(src) >= max or \
                len([c for c in tgt if c != ' ' and (not c.isalnum())]) / len(tgt) >= max

def digits_mismatch(src, tgt):
    """
    Removes lines when there are digits in source and not in target, or vice-versa
    """
    s = sum(int(num) for num in src if num.isdecimal())
    t = sum(int(num) for num in tgt if num.isdecimal())
    return (s == 0 and t > 0) or (t == 0 and s > 0)

def nonalphanum_count_mismatch(src, tgt):
    """
    Removes lines when the sum of non-alphanumeric characters (except spaces) between source and target is not the same
    """
    return sum(1 for ch in src if ch != " " and (not ch.isalnum())) != sum(1 for ch in tgt if ch != " " and (not ch.isalnum()))

def characters_count_mismatch(src, tgt, chars = '()[]?!:"“”{}'):
    """
    Removes lines when the sum of certain characters between source and target is not the same.

    :param str chars: Characters to check (default: ()[]?!:."“”{})
    """
    for ch in chars:
        if src.count(ch) != tgt.count(ch):
            return True
    return False

def first_char_mismatch(src, tgt):
    """
    Removes lines when the first character is a letter but the case is mismatched, or the first character in source is not the same as the first character in target.
    """
    if src[0].isalpha():
        if tgt[0].isalpha():
            return src[0].isupper() != tgt[0].isupper()
        else:
            return True
    elif tgt[0].isalpha():
        return True
    else:
        return src[0] != tgt[0]
    
