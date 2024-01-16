
def single_word_punctuation(src, tgt, chars = "?!."):
    """
    Adds punctuation to single words if it's missing and removes it if it's present.

    :param str chars: Punctuation characters (default: ?!.)
    """
    if src.count(" ") != 0 or tgt.count(" ") != 0:
        return []
    
    out = []
    for ch in chars:
        if src[-1] == ch and tgt[-1] == ch:
            out.append((src[:-1], tgt[:-1]))
    if src[-1] not in chars and src[-1].isalnum() and tgt[-1] not in chars and tgt[-1] not in chars:
        for ch in chars:
            out.append((src + ch, tgt + ch))

    return out

def lowercase(src, tgt):
    """
    The same sentences, all lowercased
    """
    low = (src.lower(), tgt.lower())
    if low[0] != src and low[1] != tgt:
        return [(src.lower(), tgt.lower())]
    else:
        return []