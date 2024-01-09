
def _remove_unpaired_quotes_and_brackets(line):
    # Deleting unpaired quotation marks
    if (line.count('“') + line.count('”')) % 2 != 0:
        line = line.replace('“', '')
        line = line.replace('”', '')
    if (line.count('«') + line.count('»')) % 2 != 0:
        line = line.replace('«', '')
        line = line.replace('»', '')
    while line.count('"') % 2 != 0:
        line = line.replace('"', '', 1)
    # Deleting unpaired square brackets
    while line.count('[') != line.count(']'):
        if '[' in line:
            line = line.replace('[', '', 1)
        if ']' in line:
            line = line.replace(']', '', 1)
    # Deleting unpaired parentheses
    while line.count('(') != line.count(')'):
        if '(' in line:
            line = line.replace('(', '', 1)
        if ')' in line:
            line = line.replace(')', '', 1)
    # Deleting unpaired curly braces
    while line.count('{') != line.count('}'):
        if '{' in line:
            line = line.replace('{', '', 1)
        if '}' in line:
            line = line.replace('}', '', 1)
    return line

def remove_unpaired_quotes_and_brackets(src, tgt):
    """
    Removes unmatched quotations (“, ”, ", «, ») and parentheses ((, ), [, ], {, }) 
    """
    return _remove_unpaired_quotes_and_brackets(src), _remove_unpaired_quotes_and_brackets(tgt)

def remove_chars(src, tgt, chars = []):
    """
    Remove these characters or words

    :param list(str)|str chars: List of characters or words
    """
    for c in chars:
        src = src.replace(c, '')
        tgt = tgt.replace(c, '')
    
    return src, tgt

def first_case_normalize(src, tgt):
    """
    Normalize the case of the first letter
    """
    if src[0].isalpha() and tgt[0].isalpha():
        if src[0].isupper() and not tgt[0].isupper():
            tgt = tgt[0].upper() + tgt[1:]
        elif src[0].islower() and not tgt[0].islower():
            tgt = tgt[0].lower() + tgt[1:]
    
    return src, tgt