## Filters

#### char_length
<code>Removes lines outside of a certain character length
</code>
 * min (int) : Minimum length (inclusive)
 * max (int) : Maximum length (inclusive)

#### characters_count_mismatch
<code>Removes lines when the sum of certain characters between source and target is not the same.
</code>
 * chars (str) : Characters to check (()[]?!:."“”{})

#### contains
<code>Removes lines that contain these words
</code>
 * words (list(str)) : List of words

#### digits_mismatch
<code>Removes lines when there are digits in source and not in target, or vice-versa</code>

#### digits_ratio
<code>Removes lines when the ratio of numerical characters to the total length of the line
is greather than max.
</code>
 * max (float) : Maximum ratio (0.4)

#### duplicates
<code>Remove lines when source is the same as target</code>

#### first_char_mismatch
<code>Removes lines when the first character is a letter but the case is mismatched, or the first character in source is not the same as the first character in target.</code>

#### nonalphanum_count_mismatch
<code>Removes lines when the sum of non-alphanumeric characters (except spaces) between source and target is not the same</code>

#### nonalphanum_ratio
<code>Removes lines when the ratio of non-alphanumeric characters to the total length of the line
is greather than max.
</code>
 * max (float) : Maximum ratio (0.4)

#### source_target_ratio
<code>Removes lines when the ratio (len(source) / len(target)) is outside of bounds
</code>
 * min (float) : Lower bound (inclusive)
 * max (float) : Upper bound (inclusive)

#### uppercase_count_mismatch
<code>Removes lines when source and target have a different number of uppercase letters</code>

