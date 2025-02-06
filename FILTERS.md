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

#### excerpt
<code>Selects a partial dataset located between top % and bottom % of a large dataset (useful with very large ones).</code>
* top_percentile (float) : dataset percentile where data collection begins
* bottom_percentile (float) : percentile where data collection ends

#### fast_lang
<code>Removes lines written in other languages than the ones specified for training. 
To use it, you need to download fasttext model for language identification first ; 
then, save it in the ./cache/fasttext/ directory.
</code>
* lid.176.bin from https://fasttext.cc/docs/en/language-identification.html

<mark>_Please note that fast text is distributed under CC-BY-SA 3.0 licence._</mark>

<mark>Credits: A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification</mark>

<code>@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}</code>

#### first_char_mismatch
<code>Removes lines when the first character is a letter but the case is mismatched, or the first character in source is not the same as the first character in target.</code>

#### limit_latin_chars
<code>Removes lines when the number of latin characters exceeds max. Useful with some corpora that exhibit mixed charsets in sentences when useful charset isn't "Latn".</code>
  * max (int) : Maximum count (12)

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

#### top
<code>Only add the top X% lines from the dataset
</code>
 * percent (float) : Percentage of dataset to include

#### uppercase_count_mismatch
<code>Removes lines when source and target have a different number of uppercase letters</code>

