
#!/usr/bin/env python3
# Copyright (C) 2024 MEAE
# Licensed under the CC-BY-NC-SA4.0 licence;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Command interface for scoring and filtering training data with reference free cometkiwi-models.
===============================

optional arguments:
  -h, --help            Show this help message and exit.
  -c, --corpus          Corpus of data to score anf filter
  --fm					Language considered for source sentences while scoring and filtering.
  --to 					Language considered for translated sentences while scoring and filtering.
  --batch_size BATCH_SIZE
                        (type: int, default: 16)
  --sample_size         Allows for faster inference and better memory use. Default: 32000
  --verbose             Sets loggers to other than ERROR level. (default: False)
  -s, --score			Computes and writes the scores for each sentence pair in a txt file. Requires GPU (optimized for one card).
  -f, --filter			Filters/cuts sources and translations under a certain score threshold.
  --model MODEL         COMET model to be used. Allows to use locally download checkpoint. (type: str, default: wmt20-cometkiwi-da)
"""
import itertools
import json
import logging
import os
import shutil
import mmap

import numpy as np
import torch

from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
# from sacrebleu.utils import get_reference_files, get_source_file

from comet import download_model, load_from_checkpoint
# from comet.models.utils import split_sequence_into_sublists
# as fast as medium and same precision
torch.set_float32_matmul_precision('high')

def count_lines(file):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))


parser = ArgumentParser(description="Command for scoring MT systems.")
parser.add_argument("-c", "--corpus", type=str)
parser.add_argument("--fm", type=str)
parser.add_argument("--to", type=str) 
parser.add_argument(
    "-s", "--score", action="store_true", help="Score every sentence pair in the corpus and write down scores to file."
)
parser.add_argument(
    "-r", "--report", action="store_true", help="Report on scores and write the distribution's graph to file."
)
parser.add_argument(
    "--full_report", action="store_true", help="Lunches a full report instead of a limited histogram."
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument(
    "--verbose", action="store_false", help="Sets all loggers to info instead of ERROR level."
)
parser.add_argument(
    "-f", "--filter", type=float, help="Filters segments under the mentioned score value."
)
parser.add_argument(
    "--sample_size", type=int, default=32000, help="Allows for faster inference on large files."
)
parser.add_argument("-m",
    "--model",
    type=str,
    required=False,
    default="Unbabel/wmt22-cometkiwi-da",
    help="COMET model to be used : to load locally downloaded checkpoint, indicate path.",
)
args = parser.parse_args()

if args.fm is not None and args.to is not None and args.corpus:
    fm = args.fm
    to = args.to
    corpus = args.corpus
else:
    parser.error(f"You must specify source (--fm) and target (--to) languages, as well as a corpus (-c).")
#Define working directories and filenames

current_dir = os.path.dirname(__file__)
scores_file = f"cometkiwi-scores-{fm}{to}.txt"
corpus_dirname = f"{fm}_{to}-1.1.{corpus}"
corpus_dirname_reverse = f"{to}_{fm}-1.1.{corpus}"
corpus_dir = os.path.join(current_dir, "run", corpus_dirname)
corpus_dir_reverse = os.path.join(current_dir, "run", corpus_dirname_reverse)

# If a training set is present, it contains empty lines : suppress these yields an alignable file
def condense_tgt_train(dir):
    with open(os.path.join(dir, "tgt-train.txt"), "r", encoding="utf8") as target_train, \
	open(os.path.join(dir, "tgttrain.txt"), "w", encoding="utf8") as targettrain:
        for index, line in enumerate(target_train):
            if (index % 2) == 0:
                targettrain.write(line)

# This will regroup training and validation dataset into two aligned files
def regroup_dataset(dir):
    with open(os.path.join(dir, "src-train.txt"), "r", encoding="utf8") as source_file, \
    open(os.path.join(dir, "src-val.txt"), "r", encoding="utf8") as source_file2, \
    open(os.path.join(dir, "source.txt"), "w", encoding="utf8") as out_source:
        for index, line in enumerate(source_file):
            out_source.write(line)
        for index,line in enumerate(source_file2):
            out_source.write(line)

    with open(os.path.join(dir, "tgttrain.txt"), "r", encoding="utf8") as target_file, \
    open(os.path.join(dir, "tgt-val.txt"), "r", encoding="utf8") as target_file2, \
    open(os.path.join(dir, "target.txt"), "w", encoding="utf8") as out_target:
        for index, line in enumerate(target_file):
            out_target.write(line)
        for index, line in enumerate(target_file2):
            out_target.write(line)

# Check regrouped aligned files and return correct source and target
def check_dataset(rdir):
# When first running the filter, source and target are moved in the "uncut" subdirectory
    unc_dir = os.path.join(rdir, "uncut")
    unc_src = os.path.join(unc_dir, f"{corpus}.{fm}")
    unc_tgt = os.path.join(unc_dir, f"{corpus}.{to}")
    if os.path.isfile(unc_src) and os.path.isfile(unc_tgt):
        print('Found source and target in the "uncut" subdirectory.')
        return unc_src, unc_tgt
# Former versions labelled it the "unfiltered" subdir, so rename it and the fles within for good measure
    elif os.path.isdir(os.path.join(rdir, "unfiltered"):
        os.rename(os.path.join(rdir, "unfiltered"), unc_dir)
	os.rename(os.path.join(unc_dir, f"unfiltered.{fm}"), unc_src)
	os.rename(os.path.join(unc_dir, f"unfiltered.{to}"), unc_tgt)
        return unc_src, unc_tgt
# Otherwise, they may be grouped (when using train.py with argument --data)
    rsrc = os.path.join(rdir,"source.txt")
    rtgt = os.path.join(rdir, "target.txt")
    rdirname = os.path.basename(rdir)
    if os.path.isfile(rsrc) and os.path.isfile(rtgt):
        print(f"Found regrouped source and target in the {rdirname} corpus directory.")
        if rdirname.startswith(fm):
            print('Forward direction detected.')
            return rsrc, rtgt
        if rdirname.startswith(to):
            print('Reverse direction detected, inverting source and target.')
            return rtgt, rsrc
# or dispersed (with basic train.py) or poorly renamed in the directory
    smartcounter = 0
    files = [ f for f in os.listdir(rdir) if os.path.isfile(os.path.join(rdir,f)) ]
    print(files)
    for f in files:
        if f.endswith(f".{fm}"):
            smartcounter +=5
            lsrc = os.path.join(rdir, f)
        if f.endswith(f".{to}"):
            smartcounter +=5
            ltgt = os.path.join(rdir, f)
        if f == "src-train.txt" or f == "tgt-train.txt":
            smartcounter +=2
        if f == "src-val.txt" or f == "tgt-val.txt":
            smartcounter +=1

    if smartcounter == 10 and count_lines(lsrc) == count_lines(ltgt):
        print(f"Found aligned {os.path.basename(lsrc)} and {os.path.basename(ltgt)} files in the {rdirname} corpus directory.")
        return lsrc, ltgt               
    elif smartcounter == 6:
        print('Found training and validation sets in the corpus. Regrouping.')
        condense_tgt_train(rdir)
        regroup_dataset(rdir)
        if os.path.isfile(rsrc) and os.path.isfile(rtgt) and count_lines(rsrc) == count_lines(rtgt):
            print(f"Regrouped sources in the {rdirname} directory are aligned. Deleting former set.")
            os.unlink(os.path.join(rdir, "src-train.txt"))
            os.unlink(os.path.join(rdir, "src-val.txt"))
            os.unlink(os.path.join(rdir, "tgttrain.txt"))
            os.unlink(os.path.join(rdir, "tgt-train.txt"))
            os.unlink(os.path.join(rdir, "tgt-val.txt"))
            if rdirname.startswith(fm):
                print('Forward direction detected.')
                return rsrc, rtgt
            if rdirname.startswith(to):
                print('Reverse direction detected.')
                return rtgt, rsrc
        else:			
            print('Sources are not aligned or something went wrong, repair the dataset.')
            exit(1)				
    elif smartcounter == 4:
        print ('Found a training set only. Processsing the target and renaming.')
        os.rename(os.path.join(rdir, "src-train.txt"), rsrc)
        condense_tgt_train(rdir)
        os.rename(os.path.join(rdir, "tgttrain.txt"), rtgt)
        if count_lines(rsrc) == count_lines(rtgt):
            print(f"Source and target files in the {rdirname} directory are aligned. Deleting former target.")
            os.unlink(os.path.join(rdir, "tgt-train.txt"))
            if rdirname.startswith(fm):
                print('Forward direction detected.')
                return rsrc, rtgt
            if rdirname.startswith(to):
                print('Reverse direction detected.')
                return rtgt, rsrc
        else:
            print('Something went or was wrong. Check the dataset.')				
    elif smartcounter == 2:
         print(f"Only found validation data in the {rdirname} directory. Verify the data.")
    else:
         print(f"Cannot find unambiguous source and target in the {rdirname} directory. Clean it!")

# Look for a preprocessed corpus in direct direction...
if os.path.isdir(corpus_dir):
    print("Found a preprocessed corpus (forward direction).")
    try:
     	source, target = check_dataset(corpus_dir)
    except Exception as e:
        exit(1)		
# ... then in the reverse direction ...
elif os.path.isdir(corpus_dir_reverse):
    print("Found a preprocessed corpus (reverse direction).")
    try:
        source, target = check_dataset(corpus_dir_reverse)
    except Exception as e:
        exit(1)
    else:
        corpus_dir = corpus_dir_reverse
# ... then for downloaded corpora in the cache....
else:
    cache_fm = f"{corpus}.{fm}-{to}.{fm}"
    cache_fm_reverse = f"{corpus}.{to}-{fm}.{fm}"
    path_fm = None
    for path, directories, files in os.walk(os.path.join(current_dir, "cache")):
        for file in files:
            if file == cache_fm:
                print(f"Found {os.path.join(path, file)}, is downloaded in the stated direction.")
                path_fm = os.path.join(path, cache_fm)
                path_to = os.path.join(path, f"{corpus}.{fm}-{to}.{to}")
            elif file == cache_fm_reverse:
                print(f"Found {os.path.join(path, file)}, is downloaded in the reverse direction.")
                path_fm = os.path.join(path, cache_fm_reverse)
                path_to = os.path.join(path, f"{corpus}.{to}-{fm}.{to}")
            if path_fm is not None:
                os.makedirs(corpus_dir)
                source = os.path.join(corpus_dir, "source.txt")
                target = os.path.join(corpus_dir, "target.txt")
                shutil.copy(path_fm, source)
                shutil.copy(path_to, target)
                print(f"Found cached files and copied them to {corpus_dir}.")
            else:
                print("The corpus has neither been processed nor cached before, it should be into the cache at least.")
                exit(1)
# Now we made sure source and target contain aligned sentence pairs and are ordered in the needed direction, wherever they may find themselves out. Or the script has aborted.
# Define the file to write or read the scores
cometkiwi_scores = os.path.join(corpus_dir, scores_file)


def compute_scores() -> None:
# A few failsafes not to crash scores or in case the data is corrupt
    if os.path.isfile(cometkiwi_scores) and count_lines(cometkiwi_scores) == count_lines(source) == count_lines(target):
        print("Cometkiwi scores have already been computed, exiting the compute function.")
        exit(1)
    if not count_lines(source) == count_lines(target):
        print("Source and target unaligned, one or both files are corrupt.")
        exit(1)
    if not args.verbose:
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            logger.setLevel(logging.ERROR)
# Load model locally...
    seed_everything(1)

    if args.model.endswith(".ckpt") and os.path.exists(args.model):
        model_path = args.model
# ... or download model, but then, make sure a huggingface token is installed and the token account has acknowledged cometkiwi license.
    else:
        try:
            model_path = download_model(args.model, saving_directory=os.path.join(current_dir, "cache", "wmt22-cometkiwi-da"))
        except Exception as e:
            print("A huggingface token should be installed, and the model's license acknowledged before downloading.")
            print(e)
    model = load_from_checkpoint(model_path)
    model.eval()
    model.set_embedding_cache()
    print(f"Loaded model from {args.model}.")
# Initialize counters 
    line_count = count_lines(source)
    line_no = 0
    scores_no = 0
    empty_lines = 0
    sample_size = args.sample_size
# Open files as memory maps and parses
    with open(source, "r+b") as sfp, \
    open(target, "r+b") as tfp, \
    open(cometkiwi_scores, "w", encoding="utf-8") as scfp:
        
        src_mm = mmap.mmap(sfp.fileno(), 0)
        tgt_mm = mmap.mmap(tfp.fileno(), 0)
        src_it = iter(src_mm.readline, b"")
        tgt_it = iter(tgt_mm.readline, b"")
        seg_scores, sys_scores, errors = [], [], []
        src_sample, tgt_sample = [], []
        print(f"Parsing {line_count} sentence pairs.")
        progress_bar = tqdm(total=line_count)
# Load sentence pairs in lists       
        for src_line in src_it:
            line_s = src_line.decode("utf-8").strip()
            line_t = next(tgt_it).decode("utf-8").strip()
            line_no +=1
            progress_bar.update(1)
# Skip empty lines
            if len(line_s) == 0 or len(line_t) == 0:
                empty_lines +=1
                continue
            else:
                src_sample.append(line_s)
                tgt_sample.append(line_t)
#When sample is full or at end of files calculates the scores
            if line_no % sample_size == 0 or line_no == line_count:
                data = {"src": src_sample, "mt": tgt_sample}
                seg_scores, sys_scores, errors = [], [], []
                new_data = []
                sys_data = {k: v for k, v in data.items()}
                sys_data = [dict(zip(sys_data, t)) for t in zip(*sys_data.values())]
                new_data.append(np.array(sys_data))
                outputs = model.predict(
                    samples=sys_data,
                    batch_size=args.batch_size,
                    gpus=1,
                    progress_bar=False,
                    accelerator="auto",
                    num_workers=None,
                    length_batching=True,
                )
                seg_scores.append(outputs.scores)
# Write the scores to file and resets all three lists
                for s in seg_scores:
                    for i in range(len(s)):
                        scfp.write(str(s[i]) + "\n")
                        scores_no +=1						
                seg_scores, src_sample, tgt_sample = [], [], []

        progress_bar.close()
# report the number of empty lines and scores calculated
        print(f"Scored {scores_no} sentence pairs, found {empty_lines} empty ones among {line_no} parsed.")


# Reports the score distribution values
def report_scores() -> None:
    score_list = np.loadtxt(cometkiwi_scores)
    scores_no = count_lines(cometkiwi_scores)
# Calculates the median
    median_score = round(np.median(score_list), 3)
# A full report will be useful if the distribution contains unusual values
    if args.full_report:
        bin_count = 1000
        score_histogram, score_values = np.histogram(score_list, bins=1000, range=(0, 1), density=False)
    else:
        bin_count = 130
        score_histogram, score_values = np.histogram(score_list, bins=130, range=(0.8, 0.93), density=False)
    median_over = 0

# For further purposes, calculates the number of sentence pairs over the median approximate value
    for k in range (0, bin_count):
        if score_values[k] < median_score:
            continue
        else:
            median_over = median_over + score_histogram[k]
    print(f"The median score is {median_score}, {median_over} sentences pairs score over the median.")
#Look for the score value matching the distribution's peak.
    peak_hist = max(i for i in score_histogram)
    x=0
    score_over = 0
    part_over = []
    for i in score_histogram:
        if score_histogram[x] < peak_hist:
            x +=1
        else:
            break
# Compute the number of sentence pairs scoring over this value
    for j in range (x, bin_count):
        score_over = score_over + score_histogram[j]
# Then print the results
    print(f"The distribution peaks at value {round(score_values[x], 3)}, {score_over} sentence pairs score over this value.")
    print(f"Job done!")
# Try import matplotlib and plot a graph to visually check the distribution's regularity
    try:
        import matplotlib.pyplot as plt
        if args.full_report:
            plot_file = f"cometkiwi-fulldist-{args.corpus}-{args.fm}-{args.to}.png"
            plt.hist(score_list, bins=1000, range=(0,1), density=False)            
        else:
            plot_file = f"cometkiwi-dist-{args.corpus}-{args.fm}-{args.to}.png"
            plt.hist(score_list, bins=130, range=(0.8,0.93), density=False)
        plt.title (f"Score distribution for {args.corpus} from {args.fm} to {args.to}")
        plot = os.path.join(corpus_dir, plot_file)
        plt.savefig(plot)
        print(f"Graph saved in corpus directory. Job done!")
    except Exception as e:
        print(f"Matplotlib module absent.")
        print(e)


# Filter the sentence pairs over a specified threshold (usually, distribution median or peak)
def cut_over(threshold) -> None:
# First, move the original source and target to the "uncut" subdirectory (otherwise, train.py uses it as data)
    unc_dir = os.path.join(corpus_dir, "uncut")
    unc_src = os.path.join(unc_dir, f"{corpus}.{fm}")
    unc_tgt = os.path.join(unc_dir, f"{corpus}.{to}")
    scores = scores_file #Dummy allocation (scores should be a path, not a str) serves an if branch later on
    if not os.path.isdir(unc_dir):
        os.makedirs(unc_dir)
    try:
        os.rename(source, unc_src)
        os.rename(target, unc_tgt)
    except Exception as e:
        print(e)
    cut_src = os.path.join(corpus_dir, f"{fm}{to}_{threshold}_{corpus}.{fm}")
    cut_tgt = os.path.join(corpus_dir, f"{fm}{to}_{threshold}_{corpus}.{to}")
# Moves the formerly filtered files (source and target must be unique for further training) and initializes score source, or launches score computing
    files = [ f for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir,f)) ]
    for f in files:
	    # Move formerly filtered data into a proper subdirectory 
        if f.startswith(f"{fm}{to}_"):
            drt_dir = os.path.join(corpus_dir, f"{fm}{to}")
            if not os.path.isdir(drt_dir):
                os.makedirs(drt_dir)
            move_to = os.path.join(drt_dir, f)
            os.rename(os.path.join(corpus_dir, f), move_to)
        if f.startswith(f"{to}{fm}_"):
            rev_dir = os.path.join(corpus_dir, f"{to}{fm}")
            if not os.path.isdir(rev_dir):
                os.makedirs(rev_dir)      
            move_to = os.path.join(rev_dir, f)
            os.rename(os.path.join(corpus_dir, f), move_to)
        if f == scores_file:
            scores = cometkiwi_scores #Real allocation for the variable
    if scores == scores_file: #if not allocated properly in the former loop, then no scores have been calculated yet
            print(f"There has to be a scores list to filter. Proceeding to compute the scores now.")
            compute_scores()
            report_scores()
# Initialize counters 
    line_count = count_lines(scores)
    line_no = 0
    included_lines = 0
    filtered_lines = 0
    print(f"Filtering sentence pairs scoring above {args.filter}.")
# Open all files
    with open(scores, "r+b") as cfp, \
        open(unc_src, "r+b") as sfp, \
        open(unc_tgt, "r+b") as tfp, \
        open(cut_src, "w", encoding="utf-8") as csfp, \
        open(cut_tgt, "w", encoding="utf-8") as ctfp:
# Read input as memory maps
        sco_mm = mmap.mmap(cfp.fileno(), 0)
        src_mm = mmap.mmap(sfp.fileno(), 0)
        tgt_mm = mmap.mmap(tfp.fileno(), 0)
        sco_it = iter(sco_mm.readline, b"")
        src_it = iter(src_mm.readline, b"")
        tgt_it = iter(tgt_mm.readline, b"")
		
        progress_bar = tqdm(total=line_count)
#Then parse inputs
        for sco_line in sco_it:
            line_c = sco_line.decode("utf-8").strip()
            line_s = next(src_it).decode("utf-8").strip()	
            line_t = next(tgt_it).decode("utf-8").strip()
#Skip empty lines in source or target	
            while len(line_s) == 0 or len(line_t) == 0:
                line_s = next(src_it).decode("utf-8").strip()	
                line_t = next(tgt_it).decode("utf-8").strip()
            line_no +=1
# Read score and decide whether to write the sentence pair or not		
            score = float(line_c)
            progress_bar.update(1)

            if score >= args.filter:
                csfp.write(line_s + "\n")
                ctfp.write(line_t + "\n")
                included_lines +=1
            else:
                cut_lines +=1

        progress_bar.close()

    print(f"Parsed {line_no} pairs, wrote {included_lines} in source and target files, filtered {cut_lines} over {line_count}.") 
    print(f"Job done!")    




if __name__ == "__main__":
    if (args.report or args.full_report) and not args.score:
        report_scores()
    if args.score:
        compute_scores()
        report_scores()	
    if args.filter is not None:
        threshold_value = args.filter
        cut_over(threshold_value)		
