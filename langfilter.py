import itertools
import logging
import os
import mmap
from tqdm import tqdm
import langdetect
from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr

langdetect.DetectorFactory.seed = 0

parser = ArgumentParser(description="Command for scoring MT systems.")
parser.add_argument("-c", "--corpus", type=str)
parser.add_argument("--fm", type=str)
parser.add_argument("--to", type=str)
args = parser.parse_args()

if args.fm is not None and args.to is not None and args.corpus is not None:
    fm = args.fm
    to = args.to
    corpus = args.corpus
else:
    parser.error(f"You must specify source (--fm) and target (--to) languages, as well as a directory path (-c) containing a corpus.")

lang_filtered = os.path.join(corpus, "lang-filtered")

def count_lines(file):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def langfilter() -> None:
    filelist = [ f for f in os.listdir(corpus) if os.path.isfile(os.path.join(corpus, f)) ]
	
    for f in filelist:
        if f == "source.txt" or f.endswith(fm):
            source = os.path.join(corpus, f)
        if f == "target.txt" or f.endswith(to):
            target = os.path.join(corpus, f)
    if source is None or target is None:
        print("The source or the target is missing in the corpus directory.")
        exit(1)

    if not os.path.isdir(lang_filtered):
        os.makedirs(lang_filtered, exist_ok=True)

    filtered_to = os.path.join(lang_filtered, f"filtered.{to}")
    filtered_fm = os.path.join(lang_filtered, f"filtered.{fm}")
    garbage_fm = os.path.join(lang_filtered, f"garbage_{fm}.txt")
    garbage_to = os.path.join(lang_filtered, f"garbage_{to}.txt")
    languages = os.path.join(lang_filtered, "_languages_in_garbage.txt")
    probs_fm = os.path.join(lang_filtered, f"probs_{fm}.txt")
    probs_to = os.path.join(lang_filtered, f"probs_{to}.txt")
    line_count = count_lines(source)
    line_no = 0
    garbage_pairs = 0

    with open(source, "r+b") as sfp, \
    open(target, "r+b") as tfp, \
    open(garbage_fm, "w", encoding="utf-8") as _sfp, \
    open(garbage_to, "w", encoding="utf-8") as _tfp, \
    open(probs_fm, "w", encoding="utf-8") as psfp, \
    open(probs_to, "w", encoding="utf-8") as ptfp, \
    open(filtered_fm, "w", encoding="utf-8") as gsfp, \
    open(filtered_to, "w", encoding="utf-8") as gtfp,\
    open(languages, "w", encoding="utf-8") as lafp:

        src_mm = mmap.mmap(sfp.fileno(), 0)
        tgt_mm = mmap.mmap(tfp.fileno(), 0)
        src_it = iter(src_mm.readline, b"")
        tgt_it = iter(tgt_mm.readline, b"")
        progress_bar = tqdm(total=line_count)

        for src_line in src_it:
            line_s = src_line.decode("utf-8").strip()
            line_t = next(tgt_it).decode("utf-8").strip()
            line_no +=1
            progress_bar.update(1)
            try:
                lang_s = langdetect.detect(line_s)
            except Exception:
                lang_s = "unknown"
            try:
                lang_t = langdetect.detect(line_t)
            except Exception:
                lang_t = "unknown"
				
            if lang_s == fm and lang_t == to:
                gsfp.write(line_s + "\n")
                gtfp.write(line_t + "\n")
            else:				
                _sfp.write(line_s + "\n")
                _tfp.write(line_t + "\n")
                lafp.write("from " + lang_s + " to " + lang_t + "\n")
                garbage_pairs +=1
                if lang_s == "unknown":
                    psfp.write("Unknown language found")
                else:
                    source_probs = langdetect.detect_langs(line_s)
                    psfp.write(str(source_probs) + "\n")
                if lang_t == "unknown":
                    ptfp.write("Unknown language found")
                else:
                    target_probs = langdetect.detect_langs(line_t)
                    ptfp.write(str(target_probs) + "\n")		
				
        progress_bar.close()

    print(f"Found {garbage_pairs} lines of others languages among {line_no} as a whole.")

if __name__ == "__main__":
    langfilter()
    raw_fm = os.path.join(lang_filtered, f"raw_corpus.{fm}")
    os.rename(source, raw_fm)
    os.rename(filtered_fm, source)
    raw_to = os.path.join(lang_filtered, f"raw_corpus.{to}")
    os.rename(target, raw_to)
    os.rename(filtered_to, target)