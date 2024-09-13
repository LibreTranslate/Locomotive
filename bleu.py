import os
import json
import argparse
from re import T
from sacrebleu import corpus_bleu

parser = argparse.ArgumentParser(description='Compute BLEU on translated files')
parser.add_argument("-t", "--translation", type=str)
parser.add_argument("-r", "--reference", type=str)

ts = []
rf = []

args = parser.parse_args()
current_dir = os.path.dirname(__file__)

translated = os.path.join(current_dir, args.translation)
ts = [line.rstrip('\n') for line in open(translated, encoding="utf-8")]
    
refered = os.path.join(current_dir, args.reference)
rf = [line.rstrip('\n') for line in open(refered, encoding="utf-8")]



    
bleu_score = round(corpus_bleu(
    ts, [[x] for x in rf]
).score, 5)

print(f"BLEU score: {bleu_score}")