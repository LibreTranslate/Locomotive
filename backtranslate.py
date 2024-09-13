import os
import json
import argparse
import ctranslate2
from tqdm import tqdm
import sentencepiece
from tokenizer import BPETokenizer, SentencePieceTokenizer

parser = argparse.ArgumentParser(description='Backtranslate with an argos-translate model')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--source',
    type=str,
    help='Path to source file.')
parser.add_argument('--reverse',
    action='store_true',
    help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
parser.add_argument('--cpu',
    action="store_true",
    help='Force CPU use. Default: %(default)s')
parser.add_argument('--chunk_size',
    type=int,
    default=32000,  # max speed (around 1M sentences / hour) and stability (CPU ~30%, GPU [60%;90%])
    help='Size of chunks for translation operations. Default: %(default)s')
parser.add_argument('--max-batch-size',
    type=int,
    default=32,	# same as argos
    help='Max batch size for translation. Default: %(default)s')



args = parser.parse_args()
try:
    with open(args.config) as f:
        config = json.loads(f.read())
    if args.reverse:
        config["from"], config["to"] = config["to"], config["from"]
except Exception as e:
    print(f"Cannot open config file: {e}")
    exit(1)
if not os.path.isfile(args.source):
    print(f"Cannot find source file. Please verify the path.")
    exit(1)

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)
ct2_model_dir = os.path.join(run_dir, "model")
sp_model = os.path.join(run_dir, "sentencepiece.model")
bpe_model = os.path.join(run_dir, "bpe.model")
bt_dir = os.path.dirname(args.source)
src_file = os.path.basename(args.source)

if not os.path.isdir(ct2_model_dir) or (not os.path.isfile(sp_model) and not os.path.isfile(bpe_model)):
    print(f"The model in {run_dir} is not valid. Did you run train.py first?")
    exit(1)
    
def translator():
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 and not args.cpu else "cpu"
    model = ctranslate2.Translator(ct2_model_dir, device=device, compute_type="default")
    if os.path.isfile(sp_model):
        tokenizer = SentencePieceTokenizer(sp_model)
    elif os.path.isfile(bpe_model):
        tokenizer = BPETokenizer(bpe_model, config["from"]["code"], config["to"]["code"])
    return {"model": model, "tokenizer": tokenizer}

def encode(text, tokenizer):
    return tokenizer.encode(text)

def decode(tokens, tokenizer):
#    detokenized = tokenizer.decode(tokens)
#    if len(detokenized) > 0 and detokenized[0] == " ":
#        detokenized = detokenized[1:]
#    return detokenized
    return tokenizer.lazy_processor().decode_pieces(tokens)

data = translator()

#Identify source file
src = args.source
# if source file has same name as synthetized file, rename it
if src_file == "source.txt":
    src = os.path.join(bt_dir, "src.txt")
    os.rename(args.source, src)
#  Define source as future reverse target and file path for future synthetic sources
tgt_bt = os.path.join(bt_dir, "target.txt")
src_bt = os.path.join(bt_dir, "source.txt")
# total line count prints to screen for alignment check
line_count = 0
with open (src, "r", encoding="utf-8") as source, \
    open (tgt_bt, "w", encoding="utf-8") as reversebttarget:
         
    for index, line in enumerate(source):
        reversebttarget.write(line)
        line_count += 1
# for testing purposes
#        if line_count == 100000:
#            break
print(f"Source reconstructed as future target. Launching backtranslation of {line_count} lines")
#  2. backtranslates source, to be used as synthetic source in reverse training

chunk_size = args.chunk_size
with open(tgt_bt, "r", encoding="utf-8") as source4bt, \
     open(src_bt, "w", encoding="utf-8") as backtranslation:
    progress_bar = tqdm(total=line_count)
    src_text = []
    # counters for triggering translation batch and checking global alignment
    line_number = 0
    written_lines = 0
    for line in source4bt:
        line = line.strip('\n')
        src_text.append(line)
        line_number +=1
        # encoding and decoding of chunks (to avoid memory issues)
        if line_number % chunk_size == 0 or line_number == line_count:
            backtranslation_obj = data["model"].translate_batch(
                [encode(t, data["tokenizer"]) for t in src_text],
                beam_size=4, # same as argos
                return_scores=False, # speed up,
                max_batch_size=args.max_batch_size,
                )
            backtranslated_text = [
                decode(tokens.hypotheses[0], data["tokenizer"])
                for tokens in backtranslation_obj
                ]
            for t in backtranslated_text:
                backtranslation.write(t)
                backtranslation.write("\n")
                written_lines +=1
            src_text = []
        progress_bar.update(1)
    progress_bar.close()
print(f"Backtranslated {written_lines} lines to {src_bt}")		