import os
import json
import argparse
import ctranslate2
from tqdm import tqdm
import sentencepiece
from tokenizer import BPETokenizer, SentencePieceTokenizer

parser = argparse.ArgumentParser(description='Translate whole corpora with an argos-translate model to augment a dataset.')
parser.add_argument('--source_lang', '-s',
    type=str,
    help='Language not to translate in corpus. Default: %(default)s')
parser.add_argument('--pivot_lang', '-p',
    type=str,
    default="en",
    help='Language to translate.')
parser.add_argument('--translate_into', '-t',
    type=str,
    help='Language to translate to.')
parser.add_argument('--corpus', '-c',
    type=str,
    help='Corpus that will be translated.')
parser.add_argument('--from_cache', '-k',
    action="store_true",
    help='Get corpus in cache file instead of running directory')
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
current_dir = os.path.dirname(__file__)

if args.translate_into:
    tgt = args.translate_into
    piv = args.pivot_lang
    model_dirname = f"{piv}_{tgt}"
    model_dir = os.path.join(current_dir, "run", model_dirname)	
else:
    print('Please specify the language to translate to.')
    exit(1)

if args.corpus:	
    corpus = args.corpus
else:
    print('Please specify a corpus.')
    exit(1)

if args.source_lang:
    src = args.source_lang
else:
    if not args.from_cache:
        print('A source language is required to define the working directory unambiguously.')
        exit(1)
    else:
        print(f"Looking for the {corpus} corpus in the cache directory.")

if args.from_cache:
    path_to_cache = None
    for path, directories, files in os.walk(os.path.join(current_dir, "cache")):
        for file in files:
            if file.startswith(corpus) and file.endswith(f".{piv}"):
                print(f"Found {os.path.join(path, file)} for translation.")
                path_to_cache = path
                if not args.source_lang:
                    print('Determining language pair of cached corpus')
# 1st, get the language chain in the corpus
                    chain = file[len(corpus)+1:-(len(piv)+1)]
# 2nd case A, the corpus is going TO the pivot					
                    if chain.endswith(piv):
                        src = chain[ :-len(piv)-1]
                        print(f"Source language for the cached corpus is {src}.")
# 2nd case B, the corpus is going FROM the pivot	
                    elif chain.startswith(piv):
                        src = chain[len(piv)+1: ]
                        print(f"Source language for the cached corpus is {src}.")
                    else :
                        print(f"Chain does not end or begin with {piv}, cannot extract source language from it. Wrong script")
                        exit(1)						
                if (f"{src}-{piv}") in file:
                    print('Cached corpus is direct.')
                    source = os.path.join(path_to_cache, f"{corpus}.{src}-{piv}.{src}")
                    pivot = os.path.join(path_to_cache, f"{corpus}.{src}-{piv}.{piv}")
                    target = os.path.join(path_to_cache, f"{corpus}.{src}-{tgt}.{tgt}")
                    pivoted_src = os.path.join(path_to_cache, f"{corpus}.{src}-{tgt}.{src}")
                elif (f"{piv}-{src}") in file:
                    print('Cached corpus is reverse.')
                    source = os.path.join(path_to_cache, f"{corpus}.{piv}-{src}.{src}")
                    pivot = os.path.join(path_to_cache, f"{corpus}.{piv}-{src}.{piv}")
                    target = os.path.join(path_to_cache, f"{corpus}.{tgt}-{src}.{tgt}")
                    pivoted_src = os.path.join(path_to_cache, f"{corpus}.{tgt}-{src}.{src}")
                else: 
                    print('Cached corpus does not feature the specified source language. Check both')
            if path_to_cache is not None:
                break
    run_dir = path_to_cache
else:
    if os.path.isdir(os.path.join(current_dir, "run", f"{src}_{piv}-1.1.{corpus}")):
        print('Pivot directory is direct.')
        piv_dir = os.path.join(current_dir, "run", f"{src}_{piv}-1.1.{corpus}")
        tgt_dir = os.path.join(current_dir, "run", f"{src}_{tgt}-1.1.{corpus}")
        reverse = False
    elif os.path.isdir(os.path.join(current_dir, "run", f"{piv}_{src}-1.1.{corpus}")):
        print('Pivot directory is reverse.')
        piv_dir = os.path.join(current_dir, "run", f"{piv}_{src}-1.1.{corpus}")
        tgt_dir = os.path.join(current_dir, "run", f"{tgt}_{src}-1.1.{corpus}")
        reverse = True
    else:
        print(f"Not sure there is a data corpus to translate, please check subdirectories in run for {src}_{piv}-1.1.{corpus}")
        exit(1)
    source = os.path.join(piv_dir, f"{corpus}.{src}")
    pivot = os.path.join(piv_dir, f"{corpus}.{piv}")
    target = os.path.join(piv_dir, f"{corpus}.{tgt}")
    if os.path.isfile(os.path.join(piv_dir, "source.txt")) and os.path.isfile(os.path.join(piv_dir, "target.txt")):
        if not reverse:
            os.rename(os.path.join(piv_dir, "source.txt"), source)
            os.rename(os.path.join(piv_dir, "target.txt"), pivot)
        else:
            os.rename(os.path.join(piv_dir, "source.txt"), pivot)
            os.rename(os.path.join(piv_dir, "target.txt"), source)

ct2_model_dir = os.path.join(model_dir, "model")
sp_model = os.path.join(model_dir, "sentencepiece.model")
bpe_model = os.path.join(model_dir, "bpe.model")

if not os.path.isdir(ct2_model_dir) or (not os.path.isfile(sp_model) and not os.path.isfile(bpe_model)):
    print(f"The model in {model_dir} is not valid. Did you run train.py first?")
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

def count_lines(file):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def translate_pivot(pivot, target):
# 1. total line count prints to screen for alignment check
    line_count = count_lines(pivot)
# 1a. check for existing aligned file (don't want to crash it inadvertently).
    if os.path.isfile(target):
        line_number = count_lines(target)
        if line_count == line_number:
            print('It looks like there already is a parallel translation. Please check.')
            exit(1)
#  2. if not, opens files and translates pivot into synthetic target    
    with open (pivot, "r", encoding="utf-8") as pivot, \
         open (target, "w", encoding="utf-8") as target:         
        print(f"Launching translation of {line_count} lines from {piv} to {tgt}.")
        chunk_size = args.chunk_size
        progress_bar = tqdm(total=line_count)
        src_text = []
   # counters for triggering translation batch and checking global alignment
        line_number = 0
        written_lines = 0
        for line in pivot:
            line = line.strip('\n')
            src_text.append(line)
            line_number +=1
        # encoding and decoding of chunks (to avoid memory issues)
            if line_number % chunk_size == 0 or line_number == line_count:
                translation_obj = data["model"].translate_batch(
                    [encode(t, data["tokenizer"]) for t in src_text],
                    beam_size=4, # same as argos
                    return_scores=False, # speed up,
                    max_batch_size=args.max_batch_size,
                    )
                translated_text = [
                    decode(tokens.hypotheses[0], data["tokenizer"])
                    for tokens in translation_obj
                    ]
                for t in translated_text:
                    target.write(t)
                    target.write("\n")
                    written_lines +=1
                src_text = []
            progress_bar.update(1)
    progress_bar.close()
    print(f"Pivoted {written_lines} lines of {piv} to {tgt} in {target} to use with {src}.")

if __name__ == "__main__":
    translate_pivot(pivot, target)
	
    if args.from_cache:
        os.rename(source, pivoted_src)
        print ('Renamed source file to target pair.')
    else:
        os.rename(piv_dir, tgt_dir)
        print('Renamed pivot directory to target pair.')
