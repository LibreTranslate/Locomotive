import os
import json
import argparse
import ctranslate2
import sentencepiece
import subprocess
from sacrebleu import corpus_bleu
from data import get_flores, get_flores_file_path
from tokenizer import BPETokenizer, SentencePieceTokenizer

parser = argparse.ArgumentParser(description='Evaluate LibreTranslate compatible models')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--reverse',
    action='store_true',
    help='Reverse the source and target languages in the configuration and data sources.')
parser.add_argument('--bleu',
    action="store_true",
    help='Evaluate BLEU score. Default: %(default)s')
parser.add_argument('--flores-id',
    type=int,
    default=None,
    help='Evaluate this flores sentence ID. Default: %(default)s')
parser.add_argument('--tokens',
    action="store_true",
    help='Display tokens rather than words. Default: %(default)s')
parser.add_argument('--pivot_from',
    type=str,
    help='Applies a pivotal translation from another language and evaluates the whole translation.')
parser.add_argument('--pivot_to',
    type=str,
    help='Applies a pivotal translation to another language and evaluates the whole translation.')
parser.add_argument('--flores_dataset',
    type=str,
    default="dev",
    help='Defines the flores200 dataset to translate. Default: %(default)s')
parser.add_argument('--translate_flores',
    action="store_true",
    help='Translate the flores200 corpus into a text file with .evl extension.')
parser.add_argument('--comet',
    action="store_true",
    help='Run COMET score command on the translated flores text.')
parser.add_argument('--cpu',
    action="store_true",
    help='Force CPU use. Default: %(default)s')
parser.add_argument('--max-batch-size',
    type=int,
    default=32,
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

current_dir = os.path.dirname(__file__)
utils_dir = os.path.join(current_dir, "utils")
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)
ct2_model_dir = os.path.join(run_dir, "model")
sp_model = os.path.join(run_dir, "sentencepiece.model")
bpe_model = os.path.join(run_dir, "bpe.model")

if not os.path.isdir(ct2_model_dir) or (not os.path.isfile(sp_model) and not os.path.isfile(bpe_model)):
    print(f"The model in {run_dir} is not valid. Did you run train.py first?")
    exit(1)

if args.pivot_from:
    pivot_dirname = f"{args.pivot_from}_{config['from']['code']}"
    pivot_dir = os.path.join(utils_dir, pivot_dirname)
    ct2_pivot_dir = os.path.join(pivot_dir, "model")
    sp_pivot = os.path.join(pivot_dir, "sentencepiece.model")
    if not os.path.isdir(ct2_pivot_dir) or (not os.path.isfile(sp_model)):
        print(f"The model from which to pivot in {pivot_dir} is not valid.")
        exit(1)        

if args.pivot_to:
    pivot_dirname = f"{config['to']['code']}_{args.pivot_to}"
    pivot_dir = os.path.join(utils_dir, pivot_dirname)
    ct2_pivot_dir = os.path.join(pivot_dir, "model")
    sp_pivot = os.path.join(pivot_dir, "sentencepiece.model")
    if not os.path.isdir(ct2_pivot_dir) or (not os.path.isfile(sp_model)):
        print(f"The model to pivot to in {pivot_dir} is not valid.")
        exit(1)


def pivot_translator():
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 and not args.cpu else "cpu"
    model = ctranslate2.Translator(ct2_pivot_dir, device=device, compute_type="default")
    tokenizer = SentencePieceTokenizer(sp_pivot)
    return {"model": model, "tokenizer": tokenizer}

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
    if args.tokens:
        return " ".join(tokens)
    else:
        return tokenizer.lazy_processor().decode_pieces(tokens)

def translate_flores():
    if args.pivot_from:
        tra_filename = f"flores200{dataset}-{args.pivot_from}_{model_dirname}.evl"
    elif args.pivot_to:
        tra_filename = f"flores200{dataset}-{config['from']['code']}_{config['to']['code']}_{args.pivot_to}-{config['version']}.evl"
    else:
        tra_filename = f"flores200{dataset}-{model_dirname}.evl"
    tra_f = os.path.join(run_dir, tra_filename)
    with open(tra_f, "w", encoding="utf8") as translation_file:
        for t in translated_text:
            translation_file.write(t)
            translation_file.write("\n")
    return tra_f

data = translator()

if args.bleu or args.flores_id or args.translate_flores or args.comet is not None:
    if args.flores_dataset:
        dataset = args.flores_dataset
    src_text = get_flores(config["from"]["code"], dataset)
    tgt_text = get_flores(config["to"]["code"], dataset)

    if args.flores_id is not None:
        src_text = [src_text[args.flores_id]]
        tgt_text = [tgt_text[args.flores_id]]

    if args.pivot_from:
        src_text = get_flores(args.pivot_from, dataset)
        pivot = pivot_translator()
        pivot_obj = pivot["model"].translate_batch(
            [encode(t, pivot["tokenizer"]) for t in src_text],
            beam_size=4, # same as argos
            return_scores=False, # speed up,
            max_batch_size=args.max_batch_size,
        )
		
        pivoted_text = [
            decode(tokens.hypotheses[0], pivot["tokenizer"])
            for tokens in pivot_obj
        ]
		
        src_text = pivoted_text
    


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
	
    if args.pivot_to:
        tgt_text = get_flores(args.pivot_to, dataset)
        pivot = pivot_translator()
        pivot_obj = pivot["model"].translate_batch(
            [encode(t, pivot["tokenizer"]) for t in translated_text],
            beam_size=4, # same as argos
            return_scores=False, # speed up,
            max_batch_size=args.max_batch_size,
        )
		
        pivoted_text = [
            decode(tokens.hypotheses[0], pivot["tokenizer"])
            for tokens in pivot_obj
        ]

        translated_text = pivoted_text        
    
    bleu_score = round(corpus_bleu(
        translated_text, [[x] for x in tgt_text]
    ).score, 5)

    other_bleu_score = round(corpus_bleu(
        translated_text, [tgt_text]
    ).score, 5)

    if args.translate_flores:
        translate_flores()

    if args.comet:
        if args.pivot_from:
            src_f = get_flores_file_path(args.pivot_from, dataset)
        else:
            src_f = get_flores_file_path(config["from"]["code"], dataset)
        if args.pivot_to:
            ref_f = get_flores_file_path(args.pivot_to, dataset)
        else:
            ref_f = get_flores_file_path(config["to"]["code"], dataset)
        tra_f = translate_flores()
        
        subprocess.run([
            "comet-score",
            "--sources",
            src_f,
            "--translations",
            tra_f,
            "--references",
            ref_f,
            "--quiet",
            "--only_system"]) 

    if args.flores_id is not None:
        print(f"({config['from']['code']})> {src_text[0]}\n(gt)> {tgt_text[0]}\n({config['to']['code']})> {' '.join(translated_text)}")
    else:
        print(f"BLEU score: {bleu_score}")
#        print(f"other_BLEU score: {other_bleu_score}")

else:
    # Interactive mode
    print("Starting interactive mode")

    while True:
        try:
            text = input(f"({config['from']['code']})> ")
        except KeyboardInterrupt:
            print("")
            exit(0)

        src_text = text.rstrip('\n')
        translation_obj = data["model"].translate_batch(
            [encode(src_text, data["tokenizer"])],
            beam_size=4, # same as argos
            return_scores=False, # speed up
        )

        translated_text = [
            decode(tokens.hypotheses[0], data["tokenizer"])
            for tokens in translation_obj
        ]
        print(f"({config['to']['code']})> {translated_text[0]}")
