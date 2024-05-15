import os
import json
import argparse
import ctranslate2
import sentencepiece
from sacrebleu import corpus_bleu
from data import get_flores
from tokenizer import BPETokenizer, SentencePieceTokenizer

parser = argparse.ArgumentParser(description='Evaluate LibreTranslate compatible models')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--reverse',
    action='store_true',
    help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
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
parser.add_argument('--flores_dataset',
    type=str,
    default="dev",
    help='Defines the flores200 dataset to translate. Default: %(default)s')
parser.add_argument('--translate_flores',
    action="store_true",
    help='Translate the flores200 corpus into a text file with .evl extension. Default: %(default)s')
parser.add_argument('--cpu',
    action="store_true",
    help='Force CPU use. Default: %(default)s')
parser.add_argument('--max-batch-size',
    type=int,
    default=16,
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
cache_dir = os.path.join(current_dir, "cache")
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)
ct2_model_dir = os.path.join(run_dir, "model")
sp_model = os.path.join(run_dir, "sentencepiece.model")
bpe_model = os.path.join(run_dir, "bpe.model")

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
    if args.tokens:
        return " ".join(tokens)
    else:
        detokenized = tokenizer.decode(tokens)
        if len(detokenized) > 0 and detokenized[0] == " ":
            detokenized = detokenized[1:]
        return detokenized

def translate_flores():
    tra_filename = f"flores200{dataset}-{model_dirname}.evl"
    tra_f = os.path.join(run_dir, tra_filename)
    with open(tra_f, "w", encoding="utf8") as translation_file:
        for t in translated_text:
            translation_file.write(t)
            translation_file.write("\n")
    return tra_f

data = translator()

if args.bleu or args.flores_id or args.translate_flores is not None:
    if args.flores_dataset:
        dataset = args.flores_dataset
    else: 
        dataset = "dev"
    src_text = get_flores(config["from"]["code"], dataset)
    tgt_text = get_flores(config["to"]["code"], dataset)
    
    if args.flores_id is not None:
        src_text = [src_text[args.flores_id]]
        tgt_text = [tgt_text[args.flores_id]]

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
    
    bleu_score = round(corpus_bleu(
        translated_text, [[x] for x in tgt_text]
    ).score, 5)

    if args.translate_flores:
        translate_flores()

    if args.flores_id is not None:
        print(f"({config['from']['code']})> {src_text[0]}\n(gt)> {tgt_text[0]}\n({config['to']['code']})> {' '.join(translated_text)}")
    else:
        print(f"BLEU score: {bleu_score}")
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
