import os
import json
import argparse
import ctranslate2
import sentencepiece
from sacrebleu import corpus_bleu
from data import get_flores

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

class BPETokenizer:
    def __init__(self, model):
        self.model = model
    
    def Encode(self, text, out_type=str):
        return ["Is this impeding task straight@@ for@@ ward@@ ?".split(" ")]

def translator():
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = ctranslate2.Translator(ct2_model_dir, device=device, compute_type="auto")
    if os.path.isfile(sp_model):
        tokenizer = sentencepiece.SentencePieceProcessor(sp_model)
    elif os.path.isfile(bpe_model):
        tokenizer = BPETokenizer(bpe_model)
    return {"model": model, "tokenizer": tokenizer}

def encode(text, tokenizer):
    return tokenizer.Encode(text, out_type=str)

def decode(tokens):
    from mosestokenizer import MosesTokenizer, MosesDetokenizer
    dt = MosesDetokenizer()
    return dt(tokens).replace('@@ ','')

    if args.tokens:
        return " ".join(tokens)
    else:
        detokenized = "".join(tokens).replace("▁", " ")
        if len(detokenized) > 0 and detokenized[0] == " ":
            detokenized = detokenized[1:]
        return detokenized

data = translator()

if args.bleu or args.flores_id is not None:
    src_text = get_flores(config["from"]["code"], "dev")
    tgt_text = get_flores(config["to"]["code"], "dev")
    
    if args.flores_id is not None:
        src_text = [src_text[args.flores_id]]
        tgt_text = [tgt_text[args.flores_id]]

    translation_obj = data["model"].translate_batch(
        encode(src_text, data["tokenizer"]),
        beam_size=4, # same as argos
        return_scores=False, # speed up
    )

    translated_text = [
        decode(tokens.hypotheses[0])
        for tokens in translation_obj
    ]
    
    bleu_score = round(corpus_bleu(
        translated_text, [[x] for x in tgt_text], tokenize="flores200"
    ).score, 5)

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

        src_text = [text.rstrip('\n')]
        translation_obj = data["model"].translate_batch(
            encode(src_text, data["tokenizer"]),
            beam_size=4, # same as argos
            return_scores=False, # speed up
        )
        translated_text = [
            decode(tokens.hypotheses[0])
            for tokens in translation_obj
        ]
        print(f"({config['to']['code']})> {translated_text[0]}")