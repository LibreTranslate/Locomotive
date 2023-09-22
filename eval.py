import os
import json
import argparse
import ctranslate2
import sentencepiece
from sacrebleu import corpus_bleu
from net import download

parser = argparse.ArgumentParser(description='Evaluate LibreTranslate compatible models')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--bleu',
    action="store_true",
    help='Evaluate BLEU score. Default: %(default)s')


args = parser.parse_args()
try:
    with open(args.config) as f:
        config = json.loads(f.read())
except Exception as e:
    print(f"Cannot open config file: {e}")
    exit(1)

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)
ct2_model_dir = os.path.join(run_dir, "model")
sp_model = os.path.join(run_dir, "sentencepiece.model")

if not os.path.isdir(ct2_model_dir) or not os.path.isfile(sp_model):
    print(f"The model in {run_dir} is not valid. Did you run train.py first?")
    exit(1)

def translator():
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    model = ctranslate2.Translator(ct2_model_dir, device=device, compute_type="auto")
    tokenizer = sentencepiece.SentencePieceProcessor(sp_model)
    return {"model": model, "tokenizer": tokenizer}

def encode(text, tokenizer):
    return tokenizer.Encode(text, out_type=str)

def decode(tokens, tokenizer):
    return tokenizer.Decode(tokens)

data = translator()

if args.bleu:
    flores_dataset = os.path.join(cache_dir, "flores200_dataset", "dev")

    if not os.path.isdir(flores_dataset):
        os.makedirs(cache_dir, exist_ok=True)
    
        # Download first
        print("Downloading flores200 dataset...")
        fname = os.path.join(cache_dir, "flores200.tar.gz")
        flores_url = "https://tinyurl.com/flores200dataset"
        download(flores_url, cache_dir, basename=os.path.basename(fname))

        import tarfile
        with tarfile.open(fname) as f:
            f.extractall(cache_dir)
        
        if os.path.isfile(fname):
            os.unlink(fname)

        if not os.path.isdir(flores_dataset):
            print(f"Cannot download flores200. Please manually download it from {flores_url} and place it in {cache_dir}")
            exit(1)

    nllb_langs = {
        "af":"afr_Latn",
        "ak":"aka_Latn",
        "am":"amh_Ethi",
        "ar":"arb_Arab",
        "as":"asm_Beng",
        "ay":"ayr_Latn",
        "az":"azj_Latn",
        "bm":"bam_Latn",
        "be":"bel_Cyrl",
        "bn":"ben_Beng",
        "bho":"bho_Deva",
        "bs":"bos_Latn",
        "bg":"bul_Cyrl",
        "ca":"cat_Latn",
        "ceb":"ceb_Latn",
        "cs":"ces_Latn",
        "ckb":"ckb_Arab",
        "tt":"crh_Latn",
        "cy":"cym_Latn",
        "da":"dan_Latn",
        "de":"deu_Latn",
        "el":"ell_Grek",
        "en":"eng_Latn",
        "eo":"epo_Latn",
        "et":"est_Latn",
        "eu":"eus_Latn",
        "ee":"ewe_Latn",
        "fa":"pes_Arab",
        "fi":"fin_Latn",
        "fr":"fra_Latn",
        "gd":"gla_Latn",
        "ga":"gle_Latn",
        "gl":"glg_Latn",
        "gn":"grn_Latn",
        "gu":"guj_Gujr",
        "ht":"hat_Latn",
        "ha":"hau_Latn",
        "he":"heb_Hebr",
        "hi":"hin_Deva",
        "hr":"hrv_Latn",
        "hu":"hun_Latn",
        "hy":"hye_Armn",
        "nl":"nld_Latn",
        "ig":"ibo_Latn",
        "ilo":"ilo_Latn",
        "id":"ind_Latn",
        "is":"isl_Latn",
        "it":"ita_Latn",
        "jv":"jav_Latn",
        "ja":"jpn_Jpan",
        "kn":"kan_Knda",
        "ka":"kat_Geor",
        "kk":"kaz_Cyrl",
        "km":"khm_Khmr",
        "rw":"kin_Latn",
        "ko":"kor_Hang",
        "ku":"kmr_Latn",
        "lo":"lao_Laoo",
        "lv":"lvs_Latn",
        "ln":"lin_Latn",
        "lt":"lit_Latn",
        "lb":"ltz_Latn",
        "lg":"lug_Latn",
        "lus":"lus_Latn",
        "mai":"mai_Deva",
        "ml":"mal_Mlym",
        "mr":"mar_Deva",
        "mk":"mkd_Cyrl",
        "mg":"plt_Latn",
        "mt":"mlt_Latn",
        "mni-Mtei":"mni_Beng",
        "mni":"mni_Beng",
        "mn":"khk_Cyrl",
        "mi":"mri_Latn",
        "ms":"zsm_Latn",
        "my":"mya_Mymr",
        "no":"nno_Latn",
        "ne":"npi_Deva",
        "ny":"nya_Latn",
        "om":"gaz_Latn",
        "or":"ory_Orya",
        "pl":"pol_Latn",
        "pt":"por_Latn",
        "ps":"pbt_Arab",
        "qu":"quy_Latn",
        "ro":"ron_Latn",
        "ru":"rus_Cyrl",
        "sa":"san_Deva",
        "si":"sin_Sinh",
        "sk":"slk_Latn",
        "sl":"slv_Latn",
        "sm":"smo_Latn",
        "sn":"sna_Latn",
        "sd":"snd_Arab",
        "so":"som_Latn",
        "es":"spa_Latn",
        "sq":"als_Latn",
        "sr":"srp_Cyrl",
        "su":"sun_Latn",
        "sv":"swe_Latn",
        "sw":"swh_Latn",
        "ta":"tam_Taml",
        "te":"tel_Telu",
        "tg":"tgk_Cyrl",
        "tl":"tgl_Latn",
        "th":"tha_Thai",
        "ti":"tir_Ethi",
        "ts":"tso_Latn",
        "tk":"tuk_Latn",
        "tr":"tur_Latn",
        "ug":"uig_Arab",
        "uk":"ukr_Cyrl",
        "ur":"urd_Arab",
        "uz":"uzn_Latn",
        "vi":"vie_Latn",
        "xh":"xho_Latn",
        "yi":"ydd_Hebr",
        "yo":"yor_Latn",
        "zh-CN":"zho_Hans",
        "zh":"zho_Hans",
        "zh-TW":"zho_Hant",
        "zu":"zul_Latn",
        "pa":"pan_Guru"
    }

    src_f = os.path.join(flores_dataset, nllb_langs[config["from"]["code"]] + ".dev")
    tgt_f = os.path.join(flores_dataset, nllb_langs[config["to"]["code"]] + ".dev")

    src_text = [line.rstrip('\n') for line in open(src_f, encoding="utf-8")]
    tgt_text = [line.rstrip('\n') for line in open(tgt_f, encoding="utf-8")]
    
    translation_obj = data["model"].translate_batch(
        encode(src_text, data["tokenizer"]),
        beam_size=4, # same as argos
        return_scores=False, # speed up
    )

    translated_text = [
        decode(tokens.hypotheses[0], data["tokenizer"])
        for tokens in translation_obj
    ]
    
    bleu_score = round(corpus_bleu(
        translated_text, [[x] for x in tgt_text], tokenize="flores200"
    ).score, 5)

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
            decode(tokens.hypotheses[0], data["tokenizer"])
            for tokens in translation_obj
        ]
        print(f"({config['to']['code']})> {translated_text[0]}")