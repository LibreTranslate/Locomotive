import argparse
import io
import sys
import json
import os
import argparse
import hashlib
import zipfile
import re
import shutil
import glob
import stanza
import ctranslate2
import subprocess
from net import download
import requests
import iso639

parser = argparse.ArgumentParser(description='Convert OPUS-MT bilingual models to LibreTranslate compatible models')
parser.add_argument('-s', '--source',
    type=str,
    default="en",
    required=True,
    help='Source language code: %(default)s')
parser.add_argument('-t', '--target',
    type=str,
    default="it",
    required=True,
    help='Target language code: %(default)s')
parser.add_argument('--model-url',
    type=str,
    default="",
    help='URL to OPUS model: %(default)s')
args = parser.parse_args()

src_lang = iso639.find(args.source)
tgt_lang = iso639.find(args.target)

if src_lang is None:
    print(f"Cannot find source language code: {args.source}")
    exit(1)
if tgt_lang is None:
    print(f"Cannot find target language code: {args.target}")
    exit(1)

model_url = args.model_url
if not model_url:
    # Fetch
    readme_url = f"https://raw.githubusercontent.com/Helsinki-NLP/OPUS-MT-train/master/models/{args.source}-{args.target}/README.md"
    print(f"Fetching {readme_url}")
    r = requests.get(readme_url)
    readme = r.content.decode("utf-8")
    matches = None
    for line in readme.split("\n"):
        matches = re.match(".*download: \[[^\]]+\]\((http[^)]+)\)", line)
        if matches:
            break

    if matches is None:
        print("Cannot find opus model URL. Please provide it manually via --model")
        exit(1)
    model_url = matches[1]

print(f"Model URL: {model_url}")
version = "1.0"

print(f"{src_lang['name']} --> {tgt_lang['name']} ({version})")

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

model_dirname = f"{args.source}_{args.target}-opus_{version}"
run_dir = os.path.join(current_dir, "run", model_dirname)
stanza_dir = os.path.join(run_dir, "stanza")

md5 = hashlib.md5(model_url.encode('utf-8')).hexdigest()
model_path = os.path.join(cache_dir, md5) 
zip_path = model_path + ".zip"

if not os.path.isdir(model_path):
    def print_progress(progress):
        print(f"\r{model_url} [{int(progress)}%]     ", end='\r')

    download(model_url, cache_dir, progress_callback=print_progress, basename=os.path.basename(zip_path))

    os.makedirs(model_path, exist_ok=True)
    print(f"Extracting {zip_path} to {model_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_path)

    os.unlink(zip_path)

# Identify pieces
spm_models = glob.glob(os.path.join(model_path, "*.spm"))
sp_model = None
for spm in spm_models:
    if "source" in spm.lower():
        sp_model = spm
        break

if sp_model is None:
    print("Cannot find SentencePiece source model")
    exit(1)

npz_models = glob.glob(os.path.join(model_path, "*.npz"))
npz_model = None

# Case with a single model
if len(npz_models) == 1:
    npz_model = npz_models[0]

# Search for "best" string
if npz_model is None:
    for nm in npz_models:
        if "best" in nm.lower():
            npz_model = nm
            break

# Search for largest size
if npz_model is None:
    largest = float('-inf')
    for nm in npz_models:
        size = os.path.getsize(nm)
        if size > largest:
            largest = size
            npz_model = nm

if npz_model is None:
    print("Cannot find npz model")
    exit(1)


vocabs = glob.glob(os.path.join(model_path, "*.yml"))
vocab_file = None
# Case with a single model
for v in vocabs:
    if "opus" in v.lower() and not "decoder" in v.lower():
        vocab_file = v
        break

if vocab_file is None:
    print("Cannot find vocab file")
    exit(1)

print(f"SentencePiece model: {sp_model}")
print(f"OPUS model: {npz_model}")
print(f"Vocab: {vocab_file}")

if os.path.isdir(run_dir):
    shutil.rmtree(run_dir)
os.makedirs(run_dir, exist_ok=True)

# Copy sp model
sp_model_path = os.path.join(run_dir, "sentencepiece.model")
shutil.copy(sp_model, sp_model_path)

stanza_lang_code = args.source
if not os.path.isdir(os.path.join(stanza_dir, stanza_lang_code)):
    while True:
        try:
            os.makedirs(stanza_dir, exist_ok=True)
            stanza.download(stanza_lang_code, dir=stanza_dir, processors="tokenize")
            break
        except Exception as e:
            if stanza_lang_code != "en":
                print(f'Could not locate stanza model for "{stanza_lang_code}", we will use "en" instead. Note this might not work well.')
                stanza_lang_code = "en"
            else:
                print(f'Cannot download stanza model: {str(e)}')
                exit(1)

# Quantize
ct2_model_dir = os.path.join(run_dir, "model")
if os.path.isdir(ct2_model_dir):
    shutil.rmtree(ct2_model_dir)

print("Converting to ctranslate2")
subprocess.run([
        "ct2-marian-converter",
        "--model_path",
        npz_model,
        "--vocab_paths",
        vocab_file,
        "--output_dir",
        ct2_model_dir,
        "--quantization",
        "int8"])

# Package
readme = f"""# {src_lang['name']} - {tgt_lang['name']} version {version}

Authors: Jörg Tiedemann and Santhosh Thottingal
Title: "OPUS-MT — Building open translation services for the World"
Book Title: Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (EAMT)
Year: 2020
Location: Lisbon, Portugal

The original OPUS model from which this packaged model is derived is licensed CC-BY 4.0
"""

metadata = {
    "package_version": version,
    "argos_version": "1.5",
    "from_code": args.source,
    "from_name": src_lang['name'],
    "to_code": args.target,
    "to_name": tgt_lang['name'],
}

readme_file = os.path.join(run_dir, "README.md")

with open(readme_file, "w", encoding="utf-8") as f:
    f.write(readme)

metadata_file = os.path.join(run_dir, "metadata.json")
with open(metadata_file, "w", encoding="utf-8") as f:
    f.write(json.dumps(metadata))


package_file = os.path.join(run_dir, f"translate-{args.source}_{args.target}-{version.replace('.', '_')}.argosmodel")
if os.path.isfile(package_file):
    os.unlink(package_file)

print(f"Writing {package_file}")
with zipfile.ZipFile(package_file, 'w', compression=zipfile.ZIP_STORED) as zipf:
    def add_file(f):
        zipf.write(f, arcname=os.path.basename(f))

    def add_folder(f):
        for root, dirs, files in os.walk(f):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, f)
                zipf.write(file_path, arcname=os.path.join(os.path.basename(f), arcname))

    add_file(readme_file)
    add_file(metadata_file)
    add_file(sp_model_path)
    add_folder(ct2_model_dir)
    add_folder(stanza_dir)

# Write config file
config = {
    "from": {
        "name": src_lang['name'],
        "code": args.source
    },
    "to": {
        "name": tgt_lang['name'],
        "code": args.target
    },
    "version": f"opus_{version}",
    "sources": [
    ]
}
config_file = os.path.join(run_dir, "config.json")
with open(config_file, "w", encoding="utf-8") as f:
    f.write(json.dumps(config))
print(f"Wrote {config_file}")

print("Done!")