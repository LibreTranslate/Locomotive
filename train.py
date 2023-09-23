import argparse
import io
import sys
import json
import os
import argparse
import hashlib
import zipfile
import shutil
import glob
import yaml
import subprocess
import stanza
import re
import zipfile
from net import download
from data import merge_shuffle
import sentencepiece as spm
from onmt_tools import average_models

parser = argparse.ArgumentParser(description='Train LibreTranslate compatible models')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--reverse',
    action='store_true',
    help='Reverse the source and target languages in the configuration and data sources. Default: %(default)s')
parser.add_argument('--rerun',
    action='store_true',
    help='Rerun the training from scratch. Default: %(default)s')
parser.add_argument('--toy',
    action='store_true',
    help='Train a toy model (useful for testing). Default: %(default)s')
args = parser.parse_args() 
try:
    with open(args.config) as f:
        config = json.loads(f.read())
    
    if args.reverse:
        config["from"], config["to"] = config["to"], config["from"]
except Exception as e:
    print(f"Cannot open config file: {e}")
    exit(1)

print(f"Training {config['from']['name']} --> {config['to']['name']} ({config['version']})")
print(f"Sources: {len(config['sources'])}")

metadata = {
    "package_version": config['version'],
    "argos_version": "1.5",
    "from_code": config['from']['code'],
    "from_name": config['from']['name'],
    "to_code": config['to']['code'],
    "to_name": config['to']['name'],
}
readme = f"# {config['from']['name']} - {config['to']['name']} version {config['version']}"

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)
onmt_dir = os.path.join(run_dir, "opennmt")
stanza_dir = os.path.join(run_dir, "stanza")
rel_run_dir = f"run/{model_dirname}"
rel_onmt_dir = f"{rel_run_dir}/opennmt"
os.makedirs(cache_dir, exist_ok=True)

if args.rerun:
    shutil.rmtree(run_dir)

sources = {}

for s in config['sources']:
    md5 = hashlib.md5(s.encode('utf-8')).hexdigest()
    if s.lower().startswith("file://"):
        source, target = None, None
        for f in [f.path for f in os.scandir(s[7:]) if f.is_file()]:
            if "target" in f.lower():
                target = f
            if "source" in f.lower():
                source = f
            
        if source is not None and target is not None:
            if args.reverse:
                source, target = target, source
            sources[s] = {
                'source': source,
                'target': target,
                'hash': md5
            }
        else:
            print(f"Cannot find a source.txt and a target.txt in {s}. Exiting...")
            exit(1)
    else:
        dataset_path = os.path.join(cache_dir, md5)
        # Network URL
        zip_path = dataset_path + ".zip"

        if not os.path.isdir(dataset_path):
            def download_source():
                def print_progress(progress):
                    print(f"\r{s} [{int(progress)}%]     ", end='\r')
                
                download(s, cache_dir, progress_callback=print_progress, basename=os.path.basename(zip_path))
                print()

            if not os.path.isfile(zip_path):
                download_source()
            else:
                # Quick check
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        pass
                except:
                    print(f"Corrupted .zip file, redownloading {zip_path}")
                    os.unlink(zip_path)
                    download_source()

            os.makedirs(dataset_path, exist_ok=True)
            print(f"Extracting {zip_path} to {dataset_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            
            os.unlink(zip_path)
        else:
            subfolders = [ f.path for f in os.scandir(dataset_path) if f.is_dir()]
            if len(subfolders) == 1:
                # Move files from subfolder
                for f in [f.path for f in os.scandir(subfolders[0]) if f.is_file()]:
                    shutil.move(f, dataset_path)
                
                shutil.rmtree(subfolders[0])
            
            # Find source, target files
            source, target = None, None
            for f in [f.path for f in os.scandir(dataset_path) if f.is_file()]:
                if "target" in f.lower():
                    target = f
                if "source" in f.lower():
                    source = f
                
            if source is not None and target is not None:
                if args.reverse:
                    source, target = target, source
                sources[s] = {
                    'source': source,
                    'target': target,
                    'hash': md5
                }
            else:
                print(f"Cannot find a source.txt and a target.txt in {s} ({dataset_path}). Exiting...")
                exit(1)

for k in sources:
    print(f" - {k} ({sources[k]['hash']})")


stanza_lang_code = config['from']['code']
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

os.makedirs(run_dir, exist_ok=True)
changed = merge_shuffle(sources, run_dir)

sp_model_path = os.path.join(run_dir, "sentencepiece.model")
if not os.path.isfile(sp_model_path) or changed:
    while True:
        try:
            spm.SentencePieceTrainer.train(input=glob.glob(f"{run_dir}/*-val.txt") + glob.glob(f"{run_dir}/*-train.txt"), 
                                            model_prefix=f"{run_dir}/sentencepiece", vocab_size=config.get('vocab_size', 50000),
                                            character_coverage=config.get('character_coverage', 1.0),
                                            input_sentence_size=config.get('input_sentence_size', 1000000),
                                            shuffle_input_sentence=True)
            break
        except Exception as e:
            err = str(e)
            if "Vocabulary size too high" in err:
                matches = re.match(".*Please set it to a value <= (\d+)", err)
                if matches is not None:
                    config["vocab_size"] = int(matches.group(1))
                    print(f"WARNING: vocabulary size too high, reducing to {matches.group(1)}")
                else:
                    print(err)
                    exit(1)
            else:
                print(err)
                exit(1)

os.makedirs(onmt_dir, exist_ok=True)

onmt_config = {
    'save_data': rel_onmt_dir,
    'src_vocab': f"{rel_onmt_dir}/openmt.vocab",
    'tgt_vocab': f"{rel_onmt_dir}/openmt.vocab",
    'src_vocab_size': config.get('vocab_size', 50000),
    'tgt_vocab_size': config.get('vocab_size', 50000),
    'share_vocab': True, 
    'data': {
        'corpus_1': {
            'path_src': f'{rel_run_dir}/src-train.txt', 
            'path_tgt': f'{rel_run_dir}/tgt-train.txt', 
            'transforms': ['sentencepiece', 'filtertoolong']
        }, 
        'valid': {
            'path_src': f'{rel_run_dir}/src-val.txt',
            'path_tgt': f'{rel_run_dir}/tgt-val.txt', 
            'transforms': ['sentencepiece', 'filtertoolong']
        }
    }, 
    'src_subword_model': f'{rel_run_dir}/sentencepiece.model', 
    'tgt_subword_model': f'{rel_run_dir}/sentencepiece.model', 
    'src_subword_nbest': 1, 
    'src_subword_alpha': 0.0, 
    'tgt_subword_nbest': 1, 
    'tgt_subword_alpha': 0.0, 
    'src_seq_length': 150, 
    'tgt_seq_length': 150, 
    'skip_empty_level': 'silent', 
    'save_model': f'{rel_onmt_dir}/openmt.model', 
    'save_checkpoint_steps': 1000, 
    'valid_steps': 5000, 
    'train_steps': 50000, 
    'early_stopping': 4, 
    'queue_size': 10000, 
    'bucket_size': 262144, 
    'world_size': 1, 
    'gpu_ranks': [0], 
    'batch_type': 'tokens', 
    'batch_size': 8192, 
    'valid_batch_size': 4096, 
    'max_generator_batches': 2, 
    'accum_count': [4], 
    'accum_steps': [0], 
    'model_dtype': 'fp32', 
    'optim': 'adam', 
    'learning_rate': 2, 
    'warmup_steps': 8000, 
    'decay_method': 'noam', 
    'adam_beta2': 0.998, 
    'max_grad_norm': 0, 
    'label_smoothing': 0.1, 
    'param_init': 0, 
    'param_init_glorot': True, 
    'normalization': 'tokens', 
    'encoder_type': 'transformer', 
    'decoder_type': 'transformer', 
    'position_encoding': True, 
    'enc_layers': 6, 
    'dec_layers': 6, 
    'heads': 8,
    'hidden_size': 512, 
    'rnn_size': 512,
    'word_vec_size': 512, 
    'transformer_ff': 2048, 
    'dropout_steps': [0],
    'dropout': [0.1], 
    'attention_dropout': [0.1], 
    'share_decoder_embeddings': True, 
    'share_embeddings': True
}

if args.toy:
    toy_config = {
        'valid_steps': 100, 
        'train_steps': 200, 
        'save_checkpoint_steps': 100
    }
    for k in toy_config:
        onmt_config[k] = toy_config[k]

# Config defined overrides
for k in onmt_config:
    if k in config:
        onmt_config[k] = config[k]

onmt_config_path = os.path.join(run_dir, "config.yml")
with open(onmt_config_path, "w", encoding="utf-8") as f:
    f.write(yaml.dump(onmt_config))
    print(f"Wrote {onmt_config_path}")

onmt_vocab_file = os.path.join(onmt_dir, "openmt.vocab")
if changed and os.path.isfile(onmt_vocab_file):
    os.unlink(onmt_vocab_file)
    
if not os.path.isfile(onmt_vocab_file):
    subprocess.run(["onmt_build_vocab", "-config", onmt_config_path, "-n_sample", "-1"])

last_checkpoint = os.path.join(onmt_dir, os.path.basename(onmt_config["save_model"]) + f'_step_{onmt_config["train_steps"]}.pt')
if not os.path.isfile(last_checkpoint):
    subprocess.run(["onmt_train", "-config", onmt_config_path])

# Average
average_checkpoint = os.path.join(run_dir, "averaged.pt")
checkpoints = sorted(glob.glob(os.path.join(onmt_dir, "*.pt")))
print(f"Total checkpoints: {len(checkpoints)}")

if not os.path.isfile(average_checkpoint):
    average_models(checkpoints[-2:], average_checkpoint)

# Quantize
ct2_model_dir = os.path.join(run_dir, "model")
if not os.path.isdir(ct2_model_dir):
    print("Converting to ctranslate2")
    subprocess.run([
            "ct2-opennmt-py-converter",
            "--model_path",
            average_checkpoint,
            "--output_dir",
            ct2_model_dir,
            "--quantization",
            "int8"])

# Package
readme_file = os.path.join(run_dir, "README.md")
with open(readme_file, "w", encoding="utf-8") as f:
    f.write(readme)

metadata_file = os.path.join(run_dir, "metadata.json")
with open(metadata_file, "w", encoding="utf-8") as f:
    f.write(json.dumps(metadata))

package_file = os.path.join(run_dir, f"translate-{config['from']['code']}_{config['to']['code']}-{config['version'].replace('.', '_')}.argosmodel")
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
    add_folder(ct2_model_dir)
    add_folder(stanza_dir)

print("Done!")
