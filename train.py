import argparse
import sys
import json
import os
import argparse
import hashlib
import shutil
import glob
import yaml
import subprocess
import re
import zipfile
import ctranslate2
from opus import get_opus_dataset_url
from net import download
from data import sources_changed, merge_shuffle, extract_flores_val
import sentencepiece as spm
from onmt_tools import average_models, sp_vocab_to_onmt_vocab
from sbd import package_sbd

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
parser.add_argument('--rerun-onmt',
    action='store_true',
    help='Rerun the training from ONMT training. Default: %(default)s')
parser.add_argument('--tensorboard',
    action='store_true',
    help='Run tensorboard during training. Default: %(default)s')
parser.add_argument('--toy',
    action='store_true',
    help='Train a toy model (useful for testing). Default: %(default)s')
parser.add_argument('--inflight',
    action='store_true',
    help='While training is in progress on a separate process, you can launch another instance of train.py with this flag turned on to build a model from the last available checkpoints rather that waiting until the end. Default: %(default)s')
parser.add_argument('--byte_fallback_off',
    action='store_false',
    help='Disable byte fallback during SentencePiece training. Default is enabled (True).')



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
    "argos_version": "1.9.0",
    "from_code": config['from']['code'],
    "from_name": config['from']['name'],
    "to_code": config['to']['code'],
    "to_name": config['to']['name'],
}
readme = f"# {config['from']['name']} - {config['to']['name']} version {config['version']}"

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
utils_dir = os.path.join(current_dir, "utils")
model_dirname = f"{config['from']['code']}_{config['to']['code']}-{config['version']}"
run_dir = os.path.join(current_dir, "run", model_dirname)
onmt_dir = os.path.join(run_dir, "opennmt")
rel_run_dir = f"run/{model_dirname}"
rel_onmt_dir = f"{rel_run_dir}/opennmt"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(utils_dir, exist_ok=True)

if args.rerun and os.path.isdir(run_dir):
    shutil.rmtree(run_dir)
os.makedirs(run_dir, exist_ok=True)

sources = {}

for s in config['sources']:
    filters = []
    transforms = []
    augmenters = []
    weight = None

    if isinstance(s, dict):
        if not "source" in s:
            print("Malformed source: {s}. A 'source' key is required.")
        filters = s.get('filters', [])
        transforms = s.get('transforms', [])
        augmenters = s.get('augmenters', [])
        weight = s.get("weight")
        s = s["source"]

    md5 = hashlib.md5(s.encode('utf-8')).hexdigest()
    
    def add_source_from(dir):
        source, target = None, None
        skip_reverse = False
        for f in [f.path for f in os.scandir(dir) if f.is_file()]:
            if "target" in f.lower():
                target = f
            elif f.lower().endswith(f".{config['to']['code']}"):
                target = f
                skip_reverse = True
            
            if "source" in f.lower():
                source = f
            elif f.lower().endswith(f".{config['from']['code']}"):
                source = f
                skip_reverse = True


        if source is not None and target is not None:
            if args.reverse and not skip_reverse:
                source, target = target, source
            sources[s] = {
                'source': source,
                'from': config['from']['code'],
                'target': target,
                'to': config['to']['code'],
                'hash': md5,
                'filters': filters,
                'transforms': transforms,
                'augmenters': augmenters,
                'weight': weight,
            }
        else:
            print(f"Cannot find a source.txt and a target.txt in {s} ({dir}). Exiting...")
            exit(1)

    if s.lower().startswith("file://"):
        add_source_from(s[7:])
    else:
        if s.lower().startswith("opus://"):
            try:
                s = get_opus_dataset_url(s[7:], config["from"]["code"], config["to"]["code"], run_dir)
            except Exception as e:
                print(e)
                exit(1)

        # Network/OPUS URL
        dataset_path = os.path.join(cache_dir, md5)
        zip_path = dataset_path + ".zip"

        # Download first?
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
        
            subfolders = [ f.path for f in os.scandir(dataset_path) if f.is_dir()]
            if len(subfolders) == 1:
                # Move files from subfolder
                for f in [f.path for f in os.scandir(subfolders[0]) if f.is_file()]:
                    shutil.move(f, dataset_path)
                
                shutil.rmtree(subfolders[0])
        
        add_source_from(dataset_path)

for k in sources:
    if config.get('filters'):
        for f in reversed(config['filters']):
            sources[k]['filters'].insert(0, f)
    if config.get('transforms'):
        for t in reversed(config['transforms']):
            sources[k]['transforms'].insert(0, t)
    if config.get('augmenters'):
        for a in reversed(config['augmenters']):
            sources[k]['augmenters'].insert(0, a)

    print(f" - {k} (hash:{sources[k]['hash'][:7]})")

packaged_sbd = package_sbd(run_dir, config['from']['code'])

all_weighted = sum([1 for k in sources if sources[k]['weight'] is not None]) == len(sources)
if all_weighted:
    extract_flores_val(config['from']['code'], config['to']['code'], run_dir, dataset="devtest")
changed = merge_shuffle(sources, run_dir)
has_merged = os.path.isfile(os.path.join(rel_run_dir, 'src-train.txt'))

sp_model_path = os.path.join(run_dir, "sentencepiece.model")
if not os.path.isfile(sp_model_path) or changed:
    while True:
        try:
            datasets = []
            if has_merged:
                datasets += [os.path.join(run_dir, "src-train.txt"), os.path.join(run_dir, "tgt-train.txt")]
            for k in sources:
                if sources[k]['weight'] is not None:
                    datasets += [sources[k]['source'], sources[k]['target']]
            #Byte-fallback (train byte tokens with character_coverage 0.9999, 0.9995 for CJK
            # doubling sentence input makes for more accurate sampling
            spm.SentencePieceTrainer.train(input=datasets, 
                                            model_prefix=f"{run_dir}/sentencepiece", vocab_size=config.get('vocab_size', 50000),
                                            character_coverage=config.get('character_coverage', 0.9999),
                                            input_sentence_size=config.get('input_sentence_size', 2000000),
                                            shuffle_input_sentence=True,
                                            byte_fallback=args.byte_fallback_off)
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
# different transforms at train & valid because of tokenization bug in onmt3.5
# RoPE+gated-activation requires upgrading, further details on architecture at upcoming TRANSFORMERS.md
train_transforms = ['sentencepiece', 'filtertoolong']
valid_transforms = ['sentencepiece']

corpora = {
    'valid': {
        'path_src': f'{rel_run_dir}/src-val.txt',
        'path_tgt': f'{rel_run_dir}/tgt-val.txt', 
        'transforms': valid_transforms
    }
}
if has_merged:
    corpora['corpus_1'] = {
        'path_src': f'{rel_run_dir}/src-train.txt',
        'path_tgt': f'{rel_run_dir}/tgt-train.txt',
        'transforms': train_transforms,
        'weight': 1
    }

for k in sources:
    if sources[k]['weight'] is not None:
        corpora[k] = {
            'path_src': sources[k]['source'],
            'path_tgt': sources[k]['target'],
            'weight': sources[k]['weight'],
            'transforms': train_transforms,
        }

onmt_config = {
    'save_data': rel_onmt_dir,
    'src_vocab': f"{rel_onmt_dir}/openmt.vocab",
    'tgt_vocab': f"{rel_onmt_dir}/openmt.vocab",
    'src_vocab_size': config.get('vocab_size', 50000), #default onmt value: 32768
    'tgt_vocab_size': config.get('vocab_size', 50000), #same as former
    'share_vocab': True, 
    'data': corpora, 
    'src_subword_type': 'sentencepiece',
    'tgt_subword_type': 'sentencepiece',
    'src_onmttok_kwargs': {
        'mode': 'none',
        'lang': config['from']['code'],
    },
    'tgt_onmttok_kwargs': {
        'mode': 'none',
        'lang': config['to']['code'],
    },
    'src_subword_model': f'{rel_run_dir}/sentencepiece.model', 
    'tgt_subword_model': f'{rel_run_dir}/sentencepiece.model', 
    'src_subword_nbest': 1, 
    'src_subword_alpha': 0.0, 
    'tgt_subword_nbest': 1, 
    'tgt_subword_alpha': 0.0, 
    'src_seq_length': 150, #onmt_train default si 192...
    'tgt_seq_length': 150, #same as former
    'skip_empty_level': 'silent', 
    'save_model': f'{rel_onmt_dir}/openmt.model', 
    'save_checkpoint_steps': 2500, 
    'keep_checkpoint': 10,
    'valid_steps': 2500, 
    'train_steps': 100000, 
    'early_stopping': 4, 
    'bucket_size': 262144, 
    'num_worker': 2,
    'world_size': 1, 
    'gpu_ranks': [0], 
    'batch_type': 'tokens', 
    'queue_size': 10000,
    'batch_size': 8192,
    'valid_batch_size': 2048,
    'max_generator_batches': 2, 
    'accum_count': 8, 
    'accum_steps': 0, 
    'model_dtype': 'fp16', 
    'optim': 'adam', 
    'learning_rate': 0.15,
    'warmup_steps': 16000, 
    'decay_method': 'rsqrt', 
    'adam_beta2': 0.998, 
    'max_grad_norm': 0, 
    'label_smoothing': 0.1, 
    'param_init': 0, 
    'param_init_glorot': True, 
    'normalization': 'tokens', 
    'encoder_type': 'transformer', 
    'decoder_type': 'transformer', 
    'position_encoding': True, #onmt default, False for relative [Shaw] and rotative  [RoPE] position encoding
    'max_relative_positions': 0, #onmt default, 20 and 32 will do Shaw, -1 will do RoPE
	'pos_ffn_activation_fn': 'relu', #to use "gated-gelu" or "silu", modify the CTranslate2 converter
    'enc_layers': 6, 
    'dec_layers': 6,
    'heads': 8,
    'hidden_size': 512, 
    'rnn_size': 512,
    'word_vec_size': 512, 
    'transformer_ff': 2048,
    'dropout_steps': 0,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'share_decoder_embeddings': True,
    'share_embeddings': True,
    'valid_metrics': ['BLEU'],
    'seed': -1, #onmt_default (auto seed) -when researching : any positive value-
}

no_gpu = ctranslate2.get_cuda_device_count() == 0
if sys.platform == 'darwin' or no_gpu:
    # CPU
    del onmt_config['gpu_ranks']

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

sp_vocab_file = os.path.join(run_dir, "sentencepiece.vocab")
onmt_vocab_file = os.path.join(onmt_dir, "openmt.vocab")
if changed and os.path.isfile(onmt_vocab_file):
    os.unlink(onmt_vocab_file)
    
if not os.path.isfile(onmt_vocab_file):
    #subprocess.run(["onmt_build_vocab", "-config", onmt_config_path, "-n_sample", "-1", "-num_threads", str(os.cpu_count())])
    sp_vocab_to_onmt_vocab(sp_vocab_file, onmt_vocab_file)

last_checkpoint = os.path.join(onmt_dir, os.path.basename(onmt_config["save_model"]) + f'_step_{onmt_config["train_steps"]}.pt')
def get_checkpoints():
    chkpts = [cp for cp in glob.glob(os.path.join(onmt_dir, "*.pt")) if "averaged.pt" not in cp]
    return list(sorted(chkpts, key=lambda x: int(re.findall('\d+', x)[0])))

if (not (os.path.isfile(last_checkpoint) or args.inflight)) or changed or args.rerun_onmt:
    cmd = ["onmt_train", "-config", onmt_config_path]

    if args.rerun_onmt:
        delete_checkpoints = glob.glob(os.path.join(onmt_dir, "*.pt"))
        for dc in delete_checkpoints:
            os.unlink(dc)
            print(f"Removed {dc}")

    if args.tensorboard:
        print("Launching tensorboard")

        from tensorboard import program
        import webbrowser
        import mimetypes

        log_dir = os.path.join(onmt_dir, "logs")
        
        # Allow tensorboard to run on Windows due to mimetypes bug: https://github.com/microsoft/vscode-python/pull/16203
        mimetypes.add_type("application/javascript", ".js")

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', os.path.abspath(log_dir)])
        url = tb.launch()
        print(f"Tensorboard URL: {url}")
        webbrowser.open(url)

        cmd += ["--tensorboard", "--tensorboard_log_dir", log_dir]
    
    # Resume?
    checkpoints = get_checkpoints()
    if len(checkpoints) > 0 and not changed:
        print(f"Resuming from {checkpoints[-1]}")
        cmd += ["--train_from", checkpoints[-1]]

    subprocess.run(cmd)

# Average
average_checkpoint = os.path.join(run_dir, "averaged.pt")
checkpoints = get_checkpoints()
print(f"Total checkpoints: {len(checkpoints)}")

if len(checkpoints) == 0:
    print("Something went wrong, looks like onmt_train failed?")
    exit(1)

if os.path.isfile(average_checkpoint):
    os.unlink(average_checkpoint)

if len(checkpoints) == 1 or args.inflight:
    print("Single checkpoint")
    average_checkpoint = checkpoints[-1]
else:
    if config.get('avg_checkpoints', 1) == 1:
        print("No need to average 1 model")
        average_checkpoint = checkpoints[-1]
    else:
        avg_num = min(config.get('avg_checkpoints', 1), len(checkpoints))
        print(f"Averaging {avg_num} models")
        average_models(checkpoints[-avg_num:], average_checkpoint)
# Quantize
ct2_model_dir = os.path.join(run_dir, "model")
if os.path.isdir(ct2_model_dir):
    shutil.rmtree(ct2_model_dir)

print("Converting to ctranslate2")
subprocess.run([
        "ct2-opennmt-py-converter",
        "--model_path",
        average_checkpoint,
        "--output_dir",
        ct2_model_dir,
        "--quantization",
        "int8"])

# Create .argosmodel package
package_slug = f"translate-{config['from']['code']}_{config['to']['code']}-{config['version'].replace('.', '_')}"
package_file = os.path.join(run_dir, f"{package_slug}.argosmodel")
if os.path.isfile(package_file):
    os.unlink(package_file)
package_folder = os.path.join(run_dir, package_slug)
if os.path.isdir(package_folder):
    shutil.rmtree(package_folder)
os.makedirs(package_folder, exist_ok=True)

readme_file = os.path.join(package_folder, "README.md")
with open(readme_file, "w", encoding="utf-8") as f:
    f.write(readme)
metadata_file = os.path.join(package_folder, "metadata.json")
with open(metadata_file, "w", encoding="utf-8") as f:
    f.write(json.dumps(metadata))

shutil.copy(sp_model_path, package_folder)
shutil.copytree(ct2_model_dir, os.path.join(package_folder, "model"))
if os.path.isdir(packaged_sbd):
    shutil.copytree(packaged_sbd, os.path.join(package_folder, os.path.basename(packaged_sbd)))

print(f"Writing {package_file}")
zip_filename = os.path.join(run_dir, f"{package_slug}.zip")
def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipdir(package_folder, zipf)
os.rename(zip_filename, package_file)
print("Done!")
