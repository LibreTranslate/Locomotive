import argparse
import io
import sys
import json
import os
import argparse
import hashlib
import zipfile
import shutil
from net import download
from data import merge_shuffle
import sentencepiece as spm

parser = argparse.ArgumentParser(description='Train LibreTranslate compatible models')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')

args = parser.parse_args() 
try:
    with open(args.config) as f:
        config = json.loads(f.read())
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
#rel_run_dir = f"run/{model_dirname}"
os.makedirs(cache_dir, exist_ok=True)


sources = {}

for s in config['sources']:
    md5 = hashlib.md5(s.encode('utf-8')).hexdigest()
    dataset_path = os.path.join(cache_dir, md5)
    zip_path = dataset_path + ".zip"

    if not os.path.isdir(dataset_path):
        if not os.path.isfile(zip_path):
            def print_progress(progress):
                print(f"\r{os.path.basename(zip_path)} [{int(progress)}%]     ", end='\r')
            
            download(s, cache_dir, progress_callback=print_progress, basename=os.path.basename(zip_path))
            print()

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
            sources[s] = {
                'source': source,
                'target': target,
                'hash': md5
            }

for k in sources:
    print(f" - {k} ({sources[k]['hash']})")


os.makedirs(run_dir, exist_ok=True)
merge_shuffle(sources, run_dir)

# TODO: run sentencepiece
# TODO: generate config.yml from template
# TODO: run generate vocab
# TODO: run opennmt-py
# TODO: run stanza
# TODO: package, ship
# TODO: support for files rather than URLs