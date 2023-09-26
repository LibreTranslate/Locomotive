import argparse
import io
import json
import os
import hashlib

parser = argparse.ArgumentParser(description='Find text in data sources')
parser.add_argument('--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('--text',
    type=str,
    help='Text to search. Default: %(default)s')

args = parser.parse_args() 
try:
    with open(args.config) as f:
        config = json.loads(f.read())
except Exception as e:
    print(f"Cannot open config file: {e}")
    exit(1)

current_dir = os.path.dirname(__file__)
cache_dir = os.path.join(current_dir, "cache")
sources = {}

text = args.text.lower()

for s in config['sources']:
    md5 = hashlib.md5(s.encode('utf-8')).hexdigest()
    source_dir = os.path.join(cache_dir, md5)
    if os.path.isdir(source_dir):
        source, target = None, None
        for f in [f.path for f in os.scandir(source_dir) if f.is_file()]:
            if "target" in f.lower():
                target = f
            if "source" in f.lower():
                source = f

        if source is not None and target is not None:
            def scan(file):
                with open(file, 'r', encoding='utf-8') as f:
                    i = 1
                    for line in f:
                        if text in line.lower():
                            print(f"{os.path.basename(s)} ({file}):{i} => {line}")
                        i += 1
            scan(source)
            scan(target)                
        else:
            print(f"Cannot find a source.txt and a target.txt in {s} ({dir}). Skipping...")

