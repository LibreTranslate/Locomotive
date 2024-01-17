import argparse
import io
import json
import os
import hashlib
import mmap

parser = argparse.ArgumentParser(description='Find text in data sources')
parser.add_argument('-c', '--config',
    type=str,
    default="model-config.json",
    help='Path to model-config.json. Default: %(default)s')
parser.add_argument('-t', '--text',
    type=str,
    help='Text to search. Default: %(default)s')
parser.add_argument('-e', '--exact',
    action="store_true",
    help='Exact match')

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
    if isinstance(s, dict):
        s = s["source"]

    if s.startswith("file://"):
        source_dir = s[7:]
    else:
        md5 = hashlib.md5(s.encode('utf-8')).hexdigest()
        source_dir = os.path.join(cache_dir, md5)

    if os.path.isdir(source_dir):
        source, target = None, None
        for f in [f.path for f in os.scandir(source_dir) if f.is_file()]:
            if "target" in f.lower():
                target = f
            elif f.lower().endswith(f".{config['to']['code']}"):
                target = f

            if "source" in f.lower():
                source = f
            elif f.lower().endswith(f".{config['from']['code']}"):
                source = f

        if source is not None and target is not None:
            def scan(file):
                with open(file, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    it = iter(mm.readline, b"")

                    i = 1
                    for line in it:
                        line_s = line.decode("utf-8")
                        if args.exact:
                            if text == line_s.lower().strip():
                                print(f"{os.path.basename(s)} ({file}):{i} => {line_s}")
                        else:
                            if text in line_s.lower():
                                print(f"{os.path.basename(s)} ({file}):{i} => {line_s}")
                        i += 1
                    mm.close()
            scan(source)
            scan(target)                
        else:
            print(f"Cannot find a source.txt and a target.txt in {s} ({dir}). Skipping...")
    else:
        print(f"Cannot access {source_dir}")
