import random
import os
import hashlib

def lines_count(file):
    with open(file, 'rb') as f:
        return sum(1 for i in f)

def merge_sources(sources, out_dir, max_eval_sentences=5000):
    merge_hash_file = os.path.join(out_dir, "merge-hash.txt")
    sources_hash = hashlib.md5("|".join(sorted([k for k in sources])).encode('utf-8')).hexdigest()

    if os.path.isfile(merge_hash_file):
        with open(merge_hash_file, "r", encoding="utf-8") as f:
            merge_hash = f.readline().strip()
            if merge_hash == sources_hash:
                print("Skipping merge sources, no changes in sources")
                return False

    with open(merge_hash_file, "w", encoding="utf-8") as f:
        f.write(sources_hash)

    print("Merging sources")

    data = []
    sentence_count = 0
    for k in sources:
        source = sources[k]['source']
        target = sources[k]['target']
        source_cnt = lines_count(source)
        target_cnt = lines_count(target)
        if source_cnt != target_cnt:
            print(f"Lines count is not equal between {source} ({source_cnt}) and {target} ({target_cnt}). Exiting...")
            exit(1)
        
        sentence_count += source_cnt
        print(f"New sentence count: {sentence_count}")

    if sentence_count * 0.2 < max_eval_sentences:
        max_eval_sentences = sentence_count * 0.2

    print(f"Training size: {sentence_count - max_eval_sentences}")
    print(f"Validation size: {max_eval_sentences}")

    os.makedirs(out_dir, exist_ok=True)
    train_count = 0
    eval_count = 0

    with open(os.path.join(out_dir, "src-val.txt"), "w", encoding="utf-8") as fsv, \
        open(os.path.join(out_dir, "tgt-val.txt"), "w", encoding="utf-8") as ftv, \
        open(os.path.join(out_dir, "src-train.txt"), "w", encoding="utf-8") as fst, \
        open(os.path.join(out_dir, "tgt-train.txt"), "w", encoding="utf-8") as ftt:

        for k in sources:
            source = sources[k]['source']
            target = sources[k]['target']

            print(f"Merging {source} - {target}")
            with open(source, "r", encoding="utf-8") as fs, \
                open(target, "r", encoding="utf-8") as ft:
                while True:
                    line_s = fs.readline()
                    line_t = ft.readline()
                    if not line_s or not line_t:
                        break

                    if (int(random.random() * sentence_count) < max_eval_sentences or sentence_count - (eval_count + train_count) < max_eval_sentences) and eval_count < max_eval_sentences:
                        fsv.write(line_s)
                        ftv.write(line_t)
                        eval_count += 1
                    else:
                        fst.write(line_s)
                        ftt.write(line_t)
                        train_count += 1
    print(f"Wrote train ({train_count}) and eval ({eval_count}) sets")

    return True
    

        
