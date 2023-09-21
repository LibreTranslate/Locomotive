import random
import os

def merge_shuffle(sources, out_dir, max_eval_sentences=5000):
    data = []
    print(sources)
    for k in sources:
        source = sources[k]['source']
        target = sources[k]['target']
        
        print(f"Reading {os.path.basename(source)} - {os.path.basename(target)}")
        with open(source, "r", encoding="utf-8") as fs, \
             open(target, "r", encoding="utf-8") as ft:
             while True:
                line_s = fs.readline()
                line_t = ft.readline()
                if not line_s or not line_t:
                    break
                data.append((line_s, line_t))
        print(f"Sentences: {len(data)}")
    
    print("Shuffling")
    random.shuffle(data)

    if len(data) * 0.2 < max_eval_sentences:
        max_eval_sentences = len(data) * 0.2

    print(f"Training size: {len(data) - max_eval_sentences}")
    print(f"Validation size: {max_eval_sentences}")

    # TODO: can this be faster?
    print("Writing sets")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with open(os.path.join(out_dir, "src-val.txt"), "w", encoding="utf-8") as fsv, \
        open(os.path.join(out_dir, "tgt-val.txt"), "w", encoding="utf-8") as ftv, \
        open(os.path.join(out_dir, "src-train.txt"), "w", encoding="utf-8") as fst, \
        open(os.path.join(out_dir, "tgt-train.txt"), "w", encoding="utf-8") as ftt:
        for source, target in data:
            if count < max_eval_sentences:
                fsv.write(source)
                ftv.write(target)
            else:
                fst.write(source)
                ftt.write(target)

            count += 1
    

        
