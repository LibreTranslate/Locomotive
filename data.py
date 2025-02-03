import random
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
import mmap
import time
from collections import deque
import threading
from net import download
import filters as filter_funcs
import transforms as transform_funcs
import augmenters as augment_funcs
from removedup import rdup
from fastshuffle import file_shuffle_sample
from io import StringIO

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

def count_lines(file):
    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(file, "r",encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))


def sources_changed(sources, out_dir):
    merge_hash_file = os.path.join(out_dir, "merge-hash.txt")
    sources_hash = hashlib.md5("|".join(sorted([k for k in sources])).encode('utf-8')).hexdigest()

    if os.path.isfile(merge_hash_file):
        with open(merge_hash_file, "r", encoding="utf-8") as f:
            merge_hash = f.readline().strip()
            if merge_hash == sources_hash:
                print("No changes in sources")
                return False

    with open(merge_hash_file, "w", encoding="utf-8") as f:
        f.write(sources_hash)

    return True

def get_flores_dataset_path(dataset="dev"):
    if dataset != "dev" and dataset != "devtest":
        print(f"Invalid dataset {dataset} (must be either dev or devtest)")
        exit(1)
    current_dir = os.path.dirname(__file__)
    utils_dir = os.path.join(current_dir, "utils")

    flores_dataset = os.path.join(utils_dir, "flores200_dataset", dataset)
    if not os.path.isdir(flores_dataset):
        os.makedirs(utils_dir, exist_ok=True)
    
        # Download first
        print("Downloading flores200 dataset...")
        fname = os.path.join(utils_dir, "flores200.tar.gz")
        flores_url = "https://tinyurl.com/flores200dataset"
        download(flores_url, utils_dir, basename=os.path.basename(fname))

        import tarfile
        with tarfile.open(fname) as f:
            f.extractall(utils_dir)
        
        if os.path.isfile(fname):
            os.unlink(fname)

        if not os.path.isdir(flores_dataset):
            print(f"Cannot download flores200. Please manually download it from {flores_url} and place it in {utils_dir}")
            exit(1)

    return flores_dataset

def get_flores_file_path(lang_code, dataset="dev"):
    flores_dataset = get_flores_dataset_path(dataset)
    flores_file_path = os.path.join(flores_dataset, nllb_langs[lang_code] + f".{dataset}")
    return flores_file_path

def get_flores(lang_code, dataset="dev"):
    flores_dataset = get_flores_dataset_path(dataset)
    source = os.path.join(flores_dataset, nllb_langs[lang_code] + f".{dataset}")

    vs = [line.rstrip('\n') for line in open(source, encoding="utf-8")]
    return vs

def extract_flores_val(src_code, tgt_code, out_dir, dataset="dev"):
    src_f = os.path.join(out_dir, "src-val.txt")
    tgt_f = os.path.join(out_dir, "tgt-val.txt")
    if not os.path.isfile(src_f) or not os.path.isfile(tgt_f):
        src_val = get_flores(src_code, dataset)
        tgt_val = get_flores(tgt_code, dataset)
        with open(src_f, 'w', encoding='utf-8') as f:
            f.write("\n".join(src_val) + "\n")
        print(f"Wrote {src_f}")
        with open(tgt_f, 'w', encoding='utf-8') as f:
            f.write("\n".join(tgt_val) + "\n")
        print(f"Wrote {tgt_f}")
    
def merge_shuffle(sources, out_dir, max_eval_sentences=5000, remove_duplicates=True):
    if not sources_changed(sources, out_dir):
        return False

    lines = deque()
    total_count = 0

    src_train = os.path.join(out_dir, "src-train.txt")
    tgt_train = os.path.join(out_dir, "tgt-train.txt")
    for f in [src_train, tgt_train]:
        if os.path.isfile(f):
            os.unlink(f)

    def process_source(k):
        nonlocal total_count
        source = sources[k]['source']
        target = sources[k]['target']
        if sources[k]['weight'] is not None:
            return
        
        filters = []
        transforms = []
        augmenters = []

        for f in sources[k]['filters']:
            if isinstance(f, dict):
                func_name = list(f.keys())[0]
                def get_func(name):
                    kwargs = dict(f[name])
                    func = getattr(filter_funcs, name)
                    lam = lambda src, tgt: func(src, tgt, **kwargs)
                    lam.__name__ = name
                    lam.__args__ = kwargs
                    return lam 
                filters.append(get_func(func_name))
            else:
                filters.append(getattr(filter_funcs, f))
        
        for t in sources[k]['transforms']:
            if isinstance(t, dict):
                func_name = list(t.keys())[0]
                def get_func(name):
                    kwargs = dict(t[name])
                    func = getattr(transform_funcs, name)
                    lam = lambda src, tgt: func(src, tgt, **kwargs)
                    lam.__name__ = name
                    return lam
                transforms.append(get_func(func_name))
            else:
                transforms.append(getattr(transform_funcs, t))

        for a in sources[k]['augmenters']:
            if isinstance(a, dict):
                func_name = list(a.keys())[0]
                def get_func(name):
                    kwargs = dict(a[name])
                    func = getattr(augment_funcs, name)
                    lam = lambda src, tgt: func(src, tgt, **kwargs)
                    lam.__name__ = name
                    return lam
                augmenters.append(get_func(func_name))
            else:
                augmenters.append(getattr(augment_funcs, a))


        print(f"Reading {source} - {target}")
        filtered = {}
        count = 0
        augmented = 0
        line_no = 0
        begin_at = None
        stop_at = None
        line_count = None

        for f in filters:
            if f.__name__ == "top":
                line_count = count_lines(source)
                print(f"Line count: {line_count}")
                stop_at = int((f.__args__.get("percent", 100) / 100) * line_count)
                print(f"Stop at: {stop_at}")

            if f.__name__ == "excerpt":
                line_count = count_lines(source)
                print(f"Line count: {line_count}")
                begin_at = int((f.__args__.get("top_percentile", 100) / 100) * line_count)
                print(f"Excerpt will begin at line: {begin_at}")
                stop_at = int((f.__args__.get("bottom_percentile", 100) / 100) * line_count)
                print(f"Excerpt will end at line: {stop_at}")

        with open(source, "r+b") as src_fp, \
             open(target, "r+b") as tgt_fp:
            src_mm = mmap.mmap(src_fp.fileno(), 0)
            tgt_mm = mmap.mmap(tgt_fp.fileno(), 0)
            src_it = iter(src_mm.readline, b"")
            tgt_it = iter(tgt_mm.readline, b"")

            for src_line in src_it:
                #Exit after "stop_at" line if excerpt or top filter on
                if stop_at is not None and line_no > stop_at:
                    print(f"Finished collecting before line {line_no}")
                    break

                line_s = src_line.decode("utf-8").strip()
                line_t = next(tgt_it).decode("utf-8").strip()
                
                #Start counting every line ('count' excludes filtered lines)
                line_no += 1
                
                # Skip lines until begin_at if excerpt filter on
                if begin_at is not None and line_no < begin_at:
                    continue
                
                # Skip empty
                if len(line_s) == 0 or len(line_t) == 0:
                    continue
                
                skip = False
                for f in filters:
                    if f(line_s, line_t):
                        skip = True
                        filtered[f.__name__] = filtered.get(f.__name__, 0) + 1
                        break
                
                if skip:
                    continue

                count += 1

                for t in transforms:
                    line_s, line_t = t(line_s, line_t)
                
                lines.append((line_s + '\n', line_t + '\n'))

                for a in augmenters:
                    for a_src, a_tgt in a(line_s, line_t):
                        lines.append((a_src + '\n', a_tgt + '\n'))
                        augmented += 1
            src_mm.close()
            tgt_mm.close()

        print(filtered)
        print(f"Filtered {sum(filtered.values())} lines")
        total_count += count + augmented
        print(f"Added: {count + augmented} lines")
        print(f"New sentence count: {total_count}")

    finished = False

    def write_lines():
        with open(os.path.join(out_dir, "src.txt"), "w", encoding="utf-8") as src, \
             open(os.path.join(out_dir, "tgt.txt"), "w", encoding="utf-8") as tgt:
             while True:
                count = len(lines)
                if count > 0:
                    sbuf = StringIO()
                    tbuf = StringIO()

                    for x in range(count):
                        l = lines.popleft()
                        sbuf.write(l[0])
                        tbuf.write(l[1])

                    src.write(sbuf.getvalue())
                    tgt.write(tbuf.getvalue())
                elif finished:
                    break
                else:
                    time.sleep(0.2)

    writer = threading.Thread(target=write_lines)
    writer.start()

    # for s in sources:
    #     process_source(s)
    with ThreadPoolExecutor() as executor:
        executor.map(process_source, list(sources.keys()))    
    finished = True
    writer.join()

    if total_count * 0.2 < max_eval_sentences:
        max_eval_sentences = total_count * 0.2
    max_eval_sentences = int(max_eval_sentences)

    if total_count == 0:
        print("No sources merged")
        return

    print(f"Training size: {total_count - max_eval_sentences}")
    print(f"Validation size: {max_eval_sentences}")

    print("Writing shuffled sets")
    os.makedirs(out_dir, exist_ok=True)

    src, tgt, src_sample, tgt_sample = file_shuffle_sample(os.path.join(out_dir, "src.txt"), os.path.join(out_dir, "tgt.txt"), max_eval_sentences)
    os.rename(src, src_train)
    os.rename(tgt, tgt_train)
    os.rename(src_sample, os.path.join(out_dir, "src-val.txt"))
    os.rename(tgt_sample, os.path.join(out_dir, "tgt-val.txt"))
    
    if remove_duplicates:
        print("Removing duplicates")
        src, tgt, removed = rdup(src_train, tgt_train)
        print(f"Removed {removed} lines")
        os.unlink(src_train)
        os.unlink(tgt_train)
        os.rename(src, src_train)
        os.rename(tgt, tgt_train)

    os.unlink(os.path.join(out_dir, "src.txt"))
    os.unlink(os.path.join(out_dir, "tgt.txt"))

    return True
