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

# The list is ordered according to lang_codes found on OPUS
# Some dialects and scripts listed in flores200 have not been mapped due to lack of resource on OPUS
nllb_langs = { #Name found on opus as comments, if [] other known alias, () several charsets
    "ace":"ace_Latn", #Achinese
    "af":"afr_Latn", #Afrikaans
    "ak":"aka_Latn", #Akan
    "am":"amh_Ethi", #Amharic
    "ar":"arb_Arab", #Arabic
    "as":"asm_Beng", #Assamese
    "ast":"ast_Latn", #Asturian
    "ay":"ayr_Latn", #Aymara
    "az":"azj_Latn", #Azerbaijani (Latin)
    "azb":"azb_Arab", #Azerbaijani (Arabic)
    "ba":"bak_Cyrl", #Bashkir
    "bm":"bam_Latn", #Bambara
    "ban":"ban_Latn", #Balinese
    "be":"bel_Cyrl", #Belarusian
    "bem":"bem_Latn", #Bemba
    "bn":"ben_Beng", #Bangla [Bengalese]
    "bho":"bho_Deva", #Bhojpuri [Bihari]
    "bjn":"bjn_Latn", #Banjar
    "bo":"bod_Tibt", #Tibetan
    "bs":"bos_Latn", #Bosnian
    "bug":"bug_Latn", #Buginese
    "bg":"bul_Cyrl", #Bulgarian
    "ca":"cat_Latn", #Catalan
    "ceb":"ceb_Latn", #Cebuano
    "cs":"ces_Latn", #Czech
    "cjk":"cjk_Latn", #Chokwe
    "ckb":"ckb_Arab", #Central Kurdish [Sorani], no NLLB resource
    "crh":"crh_Latn", #Crimean Tatar
    "cy":"cym_Latn", #Welsh
    "da":"dan_Latn", #Danish
    "de":"deu_Latn", #German
    "dik":"dik_Latn", #Dinka
    "dyu":"dyu_Latn", #Dyula
    "dz":"dzo_Tibt", #Dzongkha
    "el":"ell_Grek", #Greek
    "en":"eng_Latn", #English
    "eo":"epo_Latn", #Esperanto
    "es":"spa_Latn", #Spanish
    "et":"est_Latn", #Estonian
    "eu":"eus_Latn", #Basque
    "ee":"ewe_Latn", #Ewe
    "fa":"pes_Arab", #Persian
    "fi":"fin_Latn", #Finnish
    "ff":"fuv_Latn", #Fula [Nigerian Fulfulde]
    "fj":"fij_Latn", #Fijian
    "fo":"fao_Latn", #Faroese
    "fon":"fon_Latn", #Fon
    "fr":"fra_Latn", #French
    "fur":"fur_Latn", #Friulian
    "gd":"gla_Latn", #Scottish Gaelic
    "ga":"gle_Latn", #Irish
    "gl":"glg_Latn", #Galician
    "gn":"grn_Latn", #Guarani
    "gu":"guj_Gujr", #Gujarati
    "ht":"hat_Latn", #Haitian Creole
    "ha":"hau_Latn", #Hausa
    "he":"heb_Hebr", #Hebrew
    "hi":"hin_Deva", #Hindi
    "hne":"hne_Deva", #Chhattishgarhi
    "hr":"hrv_Latn", #Croatian
    "hu":"hun_Latn", #Hungarian
    "hy":"hye_Armn", #Armenian
    "ig":"ibo_Latn", #Igbo
    "ilo":"ilo_Latn", #Iloko [Ilocano]
    "id":"ind_Latn", #Indonesian
    "is":"isl_Latn", #Icelandic
    "it":"ita_Latn", #Italian
    "jv":"jav_Latn", #Javanese
    "ja":"jpn_Jpan", #Japanese
    "kn":"kan_Knda", #Kannada
    "ka":"kat_Geor", #Georgian [Kartvelian]
    "kab":"kab_latn", #Kabyle [Berber/Tamazight] (Latin)
    "kac":"kac_Latn", #Kachin
    "kam":"kam_Latn", #Kamba
    "kbp":"kbp_Latn", #Kabiye
    "kea":"kea_Latn", #Kabuverdianu
    "kg":"kon_Latn", #Kongo
    "ki":"kik_Latn", #Kikuyu
    "kk":"kaz_Cyrl", #Kazakh
    "km":"khm_Khmr", #Khmer
    "kmb":"kmb_Latn", #Kimbundu
    "ko":"kor_Hang", #Korean
    "ks-Arab":"kas_Arab", #Kashmiri (Arabic)
    "ks-Deva":"kas_Deva", #Kashmiri (Devanagari)
    "kr-Arab":"knc_Arab", #Kanuri (Arabic)
    "kr-Latn":"knc_Latn", #Kanuri (Latin)
    "ku":"kmr_Latn", #Kurdish [Kurmandji], small corpora, NLLB resource under "ku-Latn"
    "ku-Latn":"kmr_Latn", #Kurdish, NLLB resource in ku-en (also NLLB ku-ar in kmr_Arab under ku-Arab)
    "ky":"kir_Cyrl", #Kyrgyz
    "lb":"ltz_Latn", #Luxembourgish
    "lg":"lug_Latn", #Ganda
    "li":"lim_Latn", #Limburgish
    "lij":"lij_Latn", #Ligurian
    "lmo":"lmo_Latn", #Lombard
    "ln":"lin_Latn", #Lingala
    "lo":"lao_Laoo", #Laotian
    "lt":"lit_Latn", #Lithuanian
    "ltg":"ltg_Latn", #Latgalian
    "lua":"lua_Latn", #Luba-Lulua
    "luo":"luo_Latn", #Luo
    "lus":"lus_Latn", #Mizo
    "lv":"lvs_Latn", #Latvian
    "mag":"mag_Deva", #Magahi
    "mai":"mai_Deva", #Maithili
    "min-Arab":"min_Arab", #Minangkabau (Arabic), no NLLB resource
    "min":"min_Latn", #Minangkabau
    "mg":"plt_Latn", #Malagasy
    "mi":"mri_Latn", #Maori
    "mk":"mkd_Cyrl", #Macedonian
    "ml":"mal_Mlym", #Malayalam
    "mn":"khk_Cyrl", #Mongolian
    "mni":"mni_Beng", #Manipuri
    "mos": "mos_Latn", #Mossi
    "mr":"mar_Deva", #Marathi
    "ms":"zsm_Latn", #Malay
    "mt":"mlt_Latn", #Maltese
    "my":"mya_Mymr", #Burmese
    "nb":"nob_Latn", #Norwegian Bokmål, PAracrawl/HPLT/ELRC resources
    "ne":"npi_Deva", #Nepalese
    "nl":"nld_Latn", #Dutch
    "nn":"nno_Latn", #Norwegian Nynorsk, no NLLB resource
    "no":"nob_Latn", #Norwegian [Bokmål]
    "nso":"nso_Latn", #Northern Sotho
    "nus":"nus_Latn", #Nuer
    "ny":"nya_Latn", #Nyanja [Chichewa]
    "oc":"oci_Latn", #Occitan
    "om":"gaz_Latn", #Oromo
    "or":"ory_Orya", #Odia [Oriya/Odiya]
    "pag":"pag_Latn", #Pangasinan
    "pa":"pan_Guru", #Panjabi
    "pap":"pap_Latn", #Papiamento
    "pl":"pol_Latn", #Polish
    "prs":"prs_Arab", #Dari
    "ps":"pbt_Arab", #Pashto
    "pt":"por_Latn", #Portuguese
    "qu":"quy_Latn", #Quechua
    "rn":"run_Latn", #Rundi [Kirundi]
    "ro":"ron_Latn", #Romanian
    "ru":"rus_Cyrl", #Russian
    "rw":"kin_Latn", #Kinyarwanda
    "sa":"san_Deva", #Sanskrit
    "sat":"sat_Olck", #Santali
    "sc":"srd_Latn", #Sardinian
    "scn":"scn_Latn", #Sicilian
    "sd":"snd_Arab", #Sindhi
    "sg":"sag_Latn", #Sango
    "shn":"shn_Mymr", #Shan
    "si":"sin_Sinh", #Sinhalese
    "sk":"slk_Latn", #Slovak
    "sl":"slv_Latn", #Slovenian
    "sm":"smo_Latn", #Samoan
    "sn":"sna_Latn", #Shona
    "so":"som_Latn", #Somali
    "sq":"als_Latn", #Albanian
    "sr":"srp_Cyrl", #Serbian
    "ss":"ssw_Latn", #Swati [siSwati]
    "st":"sot_Latn", #Sotho [seSotho]
    "su":"sun_Latn", #Sundanese
    "sv":"swe_Latn", #Swedish
    "sw":"swh_Latn", #Swahili
    "szl":"szl_Latn", #Silesian
    "ta":"tam_Taml", #Tamil
    "taq":"taq_Latn", #Tamasheq [Tuareg]
    "te":"tel_Telu", #Telugu
    "tg":"tgk_Cyrl", #Tajik
    "th":"tha_Thai", #Thai
    "ti":"tir_Ethi", #Tigrinya
    "tk":"tuk_Latn", #Turkmen
    "tl":"tgl_Latn", #Filipino [Tagalog]
    "tn":"tsn_Latn", #Tswana
    "tpi":"tpi_Latn", #Tok Pisin
    "tr":"tur_Latn", #Turkish
    "ts":"tso_Latn", #Tsonga
    "tt":"tat_Cyrl", #Tatar
    "tum":"tum_Latn", #Tumbuka
    "tw":"twi_Latn", #Twi/Akan
    "tzm":"tzm_Tfng", #Central Atlas Tamazight (Tifinagh)
    "ug":"uig_Arab", #Uighur
    "uk":"ukr_Cyrl", #Ukrainian
    "umb":"umb_Latn", #Umbundu [Kimbundu]
    "ur":"urd_Arab", #Urdu
    "uz":"uzn_Latn", #Uzbek
    "vec":"vec_Latn", #Venetian
    "vi":"vie_Latn", #Vietnamese
    "war":"war_Latn", #Waray
    "wo":"wol_Latn", #Wolof
    "xh":"xho_Latn", #Xhosa
    "yi":"ydd_Hebr", #Yiddish
    "yo":"yor_Latn", #Yoruba
    "yue":"yue_Hant", #Yue chinese [Cantonese], no NLLB resource
    "zh-CN":"zho_Hans", #Chinese (Simplified), no NLLB resource
    "zh":"zho_Hans", #Chinese (Simplified)
    "zh-TW":"zho_Hant", #Chinese (Traditional)
    "zu":"zul_Latn" #Zulu
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
    # Swap the definitions to allow looking for translation memoirs
    current_dir = os.path.dirname(__file__)
    utils_dir = os.path.join(current_dir, "utils")
    flores_dataset = os.path.join(utils_dir, "flores200_dataset", dataset)

    if dataset != "dev" and dataset != "devtest" and not os.path.isdir(flores_dataset):
        print(f"Invalid dataset {dataset} (must be either dev, devtest), or a valid translation memoir.")
        exit(1)

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
    if not os.path.isfile(source) and dataset != "dev" and dataset != "devtest":
        print(f"The memoir version for {lang_code} is missing or should be renamed {nllb_langs[lang_code]}.{dataset}")
        exit(1)
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

    fasttext_path = None
    for s in sources:
        for f in sources[s]['filters']:
            if f == "fast_lang":
                fasttext_path = os.path.join(os.path.dirname(__file__),"utils","fasttext","lid.176.bin")
    if fasttext_path is not None:
        import fasttext
        flmodel = fasttext.load_model(fasttext_path)

    def process_source(k):
        nonlocal total_count
        source = sources[k]['source']
        src_lang = sources[k]['from']
        src_chset = nllb_langs[src_lang][-4:]
        target = sources[k]['target']
        tgt_lang = sources[k]['to']
        tgt_chset = nllb_langs[tgt_lang][-4:]
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
                    if name == 'limit_latin_chars':
                        kwargs = {"s_chset": src_chset, "t_chset": tgt_chset, **kwargs}
                    lam = lambda src, tgt: func(src, tgt, **kwargs)
                    lam.__name__ = name
                    lam.__args__ = kwargs
                    return lam 
                filters.append(get_func(func_name))
            elif f == "fast_lang":
                def fastlang_func():
                    kwargs = {"s_lang": src_lang, "t_lang": tgt_lang, "model": flmodel}
                    func = getattr(filter_funcs, "fast_lang")
                    lam = lambda src, tgt: func(src, tgt, **kwargs)
                    lam.__name__ = "fast_lang"
                    lam.__args__ = kwargs
                    return lam
                filters.append(fastlang_func())
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
