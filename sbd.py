import os
import shutil
import stanza
import spacy
from spacy.cli import download

'''
This module deals with sentence boundary detection packages:
As per last Argos PR, you may have a language-specific Spacy package (light and efficient);
a Stanza package is useful when the former requires a third-party dependency or does not exist;
then, when none of the above is available, Argos uses cached/shared spacy multilingual.
'''

def package_sbd(run_dir, lang_code):
    stanza_dir = os.path.join(run_dir, "stanza")
    spacy_dir = os.path.join(run_dir, "spacy")
    spacy_utils = os.path.join(os.path.dirname(__file__), "utils", "spacy")
    spacy_lang_utils = os.path.join(os.path.dirname(__file__), "utils", f'spacy_{lang_code}')

    # First, a list of ===third-dependency-free=== spacy packages (feel free to update)
    # ja, ko, ru, th, uk, vi & zh require external dependencies, each package+dependencies > 70Mo (stanza better)
    spacy_models = {
        "ca": "ca_core_news_sm",
        "da": "da_core_news_sm",
        "de": "de_core_news_sm",
        "el": "el_core_news_sm",
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fi": "fi_core_news_sm",
        "fr": "fr_core_news_sm",
        "hr": "hr_core_news_sm",
        "it": "it_core_news_sm",
        "lt": "lt_core_news_sm",
        "mk": "mk_mk_news_sm",
        "nb": "nb_core_news_sm",
        "nl": "nl_core_news_sm",
        "no": "nb_core_news_sm",
        "pl": "pl_core_news_sm",
        "pt": "pt_core_news_sm",
        "ro": "ro_core_news_sm",
        "sl": "sl_core_news_sm",
        'sv': 'sv_core_news_sm',
    }
    '''
    Writing a dummy file is a necessary evil: a symlink will package spacy xx at end of training
    Plus, on Windows, it requires activating Settings/System/Developer mode, that's not always policy.
    '''
    if os.path.isfile(os.path.join(spacy_dir, "senter", "model")):
        print('spaCy library ready')
        return spacy_dir
    elif os.path.isdir(os.path.join(stanza_dir, lang_code)):
        print('Stanza library ready')
        return stanza_dir
    elif os.path.isfile(spacy_dir): #Dummy file
        print('spaCy multilingual library in utils.')
        return spacy_dir
    else:
        # There may be a third-party-dependency-free Spacy library for this language
        if lang_code in spacy_models:
            try:
                # Download once (verbosissimo) and use files from utils after
                if not os.path.isfile(os.path.join(spacy_lang_utils, "senter", "model")):
                    spacy.cli.download(spacy_models[lang_code])
                    # Make it as lightweight as possible by excluding all unnecessary functions
                    nlp = spacy.load(spacy_models[lang_code], exclude=("parser", "tagger", "lemmatizer", "morphologizer",
                                                                      "ner", "tok2vec", "attribute_ruler"))
                    nlp.to_disk(spacy_lang_utils)
                    print('Language-specific Spacy model written to utils.')
                shutil.copytree(spacy_lang_utils,spacy_dir)
                print('Packaged spaCy model ready.')
                return spacy_dir
            except Exception as e:
                print(f'{str(e)}.')
        # Then there is Stanza: we'll take legacy download, strip the while True loop, it doesn't seem to add much
        else:
            try:
                os.makedirs(stanza_dir, exist_ok=True)
                stanza.download(lang_code, dir=stanza_dir, processors="tokenize")
                return stanza_dir
            except Exception as e:
                # Then spacy multilingual
                if str(e).startswith('Unsupported language'):
                    print(f'Stanza said: " {str(e)}"; hence, will use spacy multilingual.')
                    os.remove(os.path.join(stanza_dir, "resources.json"))
                    os.rmdir(stanza_dir)
                    # Avoid verbose download
                    if not os.path.isfile(os.path.join(spacy_utils, "senter", "model")):
                        try:
                            spacy.cli.download("xx_sent_ud_sm")
                            print(f'Downloaded multilingual spaCy model. Loading and writing to utils.')
                            nlp = spacy.load("xx_sent_ud_sm", exclude="parser")
                            nlp.to_disk(spacy_utils)
                            print('Multilingual spaCy model written to utils.')
                        except Exception as e:
                            print(f'{str(e)}.')
                            exit(1)
                    # Writes a dummy file and returns None, otherwise will package the cached model.
                    with open(spacy_dir, "w", encoding="utf-8") as dummy:
                        dummy.write('spaCy multilingual in utils.')
                    print('Multilingual spaCy model in utils: wrote dummy file.')
                    return spacy_dir
                else:
                    print(f'{str(e)}.')
                    exit(1)
