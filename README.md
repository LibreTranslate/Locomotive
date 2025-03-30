# Locomotive

Easy to use, cross-platform toolkit to train [argos-translate](https://github.com/argosopentech/argos-translate) models, which can be used by [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) 🚂

It can also [convert pre-trained Opus-MT models](#convert-helsinki-nlp-opus-mt-models).

## Requirements

 * Python >= 3.8
 * NVIDIA CUDA graphics card (not required, but highly recommended)

## Install

```bash
git clone https://github.com/LibreTranslate/Locomotive --depth 1
cd Locomotive
pip install -r requirements.txt
```

## Background

Language models can be trained by providing lots of example translations from a source language to a target language. All you need to get started is a set of two files (`source` and `target`). The source file containing sentences written in the source language and a corresponding file with sentences written in the target language.

For example:

`source.txt`:

```
Hello
I'm a train!
Goodbye
```

`target.txt`:

```
Hola
¡Soy un tren!
Adiós
```

You'll need a few million sentences to train decent models, and at least ~100k sentences to get some results. [OPUS](https://opus.nlpl.eu/) has a good collection of datasets to get started. You can also use any of the data sources listed on the [argos-train index](https://github.com/argosopentech/argos-train/blob/master/data-index.json). Also check [NLLU](https://nllu.libretranslate.com).

## Usage

Place `source.txt` and `target.txt` files in a folder (e.g. `mydataset-en_es`) of your choice:

```bash
mydataset-en_es/
├── source.txt
└── target.txt
```

Create a `config.json` file specifying your sources:

```json
{
    "from": {
        "name": "English",
        "code": "en"
    },
    "to": {
        "name": "Spanish",
        "code": "es"
    },
    "version": "1.0",
    "sources": [
        "file://D:\\path\\to\\mydataset-en_es",
        "opus://Ubuntu",
        "http://data.argosopentech.com/data-ccaligned-en_es.argosdata"
    ]   
}
```

Note you can specify, local folders (using the `file://` prefix), internet URLs to .zip archives (using the `http://` or `https://` prefix) or [OPUS](https://opus.nlpl.eu/) datasets (using the `opus://` prefix). For a complete list of OPUS datasets, see [OPUS.md](OPUS.md) and note that they are case-sensitive.

Then run:

```bash
python train.py --config config.json
```

Training can take a while and depending on the size of datasets can require a graphics card with lots of memory.

The output will be saved in `run/[model]/translate-[from]_[to]-[version].argosmodel`.

### Running out of memory

If you're running out of CUDA memory, decrease the `batch_size` parameter, which by default is set to `8192`:

```json
{
    "from": {
        "name": "English",
        "code": "en"
    },
    "to": {
        "name": "Spanish",
        "code": "es"
    },
    "version": "1.0",
    "sources": [
        "file://D:\\path\\to\\mydataset-en_es",
        "http://data.argosopentech.com/data-ccaligned-en_es.argosdata"
    ],
    "batch_size": 2048
}
```

### Reverse Training

Once you have trained a model from `source => target`, you can easily train a reverse model `target => source` model by passing `--reverse`:

```bash
python train.py --config config.json --reverse
```

### Tensorboard

TensorBoard allows tracking and visualizing metrics such as loss and accuracy, visualizing the model graph and other features. You can enable tensorboard with the `--tensorboard` option:

```bash
python train.py --config config.json --tensorboard
```

### Tuning

The model is generated using sensible default values. You can override the [default configuration](https://github.com/LibreTranslate/Locomotive/blob/main/train.py#L276) by adding values directly to your `config.json`. For example, to use a smaller dictionary size, add a `vocab_size` key in `config.json`:

```json
{
    "from": {
        "name": "English",
        "code": "en"
    },
    "to": {
        "name": "Spanish",
        "code": "es"
    },
    "version": "1.0",
    "sources": [
        "file://D:\\path\\to\\mydataset-en_es",
        "http://data.argosopentech.com/data-ccaligned-en_es.argosdata"
    ],
    "vocab_size": 30000
}
```

### Using Filters and Transforms

Locomotive provides various [filters](https://github.com/LibreTranslate/Locomotive/blob/main/FILTERS.md), [transforms](https://github.com/LibreTranslate/Locomotive/blob/main/TRANSFORMS.md) and [augmenters](https://github.com/LibreTranslate/Locomotive/blob/main/AUGMENTERS.md)  which can be used to dynamically cleanup, modify and augment the input sources before training: 

```json
{
    "filters": [
        "duplicates", 
        {"source_target_ratio": {"min": 0.6, "max": 1.5}}
    ],
    "transforms":[
        "remove_unpaired_quotes_and_brackets"
    ],
    "augmenters":[
        "single_word_punctuation"
    ],
    "sources": [
        {
            "source": "file://D:\\path\\to\\mydataset-en_es", 
            "filters": [
                {"char_length": {"min": 20}}
            ]
        }
    ]
}
```

Filters, transforms and augmenters can be specified globally (applied to all sources) as well as per-source (applied only to the specified source).

## Using Weights

It's possible to specify weights for each source, for example, it's possible to instruct the training to use less samples for certain datasets:

```json
{
    "sources": [
        {"source": "file://D:\\path\\to\\mydataset-en_es", "weight": 1},
        {"source": "http://data.argosopentech.com/data-ccaligned-en_es.argosdata", "weight": 5}
    ]
}
```

In the example above, 1 sample will be taken from mydataset and 5 will will be taken from CCAligned.

Specifying weights disables filtering, transformations and augmentations. The datasets are used as-is. No merging or shuffling is performed either. A weight of 1 can be used to instruct Locomotive to not preprocess a source.

## Evaluate

You can evaluate the model by running:

```bash
python eval.py --config config.json
Starting interactive mode
(en)> Hello!
(es)> ¡Hola!
(en)>
```

You can also compute [BLEU](https://en.wikipedia.org/wiki/BLEU) scores against the [flores200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) dataset for the model by running:

```bash
python eval.py --config config.json --bleu
BLEU score: 45.12354
```
You can also compute [COMET-22](https://github.com/Unbabel/COMET) scores against the [flores200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) dataset for the model by running:
```bash
python eval.py --config config.json --comet --bleu
Fetching 5 files: 100%|██████████████████████████████████████████████████████████████████████████| 5/5 [00:00<?, ?it/s]
....\pytorch_lightning\core\saving.py:195: Found keys that are not in the model state dict but in the checkpoint: ['encoder.model.embeddings.position_ids']
....\flores200devtest-.....evl score: 0.8003
BLEU score: 27.44794
```
You can also install further COMET models [AfriCOMET](https://github.com/masakhane-io/africomet) and update the script to use them.

<mark>Unzip the necessary Argos package(s) to the "utils" directory before using arg --pivot to evaluate a composite translation</mark> (i.e. to or from a third language).

## Convert Helsinki-NLP OPUS MT models

Locomotive provides a convenient script to convert pre-trained models from [OPUS-MT](https://github.com/Helsinki-NLP/OPUS-MT-train) to make them compatible with LibreTranslate:

```bash
python opus_mt_convert.py -s en -t it
```

This will attempt to automatically find/download the OPUS-MT's model archive from https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models/ or https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/. This doesn't always work, and will not always pick the best model. You can specify a model archive manually by using the `--model-url` parameter:

Some models also need a beginning of sentence (BOS) token for the model to work. You can specify a BOS token by using the `--bos` parameter:

```
python opus_mt_convert.py -s en -t vi --model-url https://object.pouta.csc.fi/Tatoeba-MT-models/eng-vie/opus+bt-2021-04-10.zip --bos ">>vie<<"
```

To run evaluation:

```bash
python eval.py --config run/en_it-opus_1.0/config.json
```

The script is experimental. If you find issues, feel free to open a pull request!

### Known Limitations

Some models fail to execute with int8 quantization. If you get a lot of repeated words, try to set `-q float32` to keep full precision.

## Contribute

Want to share your model with the world? Post it on [community.libretranslate.com](https://community.libretranslate.com) and we'll include in future releases of LibreTranslate. Make sure to share both a forward and reverse model (e.g. `en => es` and `es => en`), otherwise we won't be able to include it in the model repository.

We also welcome contributions to Locomotive! Just open a pull request.

## Use with LibreTranslate

To install the resulting .argosmodel file, locate the `~/.local/share/argos-translate/packages` folder. On Windows this is the `%userprofile%\.local\share\argos-translate\packages` folder. Then create a `[from-code]_[to-code]` folder (e.g. `en_es`). If it already exists, delete or move it.

Extract the contents of the .argosmodel file (which is just a .zip file, you might need to change the extension to .zip) into this folder. Then restart LibreTranslate.

You can also install .argosmodel packages from Python:
```
import pathlib
import argostranslate.package
package_path = pathlib.Path("/root/translate-en_it-2_0.argosmodel")
argostranslate.package.install_from_path(package_path)
```

## Credits

In no particular order, we'd like to thank:

 * [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
 * [SentencePiece](https://github.com/google/sentencepiece)
 * [Stanza](https://github.com/stanfordnlp/stanza)
 * [argos-train](https://github.com/argosopentech/argos-train)
 * [OPUS](https://opus.nlpl.eu)

For making Locomotive possible.

## License

AGPLv3
