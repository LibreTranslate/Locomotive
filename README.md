# Locomotive

Easy to use, cross-platform toolkit to train [argos-translate](https://github.com/argosopentech/argos-translate) models, which can be used by [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate) 🚂

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
        "http://data.argosopentech.com/data-ccaligned-en_es.argosdata",
    ]   
}
```

Note you can specify both local folders (using the `file://` prefix) or internet URLs to .zip archives (using the `http://` or `https://` prefix).

Then run:

```bash
python train.py --config config.json
```

Training can take a while and depending on the size of datasets can require a graphics card with lots of memory.

The output will be saved in `run/[model]/translate-[from]_[to]-[version].argosmodel`.

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

The model is generated using sensible default values. You can override the [default configuration](https://github.com/LibreTranslate/Locomotive/blob/main/train.py#L199) by adding values directly to your `config.json`. For example, to use a smaller dictionary size, add a `vocab_size` key in `config.json`:

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
        "http://data.argosopentech.com/data-ccaligned-en_es.argosdata",
    ],
    "vocab_size": 30000
}
```

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

## Contribute

Want to share your model with the world? Post it on [community.libretranslate.com](https://community.libretranslate.com) and we'll include in future releases of LibreTranslate. Make sure to share both a forward and reverse model (e.g. `en => es` and `es => en`), otherwise we won't be able to include it in the model repository.

We also welcome contributions to Locomotive! Just open a pull request.

## Use with LibreTranslate

To install the resulting .argosmodel file, locate the `~/.local/share/argos-translate/packages` folder. On Windows this is the `%userprofile%\.local\share\argos-translate\packages` folder. Then create a `[from-code]_[to-code]` folder (e.g. `en_es`). If it already exists, delete or move it.

Extract the contents of the .argosmodel file (which is just a .zip file, you might need to change the extension to .zip) into this folder. Then restart LibreTranslate.

## Credits

In no particular order, we'd like to thank:

 * [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
 * [SentencePiece](https://github.com/google/sentencepiece)
 * [Stanza](https://github.com/stanfordnlp/stanza)
 * [argos-train](https://github.com/argosopentech/argos-train)

For making Locomotive possible.

## License

AGPLv3
