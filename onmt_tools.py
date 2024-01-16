import torch
import sys
import math
from onmt.constants import DefaultTokens
from onmt.transforms import register_transform

# From: https://github.com/OpenNMT/OpenNMT-py
# MIT licensed
# Copyright (c) 2017-Present OpenNMT

def average_models(model_files, output, fp32=False):
    vocab = None
    opt = None
    avg_model = None
    avg_generator = None

    for i, model_file in enumerate(model_files):
        m = torch.load(model_file, map_location='cpu')
        model_weights = m['model']
        generator_weights = m['generator']

        if fp32:
            for k, v in model_weights.items():
                model_weights[k] = v.float()
            for k, v in generator_weights.items():
                generator_weights[k] = v.float()

        if i == 0:
            vocab, opt = m['vocab'], m['opt']
            avg_model = model_weights
            avg_generator = generator_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)

            for (k, v) in avg_generator.items():
                avg_generator[k].mul_(i).add_(generator_weights[k]).div_(i + 1)

    final = {"vocab": vocab, "opt": opt, "optim": None,
             "generator": avg_generator, "model": avg_model}
    
    torch.save(final, output)


def sp_vocab_to_onmt_vocab(sp_vocab, onmt_vocab):
    print(f"Converting {sp_vocab}")
    with open(sp_vocab, 'r', encoding="utf-8") as fin:
        with open(onmt_vocab, 'wb') as fout:
            OMIT = (DefaultTokens.UNK, DefaultTokens.BOS, DefaultTokens.EOS)
            for line in fin:
                try:
                    token_and_freq = line.rstrip("\n").split(None, 1)
                    if len(token_and_freq) != 2:
                        continue
                    w, c = token_and_freq
                    if w in OMIT:
                        continue
                    c = math.exp(float(c)) * 1000000
                    c = int(c) + 1
                    fout.write(f'{w}\t{c}\n'.encode("utf-8"))
                except Exception as e:
                    print(str(e))

    print(f"Wrote {onmt_vocab}")
