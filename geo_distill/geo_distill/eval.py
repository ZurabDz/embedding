"""Load the trained student and (a) report how well it matches the teacher and
(b) demo retrieval: given a query, show the student's nearest sentences.
"""
from __future__ import annotations

import json
import os

import numpy as np

from geo_distill import config as paths
from geo_distill.checkpoint import load_student
from geo_distill.data import load_tokenizer, metric_subset, val_split
from geo_distill.metrics import similarity_agreement
from geo_distill.model import embed_in_batches


def run(args) -> None:
    model, cfg, spec = load_student(args.model_dir)
    tok = load_tokenizer(cfg.tokenizer)
    spec.validate_tokenizer(tok)

    with open(args.sentences, encoding="utf-8") as f:
        sentences = json.load(f)
    teacher = np.load(args.teacher_emb).astype(np.float32)

    # Same stable content-based split as training, so we score on unseen sentences.
    _, val_idx = val_split(sentences, args.val_frac, args.seed)

    va_sents = [sentences[i] for i in val_idx]
    va_teacher = teacher[val_idx]
    # The same bounded subset the training loop scored, given the same corpus,
    # --val-frac, --seed and --val-metric-n — so this number is the one the
    # epoch lines were reporting, not a differently-drawn benchmark.
    keep = metric_subset(va_sents, args.val_metric_n, args.seed)
    if keep is not None:
        va_sents = [va_sents[i] for i in keep]
        va_teacher = va_teacher[keep]
    va_tokens, va_mask = spec.encode(tok, va_sents, cfg.max_len)
    va_student = embed_in_batches(model, va_tokens, va_mask)

    m = similarity_agreement(va_student, va_teacher)
    scored = (f"{len(va_sents)} of {len(val_idx)}" if keep is not None
              else f"all {len(val_idx)}")
    print(f"Held-out agreement with teacher ({scored} held-out sentences):")
    print(f"  Pearson  (similarity) : {m['pearson']:.3f}")
    print(f"  Spearman (similarity) : {m['spearman']:.3f}")
    print(f"  top-1 NN agreement    : {m['top1_nn_agreement']:.3f}")

    mean_path = os.path.join(args.model_dir, paths.TEACHER_MEAN)
    if cfg.center and os.path.isfile(mean_path):
        print(f"  (targets were mean-centered; the train mean is saved at "
              f"{mean_path} — subtract it from raw teacher vectors to map "
              f"them into the student's space)")

    if args.query:
        c_tokens, c_mask = spec.encode(tok, sentences, cfg.max_len)
        corpus_emb = embed_in_batches(model, c_tokens, c_mask)
        q_tokens, q_mask = spec.encode(tok, [args.query], cfg.max_len)
        q = embed_in_batches(model, q_tokens, q_mask)[0]
        # both are already L2-normalized by the model's output head
        scores = corpus_emb @ q
        top = np.argsort(-scores)[: args.topk]
        print(f"\nQuery: {args.query}")
        for rank, i in enumerate(top, 1):
            print(f"  {rank}. ({scores[i]:.3f}) {sentences[i][:90]}")
