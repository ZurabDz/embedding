"""Load the trained student and (a) report how well it matches the teacher and
(b) demo retrieval: given a query, show the student's nearest sentences.

Usage:
    python eval.py
    python eval.py --query "საქართველოს ისტორია" --topk 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import jax.numpy as jnp
from flax import nnx, serialization

from model import EmbeddingModel
from distill_lib import load_tokenizer, encode_batch, similarity_agreement, val_split


def load_student(config_path, params_path):
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("student_type") == "mlm":
        from mlm_student import build_mlm_student_from_config

        model = build_mlm_student_from_config(cfg)
    else:
        model = EmbeddingModel(
            vocab_size=cfg["vocab_size"], dim=cfg["dim"], depth=cfg["depth"],
            heads=cfg["heads"], mlp_dim=cfg["mlp_dim"], out_dim=cfg["out_dim"],
            max_len=cfg["max_len"], dropout=0.0, embed_dim=cfg.get("embed_dim"),
            rngs=nnx.Rngs(0))
    state = nnx.state(model, nnx.Param)  # template with the right structure/shapes
    with open(params_path, "rb") as f:
        pure = serialization.from_bytes(nnx.to_pure_dict(state), f.read())
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)
    return model, cfg


def embed(model, tok, texts, max_len, batch=256, encode=encode_batch):
    tokens, mask = encode(tok, texts, max_len)
    outs = []
    for i in range(0, len(texts), batch):
        e = model(jnp.asarray(tokens[i : i + batch]),
                  jnp.asarray(mask[i : i + batch]), deterministic=True)
        outs.append(np.asarray(e))
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="artifacts/student_config.json")
    ap.add_argument("--params", default="artifacts/student_params.msgpack")
    ap.add_argument("--sentences", default="artifacts/sentences.json")
    ap.add_argument("--teacher_emb", default="artifacts/teacher_emb.npy")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--query", default=None)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    model, cfg = load_student(args.config, args.params)
    tok = load_tokenizer(cfg["tokenizer"])
    if cfg.get("student_type") == "mlm":
        from mlm_student import encode_batch_mlm as _encode
    else:
        _encode = encode_batch

    with open(args.sentences, encoding="utf-8") as f:
        sentences = json.load(f)
    teacher = np.load(args.teacher_emb).astype(np.float32)

    # Same stable content-based split as training, so we score on unseen sentences.
    _, val_idx = val_split(sentences, args.val_frac, args.seed)

    va_sents = [sentences[i] for i in val_idx]
    va_teacher = teacher[val_idx]
    va_student = embed(model, tok, va_sents, cfg["max_len"], encode=_encode)

    m = similarity_agreement(va_student, va_teacher)
    print("Held-out agreement with teacher:")
    print(f"  Pearson  (similarity) : {m['pearson']:.3f}")
    print(f"  Spearman (similarity) : {m['spearman']:.3f}")
    print(f"  top-1 NN agreement    : {m['top1_nn_agreement']:.3f}")

    if args.query:
        corpus_emb = embed(model, tok, sentences, cfg["max_len"], encode=_encode)
        q = embed(model, tok, [args.query], cfg["max_len"], encode=_encode)[0]
        corpus_emb /= (np.linalg.norm(corpus_emb, axis=-1, keepdims=True) + 1e-12)
        q /= (np.linalg.norm(q) + 1e-12)
        scores = corpus_emb @ q
        top = np.argsort(-scores)[: args.topk]
        print(f"\nQuery: {args.query}")
        for rank, i in enumerate(top, 1):
            print(f"  {rank}. ({scores[i]:.3f}) {sentences[i][:90]}")


if __name__ == "__main__":
    main()
