"""OFFLINE DEMO. Stand in for the teacher endpoint so you can run the whole
pipeline without spending any API credits. It fabricates a *structured* signal
the student can actually learn: a random projection of L2-normalized bag-of-token
counts, so sentences that share subwords get similar vectors.

    python make_synthetic_teacher.py      # writes artifacts/{teacher_emb.npy,sentences.json}
    python train.py --epochs 40 --batch_size 16 --warmup 40

Replace this step with embed_teacher.py to distill the real model.
"""
import json
import numpy as np
from distill_lib import load_tokenizer

tok = load_tokenizer("artifacts/tokenizer.json")
sents = [l.strip() for l in open("data/sentences.txt", encoding="utf-8") if l.strip()]

V = tok.get_vocab_size()
bow = np.zeros((len(sents), V), dtype=np.float32)
for i, enc in enumerate(tok.encode_batch(sents)):
    for t in enc.ids:
        bow[i, t] += 1.0
bow /= (np.linalg.norm(bow, axis=1, keepdims=True) + 1e-8)

rng = np.random.default_rng(0)
proj = rng.standard_normal((V, 128)).astype(np.float32) / np.sqrt(V)
teacher = bow @ proj + 0.01 * rng.standard_normal((len(sents), 128)).astype(np.float32)

np.save("artifacts/teacher_emb.npy", teacher.astype(np.float32))
json.dump(sents, open("artifacts/sentences.json", "w"), ensure_ascii=False)
print("synthetic teacher:", teacher.shape, "| sentences:", len(sents))
