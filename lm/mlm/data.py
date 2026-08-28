"""Corpus streaming, tokenisation, windowing, train/val splitting, and the
grain input pipeline — everything between raw text and a masked batch."""

import json
import os
import time
from typing import Iterable, Iterator

import grain
import numpy as np

from mlm import hub
from mlm.config import MIN_VAL_BATCHES, MIN_WINDOW_TOKENS, N_SPECIAL
from mlm.encoding import cls_row
from mlm.masking import mask_batch
from mlm.progress import bar
from mlm.tokenizer import resolve_tokenizer, train_bpe

SMOKE_SENTENCES = [
    "ქართული ენა არის ქართველური ენების ჯგუფის ყველაზე გავრცელებული ენა.",
    "თბილისი საქართველოს დედაქალაქი და უდიდესი ქალაქია მდინარე მტკვრის ნაპირზე.",
    "ქართული დამწერლობა სამი ანბანისგან შედგება: ასომთავრული, ნუსხური და მხედრული.",
    "მთაწმინდა თბილისის ერთ-ერთი უბანია, სადაც მდებარეობს ტელევიზიის ანძა.",
    "ენის მოდელის წინასწარი ვარჯიში საჭიროებს დიდი რაოდენობის ტექსტურ მონაცემებს.",
]


def iter_texts(dataset: str, config: str, n_docs: int, smoke: bool) -> Iterator[str]:
    """Up to n_docs *usable* documents from the stream.

    Counts what it yields, not what it scans: rows shorter than 200 characters
    are filtered before counting, so --docs means documents that actually enter
    the corpus and every progress bar totalled with it can reach 100%.
    """
    if smoke:
        for i in range(n_docs):
            yield " ".join(SMOKE_SENTENCES[(i + j) % len(SMOKE_SENTENCES)] for j in range(4))
        return

    from datasets import load_dataset

    stream = load_dataset(dataset, name=config, split="train", streaming=True)
    kept = 0
    for row in stream:
        text = row.get("text") or ""
        if len(text) > 200:
            yield text
            kept += 1
            if kept >= n_docs:
                return


def download_texts(dataset: str, config: str, n_docs: int, smoke: bool,
                   desc: str = "download") -> list[str]:
    """iter_texts into a list, with a progress bar — streaming a large corpus
    over the network is minutes of otherwise silent time."""
    texts = []
    with bar(total=n_docs, desc=desc, unit="doc") as pb:
        for text in iter_texts(dataset, config, n_docs, smoke):
            texts.append(text)
            pb.update(1)
    return texts


def push_tokenizer_if_asked(args, vocab_size: int) -> None:
    """Push the local tokenizer file to the Hub when --push-tokenizer-to is set.

    Runs for freshly trained *and* cache-reused tokenizers (the Hub commit is a
    no-op when the content is unchanged). A tokenizer that itself came from the
    Hub has no local file and nothing to push.
    """
    if not args.push_tokenizer_to or args.smoke:
        return
    if not os.path.exists(args.tokenizer_path):
        print(f"tokenizer came from the Hub ({args.tokenizer_path}); nothing to push")
        return
    hub.push_tokenizer(args.tokenizer_path, args.push_tokenizer_to, details={
        "vocab size": vocab_size,
        "corpus": f"{args.dataset} ({args.config})",
        "trained by": "lm.py --tokenize-to / training run",
    })


class ByteTokenizer:
    """Zero-dependency fallback for --smoke: UTF-8 bytes offset past specials."""

    vocab = 256 + N_SPECIAL

    def encode_ids(self, text: str) -> list[int]:
        return [b + N_SPECIAL for b in text.encode("utf-8")]


def chunk_documents(texts: Iterable[str], tokenizer, seq_len: int, *,
                    with_doc_ids: bool = False, consume: bool = False,
                    progress=None):
    """One document per chunk boundary — never packs two documents together.

    NeoBERT measured cross-document sequence packing at -2.9 GLUE, so each
    window here stays inside a single document. Partial tails are padded.

    With `with_doc_ids`, also returns the source document index of every window,
    which is what lets the train/val split cut on document boundaries.

    With `consume`, pops each document off `texts` as it is tokenised, so the
    corpus of Python strings is released during the pass rather than staying
    live alongside the growing window table.
    """
    # Grow a preallocated array instead of stacking a list of rows at the end:
    # the list of ndarrays is slightly *larger* than the array it becomes, and
    # np.stack holds both at once.
    cap, n = 1024, 0
    rows = np.empty((cap, seq_len), dtype=np.int32)
    doc_ids = np.empty(cap, dtype=np.int32)

    for doc, ids in _encoded(texts, tokenizer, consume, progress=progress):
        for row in _windows(ids, seq_len):
            if n == cap:
                cap *= 2
                rows = np.resize(rows, (cap, seq_len))
                doc_ids = np.resize(doc_ids, cap)
            rows[n] = row
            doc_ids[n] = doc
            n += 1
    if not n:
        raise RuntimeError("no chunks produced — corpus too small or all filtered")
    rows = rows[:n].copy()
    return (rows, doc_ids[:n].copy()) if with_doc_ids else rows


def _draining(texts):
    """enumerate(), but releases each document as it is handed over.

    At --docs 200000 the corpus is a gigabyte or more of Python strings, and it
    would otherwise stay resident through the whole chunking pass on top of the
    window table being built.
    """
    if not isinstance(texts, list):
        yield from enumerate(texts)
        return
    texts.reverse()  # so popping from the end walks forward through the corpus
    i = 0
    while texts:
        yield i, texts.pop()
        i += 1


def _encoded(texts, tokenizer, consume: bool = False, batch_size: int = 1000,
             progress=None):
    """(document index, token ids), tokenised a batch at a time.

    encode_batch_fast hands the whole batch to Rust, which spreads it across
    every core, and skips the per-token character offsets and word ids that
    encode() builds and this file throws away. Measured against the old
    one-document-at-a-time loop on a 40M-character Georgian corpus: 5.7x on ten
    cores, 2.2x on two — and 1.6x of that is the offset-free path rather than
    the threading, so the win survives even on a single core. The ids are
    byte-identical either way.

    batch_size is a memory dial, not a speed one: 256 through 20000 all measured
    within noise of each other, while peak RSS doubled at the top end.
    """
    source = _draining(texts) if consume else enumerate(texts)
    tick = (lambda k: None) if progress is None else progress.update

    if hasattr(tokenizer, "encode_ids"):  # ByteTokenizer (--smoke): no batch API
        for doc, text in source:
            yield doc, tokenizer.encode_ids(text)
            tick(1)
        return

    batch: list[str] = []
    first = 0
    for doc, text in source:
        if not batch:
            first = doc
        batch.append(text)
        if len(batch) == batch_size:
            for k, enc in enumerate(tokenizer.encode_batch_fast(batch)):
                yield first + k, enc.ids
            tick(len(batch))
            batch = []
    if batch:
        for k, enc in enumerate(tokenizer.encode_batch_fast(batch)):
            yield first + k, enc.ids
        tick(len(batch))


def _windows(ids, seq_len: int):
    """Token ids -> padded [CLS] ... [SEP] rows of exactly seq_len.

    Shared by the in-memory chunker and the streaming tokenise pass so the two
    cannot drift apart in how they cut documents.
    """
    body = seq_len - 2  # room for [CLS] and [SEP]
    for start in range(0, len(ids), body):
        window = ids[start:start + body]
        # Only drop windows too short to carry context. The old threshold of
        # body // 4 — 127 tokens at seq_len 512 — discarded every short
        # document outright and roughly a quarter of all document tails.
        if len(window) < MIN_WINDOW_TOKENS:
            continue
        yield cls_row(window, seq_len)


def split_by_document(doc_ids, val_frac: float, batch_size: int, seed: int):
    """Held-out mask over windows, cut on document boundaries.

    Splitting on windows instead would drop two chunks of the same article on
    opposite sides and the held-out loss would read back part of the training
    set.

    Uses the same per-document hash as the streaming pass, deliberately: the
    in-memory and --data-dir paths then produce byte-identical splits for the
    same corpus and seed, so a run can move between them without the held-out
    number shifting underneath it.
    """
    n_docs = int(doc_ids.max()) + 1
    is_val = np.fromiter((split_hash(int(d), seed) < val_frac for d in doc_ids),
                         dtype=bool, count=len(doc_ids))
    n_val = len({int(d) for d in doc_ids[is_val]})
    want = batch_size * MIN_VAL_BATCHES
    if int(is_val.sum()) < want:
        raise RuntimeError(
            f"only {int(is_val.sum())} held-out windows from {n_val} documents; "
            f"need {want} ({MIN_VAL_BATCHES} batches of {batch_size}). "
            f"Raise --val-frac above {val_frac}, or lower --batch-size."
        )
    return is_val, n_val, n_docs


def split_hash(doc: int, seed: int) -> float:
    """A deterministic uniform draw in [0, 1) keyed on the document index.

    The streaming pass never has every document id in hand at once, so the
    train/val decision has to be a pure function of the index rather than a
    permutation. This is a splitmix-style 32-bit mixer — uniform enough for a
    split, and it costs nothing.
    """
    x = (doc * 0x9E3779B1 + seed * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967296.0


def tokenize_to(args, out_dir: str) -> None:
    """One-time CPU pass: corpus -> train/val ArrayRecord files + meta.json.

    Streams throughout. Nothing accumulates in host RAM but the tokenizer's
    training sample, so --docs is bounded by disk rather than by memory, and a
    GPU or TPU session never spends its time on this.
    """
    from array_record.python import array_record_module

    os.makedirs(out_dir, exist_ok=True)
    timings: dict[str, float] = {}

    # Pass 1: a prefix, only to fit the vocabulary. A 32k byte-level BPE is
    # converged long before a million documents, and training scales linearly,
    # so feeding it the whole corpus costs minutes for a vocabulary that does
    # not differ.
    t0 = time.time()
    if args.smoke:
        tokenizer, vocab_size = ByteTokenizer(), ByteTokenizer.vocab
        sample = list(iter_texts(args.dataset, args.config, args.docs, True))
        timings["sample"] = time.time() - t0
    else:
        tokenizer = resolve_tokenizer(args.tokenizer_path, args.vocab_size)
        if tokenizer is None:
            n_sample = min(args.tokenizer_docs, args.docs)
            sample = download_texts(args.dataset, args.config, n_sample, False,
                                    desc="1/3 sample")
            timings["sample"] = time.time() - t0

            t0 = time.time()
            tokenizer = train_bpe(sample, args.vocab_size, args.tokenizer_path)
            timings["train tokenizer"] = time.time() - t0
        else:
            # An already-available tokenizer (local cache or Hub) makes the
            # sampling pass pointless — pass 2 streams the corpus regardless.
            sample = []
        vocab_size = tokenizer.get_vocab_size()
        push_tokenizer_if_asked(args, vocab_size)

    # Pass 2: re-stream, encode in batches, and write rows out as they appear.
    # When the sample already covers the whole corpus there is nothing to
    # re-download — drain it instead, which is the --smoke case and any run
    # with --docs at or below --tokenizer-docs.
    t0 = time.time()
    reuse = len(sample) >= args.docs
    source = sample if reuse else iter_texts(args.dataset, args.config, args.docs, args.smoke)
    if not reuse:
        del sample

    paths = {k: os.path.join(out_dir, f"{k}.array_record") for k in ("train", "val")}
    # group_size is the decompression unit, and grain reads every source through
    # random access even when the consumer is sequential — it warns loudly at
    # anything above 1. So 1 for both files, val included.
    writers = {k: array_record_module.ArrayRecordWriter(v, "group_size:1")
               for k, v in paths.items()}
    rows_of = {"train": 0, "val": 0}
    docs_of = {"train": 0, "val": 0}
    pb = bar(total=args.docs, desc="3/3 encode", unit="doc")
    try:
        for doc, ids in _encoded(source, tokenizer, consume=reuse, progress=pb):
            side = "val" if split_hash(doc, args.seed) < args.val_frac else "train"
            docs_of[side] += 1
            for row in _windows(ids, args.seq_len):
                writers[side].write(row.astype("<i4").tobytes())
                rows_of[side] += 1
    finally:
        pb.close()
        for w in writers.values():
            w.close()  # required: the chunk index is written here
    timings["encode + write"] = time.time() - t0

    want = args.batch_size * MIN_VAL_BATCHES
    if rows_of["val"] < want:
        raise RuntimeError(
            f"only {rows_of['val']} held-out windows from {docs_of['val']} documents; "
            f"need {want} ({MIN_VAL_BATCHES} batches of {args.batch_size}). "
            f"Raise --val-frac above {args.val_frac}, or lower --batch-size."
        )

    n_docs = docs_of["train"] + docs_of["val"]
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"seq_len": args.seq_len, "vocab_size": vocab_size,
                   "tokenizer": None if args.smoke else args.tokenizer_path,
                   "n_train": rows_of["train"], "n_val": rows_of["val"],
                   "n_docs": n_docs, "val_frac": args.val_frac,
                   "seed": args.seed}, f, indent=2)

    total = sum(timings.values()) or 1e-9
    print(f"wrote {rows_of['train']} train / {rows_of['val']} val windows of "
          f"{args.seq_len} tokens ({docs_of['val']}/{n_docs} documents held out), "
          f"vocab {vocab_size}")
    print("  timing: " + "  ".join(f"{k} {v:.1f}s ({v / total:.0%})"
                                   for k, v in timings.items()))
    print(f"  -> {os.path.abspath(out_dir)}; train with --data-dir {out_dir}")


def read_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def decode_record(raw: bytes) -> np.ndarray:
    """One ArrayRecord payload -> one window.

    The astype is not redundant: np.frombuffer returns a read-only view over
    immutable bytes, and a writeable copy costs 2 KB here.
    """
    return np.frombuffer(raw, dtype="<i4").astype(np.int32)


class MaskExample(grain.transforms.RandomMap):
    """Per-example dynamic masking.

    Grain derives this element's RNG from its index by resetting a Philox
    counter, so the mask a given window receives is a pure function of that
    index. Two consequences worth having: changing --batch-size no longer
    changes every mask, and a resumed run reproduces masks exactly rather than
    relying on the RNG state having been checkpointed.
    """

    def __init__(self, mask_prob: float, n_pred: int):
        self.mask_prob = mask_prob
        self.n_pred = n_pred

    def random_map(self, row, rng):
        out = mask_batch(row, rng, self.mask_prob, self.n_pred)
        return {k: v[0] for k, v in out.items()}  # drop the batch axis


def make_dataset(source, batch_size: int, mask_prob: float, n_pred: int,
                 seed: int, *, decode: bool = False, repeat: bool = True):
    """The whole input pipeline.

    `source` is either the in-memory [N, seq_len] chunk table (a 2-D ndarray
    already satisfies grain's RandomAccessDataSource protocol) or an
    ArrayRecordDataSource, in which case `decode` turns bytes back into rows.

    Deliberately no mp_prefetch: grain spawns workers and ships the dataset
    graph through cloudpickle, which would capture the source array itself and
    give every worker a full resident copy with no copy-on-write. On an
    in-memory source that is a much larger memory problem than the one this
    pipeline solves. num_threads=0 for the same reason grain's own docs give —
    the data is already in RAM, so reader threads only contend on the GIL.
    """
    ds = grain.MapDataset.source(source)
    if decode:
        ds = ds.map(decode_record)
    ds = ds.seed(seed).shuffle()
    if repeat:
        ds = ds.repeat()
    return (
        ds.random_map(MaskExample(mask_prob, n_pred))
        .batch(batch_size, drop_remainder=True)
        .to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=64))
    )


def load_sources(args):
    """Everything between the CLI flags and a (train, val) pair of row sources.

    Returns (train_source, val_source, decode, vocab_size): with --data-dir the
    sources are ArrayRecord files and `decode` is True; otherwise the corpus is
    downloaded, tokenised and split in memory.
    """
    if args.data_dir:
        # Pre-tokenised: windows stream off disk, so host RAM no longer caps the
        # corpus and this session spends none of its time tokenising.
        meta = read_meta(args.data_dir)
        if meta["seq_len"] != args.seq_len:
            raise RuntimeError(
                f"{args.data_dir} was written at --seq-len {meta['seq_len']}, "
                f"but this run asks for {args.seq_len}"
            )
        vocab_size = meta["vocab_size"]
        train_source = grain.sources.ArrayRecordDataSource(
            os.path.join(args.data_dir, "train.array_record"))
        val_source = grain.sources.ArrayRecordDataSource(
            os.path.join(args.data_dir, "val.array_record"))
        print(f"corpus: {len(train_source)} train / {len(val_source)} val windows "
              f"of {args.seq_len} tokens from {args.data_dir}, vocab {vocab_size}")
        return train_source, val_source, True, vocab_size

    print("building corpus ...", flush=True)
    # Resolve the tokenizer *before* the corpus download — same order as
    # tokenize_to: a typo'd Hub id or a missing/read-only token has to fail in
    # seconds, not after --docs documents have streamed.
    tokenizer = (None if args.smoke
                 else resolve_tokenizer(args.tokenizer_path, args.vocab_size))
    texts = download_texts(args.dataset, args.config, args.docs, args.smoke)
    if args.smoke:
        tokenizer, vocab_size = ByteTokenizer(), ByteTokenizer.vocab
    else:
        if tokenizer is None:
            tokenizer = train_bpe(texts, args.vocab_size, args.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        push_tokenizer_if_asked(args, vocab_size)

    n_texts = len(texts)
    pb = bar(total=n_texts, desc="encode", unit="doc")
    chunks, doc_ids = chunk_documents(texts, tokenizer, args.seq_len,
                                      with_doc_ids=True, consume=True, progress=pb)
    pb.close()
    print(f"corpus: {n_texts} docs -> {len(chunks)} sequences of {args.seq_len} tokens "
          f"({len(chunks) * args.seq_len / 1e6:.2f}M tokens), vocab {vocab_size}")
    del texts  # already drained by consume=True; drop the empty list too

    is_val, n_val_docs, n_docs = split_by_document(
        doc_ids, args.val_frac, args.batch_size, args.seed)
    train_source, val_source = chunks[~is_val], chunks[is_val]
    # the fancy-indexed halves above are copies; without this the whole
    # corpus stays resident twice for the life of the run
    del chunks, doc_ids, is_val
    if len(val_source) < args.batch_size or len(train_source) < args.batch_size:
        raise RuntimeError(
            f"corpus too small to split: {n_docs} docs, batch {args.batch_size}"
        )
    print(f"split: {len(train_source)} train / {len(val_source)} held-out windows "
          f"({n_val_docs}/{n_docs} documents held out)")
    return train_source, val_source, False, vocab_size
