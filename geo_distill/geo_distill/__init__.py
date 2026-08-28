"""Distill teacher embeddings into a small Georgian student model.

The pipeline, each stage a subcommand of `python -m geo_distill` (or the
installed `geo-distill` script):

    data               stream a HF corpus -> data/sentences.txt
    tokenizer          train a BPE/Unigram carrying the mlm special-token layout
    teacher            embed sentences with the Gemini teacher (cached, rate-limited)
    local-teacher      embed with a local open-weights teacher (Qwen3-Embedding, GPU)
    fetch-teacher      pull teacher artifacts back from a HF dataset repo
    synthetic-teacher  offline stand-in for the teacher (no API key, no credits)
    train              distill the teacher's geometry into a student
    eval               score the student against the teacher + retrieval demo

Two students share the training loop: a tiny from-scratch EmbeddingModel, or
(recommended) the lm-pretrained MLM encoder fine-tuned via --mlm-checkpoint
(e.g. ZurabDz/ka-mlm, paired with its tokenizer --tokenizer ZurabDz/ka-bpe-32k).

This module deliberately imports nothing heavy, so `--help` stays instant;
each subcommand loads only what it needs.
"""
