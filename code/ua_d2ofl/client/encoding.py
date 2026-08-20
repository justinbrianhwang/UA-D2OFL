"""CLIP encoding + prompt fusion + Mahalanobis filtering -> per-class prototypes.

For each class this produces, from one client's captions:
  cond   [77, 768] mean of full CLIP text hidden states (Stable Diffusion
                   conditioning, same text encoder as SD v1.x)
  pooled [768]     mean of projected CLIP text embeddings (UA-D2OFL
                   prototype-relevance signal, cosine-comparable with
                   CLIP image embeddings)
One filter decision (on pooled, D2OFL Eq. 11-15) is applied to both
representations; filtered stats are stored per class for the
missing-prototype/diagnostics logging the research prompt requires.

Usage: python -m ua_d2ofl.client.encoding --captions captions0.jsonl \
    --classes classes.json --out prototypes0.pt
"""

import argparse
import json
from collections import defaultdict

import torch

from ua_d2ofl.client.filtering import filter_and_average

CLIP_MODEL = "openai/clip-vit-large-patch14"  # == SD v1.x text encoder
PROMPT_TEMPLATE = "a photo of a {name}"


@torch.no_grad()
def encode_texts(texts: list[str], device: str, batch_size: int = 64):
    """Return (hidden [N,77,768], pooled [N,768]) for a list of texts."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    if not hasattr(encode_texts, "_model"):
        encode_texts._tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
        encode_texts._model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
    tok, model = encode_texts._tok, encode_texts._model
    hiddens, pooleds = [], []
    for i in range(0, len(texts), batch_size):
        inputs = tok(texts[i:i + batch_size], padding="max_length", max_length=77,
                     truncation=True, return_tensors="pt").to(device)
        out = model(**inputs)
        hiddens.append(out.last_hidden_state.cpu())
        pooleds.append(out.text_embeds.cpu())
    return torch.cat(hiddens), torch.cat(pooleds)


def build_prototypes(captions_path: str, class_names: list[str], k: float = 3.0,
                     device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    by_class = defaultdict(list)
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_class[row["label"]].append(row["caption"])

    prototypes = {}
    for label, caps in sorted(by_class.items()):
        prompt = PROMPT_TEMPLATE.format(name=class_names[label].replace("_", " "))
        hidden, pooled = encode_texts(caps + [prompt], device)  # Eq. 10 fusion
        protos, stats = filter_and_average({"cond": hidden, "pooled": pooled}, k=k)
        prototypes[label] = {**protos, "stats": stats}
        print(f"class {label} ({class_names[label]}): kept {stats['n_kept']}/{stats['n_total']}")
    return prototypes


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--captions", required=True)
    p.add_argument("--classes", required=True, help="JSON list of class names, index = label")
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=float, default=3.0)
    a = p.parse_args()
    with open(a.classes, encoding="utf-8") as f:
        names = json.load(f)
    torch.save(build_prototypes(a.captions, names, k=a.k), a.out)
    print(f"saved -> {a.out}")
