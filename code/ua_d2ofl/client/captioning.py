"""BLIP2 caption generation for one client's manifest. GPU (Colab) stage.

Writes JSONL rows {path, label, domain, caption}; reruns skip already
captioned paths, so interrupted Colab sessions just resume.

Usage: python -m ua_d2ofl.client.captioning --manifest client0.json --out captions0.jsonl
"""

import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from ua_d2ofl.data.dataset import load_manifest

BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"


def caption_manifest(manifest_path: str, out_path: str, batch_size: int = 16) -> None:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(BLIP2_MODEL)
    # device_map loads weights straight to GPU — no CPU staging, which OOMs
    # small-RAM Colab VMs
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL, torch_dtype=torch.float16,
        device_map={"": device}).eval()

    samples = load_manifest(manifest_path)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["path"] for line in f if line.strip()}
    todo = [s for s in samples if s["path"] not in done]
    print(f"{len(done)} cached, {len(todo)} to caption")

    with open(out_path, "a", encoding="utf-8") as out:
        for i in tqdm(range(0, len(todo), batch_size)):
            batch = todo[i:i + batch_size]
            images = [Image.open(s["path"]).convert("RGB") for s in batch]
            inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=30)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            for s, text in zip(batch, texts):
                out.write(json.dumps({**s, "caption": text.strip()}) + "\n")
            out.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    a = p.parse_args()
    caption_manifest(a.manifest, a.out, a.batch_size)
