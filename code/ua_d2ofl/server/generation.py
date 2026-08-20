"""Server-side synthetic data generation (D2OFL data synthesis). GPU (Colab) stage.

Conditions Stable Diffusion v1.4 directly on the clients' `cond` prototypes
([77, 768] CLIP text hidden states) via modern diffusers' `prompt_embeds`,
replacing OSCAR's hand-rolled DDIM loop. Deterministic seeds per
(client, label, index); already-generated images are skipped, so
interrupted Colab sessions resume.

Output: out_dir/client{r}/{label}/{i}.png + manifest.jsonl rows
{path, label, client, seed}.

Usage: python -m ua_d2ofl.server.generation --prototypes proto0.pt:0 proto1.pt:1 \
    --out D_syn --num_images 30
"""

import argparse
import json
import os

import torch

SD_MODEL = "CompVis/stable-diffusion-v1-4"
GUIDANCE_SCALE = 7.5
NUM_STEPS = 50


def _load_pipe(device: str):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_dataset(prototype_files: dict[int, str], out_dir: str,
                     num_images: int = 30, steps: int = NUM_STEPS,
                     batch: int = 5, device: str | None = None) -> list[dict]:
    device = device or "cuda"
    pipe = _load_pipe(device)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for client, proto_file in sorted(prototype_files.items()):
            protos = torch.load(proto_file, map_location="cpu", weights_only=False)
            for label, entry in sorted(protos.items()):
                img_dir = os.path.join(out_dir, f"client{client}", str(label))
                os.makedirs(img_dir, exist_ok=True)
                cond = entry["cond"].to(device, torch.float16).unsqueeze(0)
                todo = [i for i in range(num_images)
                        if not os.path.exists(os.path.join(img_dir, f"{i}.png"))]
                for start in range(0, len(todo), batch):
                    chunk = todo[start:start + batch]
                    seed = client * 1_000_000 + label * 1_000 + chunk[0]
                    images = pipe(prompt_embeds=cond.expand(len(chunk), -1, -1),
                                  negative_prompt=[""] * len(chunk),
                                  num_inference_steps=steps,
                                  guidance_scale=GUIDANCE_SCALE,
                                  generator=torch.Generator(device).manual_seed(seed)).images
                    for i, image in zip(chunk, images):
                        path = os.path.join(img_dir, f"{i}.png")
                        image.save(path)
                        row = {"path": path, "label": label, "client": client, "seed": seed}
                        mf.write(json.dumps(row) + "\n")
                        mf.flush()
                        rows.append(row)
    return rows


def load_syn_manifest(out_dir: str) -> list[dict]:
    """Read manifest.jsonl, dropping rows whose image vanished, dedup by path."""
    seen = {}
    with open(os.path.join(out_dir, "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if os.path.exists(row["path"]):
                    seen[row["path"]] = row
    return list(seen.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prototypes", nargs="+", required=True,
                   help="prototype files as path:client_id, e.g. proto0.pt:0")
    p.add_argument("--out", required=True)
    p.add_argument("--num_images", type=int, default=30)
    a = p.parse_args()
    files = {int(spec.rsplit(":", 1)[1]): spec.rsplit(":", 1)[0] for spec in a.prototypes}
    generate_dataset(files, a.out, a.num_images)
    print(f"done -> {a.out}")
