"""One-time server-side caches shared by every distillation method.

Teacher logits, CLIP image embeddings, and prototype similarities depend
only on the frozen teachers and the fixed D_syn — never on the student —
so they are computed once and reused by Uniform and every UA variant.
This both saves compute and guarantees the controlled comparison sees
identical teacher signals.

Cache dict (torch.save-able):
  teacher_logits [N, R, C]   T=1 logits of every teacher on every sample
  image_embeds   [N, 768]    CLIP (ViT-L/14) image embeddings
  sims           [N, R]      cosine(z_x, pooled prototype of teacher r for label(x));
                             0 where unavailable
  available      [N, R]      1 if teacher r has a prototype for label(x)
  labels         [N]
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ua_d2ofl.data.dataset import ManifestDataset, TEST_TF
from ua_d2ofl.models import initialize_model

CLIP_MODEL = "openai/clip-vit-large-patch14"


@torch.no_grad()
def teacher_logits(samples: list[dict], teacher_ckpts: list[str], batch_size: int = 128,
                   device: str | None = None) -> torch.Tensor:
    """[N, R, C] logits; teachers loaded one at a time to bound GPU memory."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(ManifestDataset(samples, TEST_TF), batch_size=batch_size,
                        shuffle=False, num_workers=2)
    per_teacher = []
    for ckpt_path in teacher_ckpts:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = initialize_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()
        outs = [model(x.to(device)).cpu() for x, _ in tqdm(loader, desc=ckpt_path, leave=False)]
        per_teacher.append(torch.cat(outs))
        del model
    return torch.stack(per_teacher, dim=1)  # [N, R, C]


@torch.no_grad()
def clip_image_embeds(samples: list[dict], batch_size: int = 128,
                      device: str | None = None) -> torch.Tensor:
    """[N, 768] projected CLIP image embeddings (same space as pooled prototypes)."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
    embeds = []
    for i in tqdm(range(0, len(samples), batch_size), desc="clip image embeds"):
        images = [Image.open(s["path"]).convert("RGB") for s in samples[i:i + batch_size]]
        inputs = proc(images=images, return_tensors="pt").to(device)
        embeds.append(model(**inputs).image_embeds.cpu())
    return torch.cat(embeds)


def prototype_sims(image_embeds: torch.Tensor, labels: torch.Tensor,
                   prototype_files: dict[int, str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sims [N, R], available [N, R]). Pure torch — no GPU models."""
    n = image_embeds.shape[0]
    clients = sorted(prototype_files)
    sims = torch.zeros(n, len(clients))
    avail = torch.zeros(n, len(clients))
    z = F.normalize(image_embeds.float(), dim=-1)
    for r, client in enumerate(clients):
        protos = torch.load(prototype_files[client], map_location="cpu", weights_only=False)
        pooled = {lb: F.normalize(e["pooled"].float(), dim=-1) for lb, e in protos.items()}
        for lb, proto in pooled.items():
            idx = (labels == lb).nonzero(as_tuple=True)[0]
            if idx.numel():
                sims[idx, r] = z[idx] @ proto
                avail[idx, r] = 1.0
    return sims, avail


def build_cache(samples: list[dict], teacher_ckpts: list[str],
                prototype_files: dict[int, str], device: str | None = None,
                batch_size: int = 128) -> dict:
    labels = torch.tensor([s["label"] for s in samples])
    logits = teacher_logits(samples, teacher_ckpts, batch_size, device)
    z = clip_image_embeds(samples, batch_size, device)
    sims, avail = prototype_sims(z, labels, prototype_files)
    missing_rate = 1.0 - avail.mean().item()
    print(f"cache built: N={len(samples)} R={len(teacher_ckpts)} "
          f"missing-prototype rate={missing_rate:.3f}")
    return {"teacher_logits": logits, "image_embeds": z, "sims": sims,
            "available": avail, "labels": labels}
