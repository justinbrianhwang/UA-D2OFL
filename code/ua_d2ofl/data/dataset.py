import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    IMAGENET_NORM,
])
TEST_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    IMAGENET_NORM,
])


class ManifestDataset(Dataset):
    """Dataset over a list of {path, label, ...} samples (see data.partition)."""

    def __init__(self, samples: list[dict], transform=TEST_TF):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        return self.transform(img), s["label"]


def save_manifest(samples: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f)


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
