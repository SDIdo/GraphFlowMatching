"""
Image-space dataset wrappers that make the Graph Flow Matching (GFM) pipeline
usable on CIFAR-10, ImageNet (ILSVRC-2012) and ImageNet-LT.

Nothing here touches the model.  These classes only produce tensors in exactly
the format the existing training loop already consumes, i.e.

    (x0, x1)          -- when ``return_label=False``
    (x0, x1, y)       -- when ``return_label=True``

where ``x0`` is the empty placeholder tensor the repository already uses for the
"source" sample, ``x1`` is the image normalised to [-1, 1] with shape
[3, H, W], and ``y`` is the integer class index.

The GFM velocity network operates on Stable-Diffusion VAE latents of shape
[4, S, S] with S = image_size / 8.  All three datasets are therefore rendered at
a VAE-friendly ``image_size`` (default 256, giving the S = 32 latents the
released configurations expect).  CIFAR-10 images are upsampled to that
resolution; ImageNet / ImageNet-LT use the ADM center-crop used by the paper.
"""

import os
import pathlib
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# ADM/LFM center crop, shared with the existing ImageNet loader in this repo.
from datasets.imagenet_dataset import center_crop_arr

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}

_EMPTY = torch.empty(0, dtype=torch.float32)


def _norm_tfms(image_size, horizontal_flip, resize_mode="bicubic"):
    """Common tail of the transform pipeline: -> float tensor in [-1, 1]."""
    interp = {
        "bicubic": T.InterpolationMode.BICUBIC,
        "bilinear": T.InterpolationMode.BILINEAR,
        "nearest": T.InterpolationMode.NEAREST,
    }[resize_mode]
    tfms = [T.Resize((image_size, image_size), interpolation=interp)]
    if horizontal_flip:
        tfms.append(T.RandomHorizontalFlip(p=0.5))
    tfms += [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return T.Compose(tfms)


# --------------------------------------------------------------------------- #
#  CIFAR-10
# --------------------------------------------------------------------------- #
class CIFAR10Dataset(Dataset):
    """CIFAR-10 rendered at ``image_size`` (default 256) in [-1, 1].

    CIFAR-10 is natively 32x32.  Because GFM is a *latent* flow-matching model
    that lives in the SD-VAE latent space, the images are bicubically upsampled
    to ``image_size`` before encoding, so the velocity network sees exactly the
    4 x 32 x 32 latents it was designed for.  Generated samples are downsampled
    back to 32x32 at evaluation time (see ``evaluation/evaluate_gfm.py``).
    """

    NUM_CLASSES = 10
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
               "frog", "horse", "ship", "truck"]

    def __init__(self, root_dir, split="train", image_size=256,
                 use_horizontal_flips=False, download=True, return_label=True):
        from torchvision.datasets import CIFAR10  # local import: optional dep

        assert split in ("train", "test"), f"unknown CIFAR-10 split: {split}"
        self.image_size = image_size
        self.return_label = return_label
        self.transform = _norm_tfms(image_size, use_horizontal_flips)
        self.base = CIFAR10(root=root_dir, train=(split == "train"),
                            download=download, transform=self.transform)

    def __len__(self):
        return len(self.base)

    def class_counts(self):
        return Counter(int(y) for y in self.base.targets)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        if self.return_label:
            return _EMPTY, img, int(y)
        return _EMPTY, img


# --------------------------------------------------------------------------- #
#  ImageNet (ILSVRC-2012)
# --------------------------------------------------------------------------- #
def _scan_imagenet_folder(root):
    """Return (samples, class_to_idx) for a wnid-per-subfolder ImageNet tree."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet root not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(
            f"{root} contains no class sub-directories (expected n01440764/, ...)")
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for cls in classes:
        for p in sorted((root / cls).rglob("*")):
            if p.is_file() and p.suffix in IMG_EXTS:
                samples.append((str(p.relative_to(root)).replace("\\", "/"),
                                class_to_idx[cls]))
    return samples, class_to_idx


class ImageNetDataset(Dataset):
    """ImageNet-1k with the ADM center-crop preprocessing used by LFM/GFM.

    ``root_dir`` must be the split directory that holds the 1000 wnid folders,
    e.g. ``/data/imagenet/train``.
    """

    NUM_CLASSES = 1000

    def __init__(self, root_dir, image_size=256, use_horizontal_flips=True,
                 return_label=True, samples=None, class_to_idx=None):
        self.root = pathlib.Path(root_dir)
        self.image_size = image_size
        self.return_label = return_label
        self.horizontal_flip = use_horizontal_flips
        if samples is None:
            samples, class_to_idx = _scan_imagenet_folder(root_dir)
        self.samples = samples
        self.class_to_idx = class_to_idx
        self._flip = T.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.samples)

    def class_counts(self):
        return Counter(y for _, y in self.samples)

    def _load(self, relpath):
        with Image.open(self.root / relpath) as im:
            im = im.convert("RGB")
            im = center_crop_arr(im, self.image_size)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        return x

    def __getitem__(self, idx):
        relpath, y = self.samples[idx]
        x = self._load(relpath)
        if self.horizontal_flip:
            x = self._flip(x)
        if self.return_label:
            return _EMPTY, x, int(y)
        return _EMPTY, x


# --------------------------------------------------------------------------- #
#  ImageNet-LT
# --------------------------------------------------------------------------- #
#  The official split of Liu et al., "Large-Scale Long-Tailed Recognition in an
#  Open World" (CVPR 2019): 115,846 train images over all 1000 classes, with
#  1280 images for the most frequent class and 5 for the rarest.
LT_SPLIT_FILES = {
    "train": "ImageNet_LT_train.txt",
    "val": "ImageNet_LT_val.txt",
    "test": "ImageNet_LT_test.txt",
}

# Mirrors of the official file lists, verified reachable. The primary is
# facebookresearch/classifier-balancing ("Decoupling Representation and
# Classifier for Long-Tailed Recognition"), which redistributes the OLTR splits.
_FB = ("https://raw.githubusercontent.com/facebookresearch/classifier-balancing/"
       "{branch}/data/ImageNet_LT/{name}")
_MISLAS = ("https://raw.githubusercontent.com/dvlab-research/MiSLAS/"
           "main/datasets/data_txt/{name}")

LT_SPLIT_URLS = {
    split: [
        _FB.format(branch="main", name=fname),
        _FB.format(branch="master", name=fname),
        _MISLAS.format(name=fname),
    ]
    for split, fname in LT_SPLIT_FILES.items()
}

# Expected size of the official train split, used as a sanity check.
LT_TRAIN_EXPECTED = {"num_images": 115846, "num_classes": 1000,
                     "max_per_class": 1280, "min_per_class": 5}


def download_lt_split(split, dst_dir):
    """Best-effort download of an official ImageNet-LT split file.

    Returns the local path on success, or ``None`` if every mirror failed.  The
    caller is expected to fall back to :func:`build_lt_split_pareto`.
    """
    import urllib.request

    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, LT_SPLIT_FILES[split])
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    for url in LT_SPLIT_URLS.get(split, []):
        try:
            print(f"[ImageNet-LT] downloading {url}")
            urllib.request.urlretrieve(url, dst)
            if os.path.getsize(dst) > 0:
                return dst
        except Exception as exc:  # noqa: BLE001 - mirrors are flaky by nature
            print(f"[ImageNet-LT] mirror failed ({exc})")
            if os.path.exists(dst):
                os.remove(dst)
    return None


def read_lt_split_file(path, class_to_idx=None):
    """Parse ``<relative/path> <class_idx>`` lines into (relpath, label) pairs.

    The official files are relative to the ImageNet *root* (they start with
    ``train/`` or ``val/``); we keep the path verbatim and let the dataset join
    it against its own root.
    """
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            relpath = parts[0].replace("\\", "/")
            if len(parts) > 1:
                label = int(parts[1])
            elif class_to_idx is not None:
                label = class_to_idx[relpath.split("/")[-2]]
            else:
                raise ValueError(f"cannot infer label for line: {line!r}")
            samples.append((relpath, label))
    return samples


def build_lt_split_pareto(samples, num_classes=1000, max_per_class=1280,
                          min_per_class=5, pareto_alpha=6.0, seed=0):
    """Reconstruct an ImageNet-LT-style long-tailed subset from a full split.

    Fallback for when the official ``ImageNet_LT_train.txt`` is unavailable.
    Reproduces the *profile* of the official split -- a Pareto(alpha) decay from
    ``max_per_class`` down to ``min_per_class`` over ``num_classes`` classes --
    but not its exact file list.  Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    by_class = {}
    for relpath, y in samples:
        by_class.setdefault(y, []).append(relpath)

    classes = sorted(by_class.keys())
    num_classes = min(num_classes, len(classes))
    # Class order is shuffled so the head is not simply wnid-sorted.
    order = list(classes)
    rng.shuffle(order)

    lt_samples = []
    counts = {}
    last = float(num_classes) ** (-1.0 / pareto_alpha)
    for rank, cls in enumerate(order):
        # Pareto decay: n(rank) ~ (1 + rank)^(-1/alpha), rescaled so that the
        # head class gets `max_per_class` and the tail class `min_per_class`.
        frac = (1.0 + rank) ** (-1.0 / pareto_alpha)
        n = min_per_class + (max_per_class - min_per_class) * \
            (frac - last) / (1.0 - last)
        n = int(max(min_per_class, round(n)))
        pool = by_class[cls]
        n = min(n, len(pool))
        picked = rng.sample(pool, n)
        counts[cls] = n
        lt_samples.extend((p, cls) for p in picked)

    lt_samples.sort()
    return lt_samples, counts


class ImageNetLTDataset(ImageNetDataset):
    """ImageNet-LT: the long-tailed subset of ImageNet-1k.

    Args:
        root_dir: ImageNet *root* containing ``train/`` and ``val/``.  The
            official split files address images relative to this root.
        split_file: path to ``ImageNet_LT_train.txt``.  If ``None``, the file is
            downloaded into ``split_dir``; if that fails and
            ``allow_pareto_fallback`` is set, a Pareto-reconstructed split is
            built from the full ``train/`` folder instead.
    """

    NUM_CLASSES = 1000

    def __init__(self, root_dir, split="train", image_size=256,
                 use_horizontal_flips=True, return_label=True,
                 split_file=None, split_dir=None,
                 allow_pareto_fallback=True, pareto_seed=0,
                 full_split_subdir="train"):
        root = pathlib.Path(root_dir)
        split_dir = split_dir or str(root / "ImageNet_LT_splits")

        if split_file is None:
            split_file = download_lt_split(split, split_dir)

        if split_file is not None and os.path.exists(split_file):
            samples = read_lt_split_file(split_file)
            class_to_idx = None
            self.split_source = f"official:{os.path.basename(split_file)}"
            if split == "train":
                counts = Counter(y for _, y in samples)
                vals = sorted(counts.values(), reverse=True)
                got = {"num_images": len(samples), "num_classes": len(counts),
                       "max_per_class": vals[0], "min_per_class": vals[-1]}
                if got == LT_TRAIN_EXPECTED:
                    print(f"[ImageNet-LT] official split verified: {got}")
                else:
                    print(f"[ImageNet-LT] WARNING: split statistics {got} differ "
                          f"from the published {LT_TRAIN_EXPECTED}")
        else:
            if not allow_pareto_fallback:
                raise FileNotFoundError(
                    "ImageNet-LT split file unavailable. Download "
                    f"{LT_SPLIT_FILES[split]} into {split_dir} or pass "
                    "--imagenet_lt_split_file.")
            print("[ImageNet-LT] official split unavailable -> reconstructing "
                  "a Pareto(alpha=6) long-tailed split from the full train set.")
            full, class_to_idx = _scan_imagenet_folder(root / full_split_subdir)
            # Re-root the paths so they are relative to `root`.
            full = [(f"{full_split_subdir}/{p}", y) for p, y in full]
            samples, _ = build_lt_split_pareto(full, seed=pareto_seed)
            self.split_source = f"pareto_reconstruction(seed={pareto_seed})"

        super().__init__(root_dir=root_dir, image_size=image_size,
                         use_horizontal_flips=use_horizontal_flips,
                         return_label=return_label,
                         samples=samples, class_to_idx=class_to_idx)

    def imbalance_report(self):
        counts = self.class_counts()
        vals = sorted(counts.values(), reverse=True)
        return {
            "num_images": len(self.samples),
            "num_classes": len(counts),
            "max_per_class": vals[0] if vals else 0,
            "min_per_class": vals[-1] if vals else 0,
            "imbalance_ratio": (vals[0] / vals[-1]) if vals and vals[-1] else float("inf"),
            "split_source": getattr(self, "split_source", "unknown"),
        }


# --------------------------------------------------------------------------- #
#  Shot-group helper (many / medium / few) -- used for ImageNet-LT reporting
# --------------------------------------------------------------------------- #
def shot_groups(class_counts, many_thr=100, few_thr=20):
    """Split classes into the standard many/medium/few-shot groups."""
    many, medium, few = [], [], []
    for cls, n in class_counts.items():
        if n > many_thr:
            many.append(cls)
        elif n >= few_thr:
            medium.append(cls)
        else:
            few.append(cls)
    return {"many": sorted(many), "medium": sorted(medium), "few": sorted(few)}


def build_image_dataset(name, root_dir, split="train", image_size=256,
                        use_horizontal_flips=False, return_label=True,
                        **kwargs):
    """Factory used by the encoder / stats scripts and by train.py."""
    name = name.lower()
    if name in ("cifar10", "cifar-10"):
        return CIFAR10Dataset(root_dir, split=split, image_size=image_size,
                              use_horizontal_flips=use_horizontal_flips,
                              return_label=return_label,
                              download=kwargs.get("download", True))
    if name in ("imagenet", "imnet", "imagenet-1k"):
        sub = kwargs.get("imagenet_subdir", split)
        root = os.path.join(root_dir, sub) if sub else root_dir
        return ImageNetDataset(root, image_size=image_size,
                               use_horizontal_flips=use_horizontal_flips,
                               return_label=return_label)
    if name in ("imagenet-lt", "imagenet_lt", "imnet-lt"):
        return ImageNetLTDataset(
            root_dir, split=split, image_size=image_size,
            use_horizontal_flips=use_horizontal_flips,
            return_label=return_label,
            split_file=kwargs.get("split_file"),
            split_dir=kwargs.get("split_dir"),
            allow_pareto_fallback=kwargs.get("allow_pareto_fallback", True),
            pareto_seed=kwargs.get("pareto_seed", 0),
            full_split_subdir=kwargs.get("imagenet_subdir", "train"),
        )
    raise ValueError(f"Unknown dataset: {name}")
