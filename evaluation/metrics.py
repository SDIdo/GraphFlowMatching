"""
Generative-model metrics for Graph Flow Matching: FID, Inception Score, MMD,
KID and t-SNE embeddings.

Design notes
------------
* **FID** is delegated to ``clean-fid`` whenever it is installed, because that
  package reproduces the reference Inception graph and the correct resizing.  A
  self-contained fallback (torchvision Inception-V3 pool features + the usual
  Frechet formula) is provided so the pipeline still runs without clean-fid; the
  two are *not* numerically interchangeable and the reported number always
  records which backend produced it.
* **IS**, **MMD**, **KID** and the **t-SNE** embedding all share one feature
  pass over the images using torchvision's Inception-V3
  (``IMAGENET1K_V1``): the 2048-d ``avgpool`` features for MMD/KID/t-SNE and the
  1000-way softmax for IS.
* MMD is estimated with the *unbiased* U-statistic and a multi-bandwidth RBF
  kernel whose scale is set by the median heuristic on the pooled sample.  With
  tens of thousands of images the full Gram matrix is unnecessary, so the
  estimate is averaged over ``n_subsets`` random subsets of ``subset_size``
  images each -- the same subset scheme KID conventionally uses -- and both the
  mean and the standard deviation are reported.
"""

import contextlib
import functools
import json
import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}


# --------------------------------------------------------------------------- #
#  Image folder -> tensors
# --------------------------------------------------------------------------- #
def list_images(folder, limit=0):
    folder = pathlib.Path(folder)
    paths = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix in IMG_EXTS)
    if limit:
        paths = paths[:limit]
    return [str(p) for p in paths]


class _ImagePathDataset(Dataset):
    """Loads images as float tensors in [0, 1], resized to ``size``."""

    def __init__(self, paths, size=299):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            # np.array (not asarray): PIL's buffer is read-only and torch wants
            # a writable one.
            arr = np.array(im, dtype=np.uint8)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return x


def _collate_resize(batch, size=299):
    x = torch.stack(batch, dim=0)
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear",
                          align_corners=False)
    return x


# --------------------------------------------------------------------------- #
#  Inception-V3 feature extractor
# --------------------------------------------------------------------------- #
_INCEPTION_CACHE = {}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_inception(device="cuda"):
    """torchvision Inception-V3 with the classifier head kept, in eval mode."""
    key = str(device)
    if key in _INCEPTION_CACHE:
        return _INCEPTION_CACHE[key]
    from torchvision.models import inception_v3
    try:
        from torchvision.models import Inception_V3_Weights
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                           aux_logits=True, init_weights=False)
    except Exception:  # older torchvision
        net = inception_v3(pretrained=True, aux_logits=True)
    net.fc_backup = net.fc
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    _INCEPTION_CACHE[key] = net
    return net


@torch.no_grad()
def _forward_inception(net, x):
    """Return (pool_features [B, 2048], logits [B, 1000]) for x in [0, 1]."""
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    x = net.Conv2d_1a_3x3(x)
    x = net.Conv2d_2a_3x3(x)
    x = net.Conv2d_2b_3x3(x)
    x = net.maxpool1(x)
    x = net.Conv2d_3b_1x1(x)
    x = net.Conv2d_4a_3x3(x)
    x = net.maxpool2(x)
    x = net.Mixed_5b(x)
    x = net.Mixed_5c(x)
    x = net.Mixed_5d(x)
    x = net.Mixed_6a(x)
    x = net.Mixed_6b(x)
    x = net.Mixed_6c(x)
    x = net.Mixed_6d(x)
    x = net.Mixed_6e(x)
    x = net.Mixed_7a(x)
    x = net.Mixed_7b(x)
    x = net.Mixed_7c(x)
    x = net.avgpool(x)
    feats = torch.flatten(x, 1)
    logits = net.fc_backup(feats)
    return feats, logits


@torch.no_grad()
def extract_inception_features(folder, device="cuda", batch_size=50,
                               num_workers=0, limit=0, verbose=True):
    """Return (pool_features [N, 2048], probs [N, 1000]) as float32 numpy."""
    paths = list_images(folder, limit=limit)
    if not paths:
        raise FileNotFoundError(f"no images found in {folder}")
    net = get_inception(device)
    ds = _ImagePathDataset(paths, size=299)
    # functools.partial (not a lambda) so DataLoader worker processes can
    # pickle the collate function on Windows/spawn.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        collate_fn=functools.partial(_collate_resize, size=299))
    feats, probs = [], []
    for i, x in enumerate(loader):
        x = x.to(device, non_blocking=True)
        f, logits = _forward_inception(net, x)
        feats.append(f.cpu().numpy().astype(np.float32))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
        if verbose and (i % 20 == 0):
            print(f"  inception {min((i + 1) * batch_size, len(paths))}/{len(paths)}",
                  flush=True)
    return np.concatenate(feats, 0), np.concatenate(probs, 0)


# --------------------------------------------------------------------------- #
#  Inception Score
# --------------------------------------------------------------------------- #
def inception_score(probs, splits=10, eps=1e-12):
    """Salimans et al. (2016) Inception Score from a [N, 1000] softmax matrix."""
    n = probs.shape[0]
    if n < splits:
        splits = max(1, n)
    scores = []
    for k in range(splits):
        part = probs[k * n // splits:(k + 1) * n // splits]
        if part.shape[0] == 0:
            continue
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------- #
#  Frechet Inception Distance
# --------------------------------------------------------------------------- #
def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
                 - 2 * np.trace(covmean))


def fid_from_features(feats_real, feats_fake):
    """Self-contained FID on torchvision-Inception pool features."""
    mu_r, mu_f = feats_real.mean(0), feats_fake.mean(0)
    sig_r = np.cov(feats_real, rowvar=False)
    sig_f = np.cov(feats_fake, rowvar=False)
    return _frechet_distance(mu_r, sig_r, mu_f, sig_f)


def _assert_pure_image_folder(folder):
    """clean-fid globs ``**/*.{ext}`` with 'npy' among its extensions, so a
    stray .npy anywhere under an image folder is loaded as an image and crashes
    resizing. Catch that here with a message that says what to do."""
    if not folder or not os.path.isdir(folder):
        return
    stray = [str(p) for p in pathlib.Path(folder).rglob("*.npy")]
    if stray:
        raise ValueError(
            f"{folder} contains .npy files that clean-fid would try to read as "
            f"images: {stray[:3]}. Move them out of the image tree (the "
            f"reference builder writes them to <dir>_meta/).")


def _fs_is_case_insensitive(path):
    """True on NTFS/APFS-style filesystems where FOO.PNG and foo.png collide."""
    try:
        p = pathlib.Path(path)
        for child in p.iterdir():
            if child.is_file():
                swapped = child.with_name(child.name.upper()
                                          if child.name.islower()
                                          else child.name.lower())
                if swapped.name != child.name:
                    return swapped.exists()
        return os.name == "nt"
    except OSError:
        return os.name == "nt"


@contextlib.contextmanager
def _dedup_cleanfid_extensions(folder):
    """Work around clean-fid double-counting files on case-insensitive volumes.

    clean-fid globs once per entry of its EXTENSIONS set, which holds both
    ``png`` and ``PNG``. On Windows every file therefore matches twice. Exact
    duplication leaves the feature mean untouched and inflates the covariance
    by only 2(N-1)/(2N-1), so FID is barely biased at realistic N -- but it
    doubles the feature-extraction cost, and at small N the bias is visible.
    Inside this context, EXTENSIONS is narrowed to the distinct suffixes
    actually present in ``folder``, one casing each.
    """
    from cleanfid import fid as cfid

    if not folder or not os.path.isdir(folder) or not _fs_is_case_insensitive(folder):
        yield
        return

    present = {p.suffix.lstrip(".").lower()
               for p in pathlib.Path(folder).rglob("*") if p.is_file()}
    keep = {e for e in cfid.EXTENSIONS if e.lower() in present and e.islower()}
    if not keep:
        yield
        return
    original = cfid.EXTENSIONS
    cfid.EXTENSIONS = keep
    try:
        yield
    finally:
        cfid.EXTENSIONS = original


def fid_cleanfid(gen_dir, ref_dir=None, dataset_name=None, mode="clean",
                 device="cuda", num_workers=0, batch_size=32):
    """FID via clean-fid, either folder-vs-folder or folder-vs-custom-stats."""
    from cleanfid import fid as cfid
    _assert_pure_image_folder(gen_dir)
    _assert_pure_image_folder(ref_dir)
    dev = torch.device(device)
    with _dedup_cleanfid_extensions(gen_dir):
        if dataset_name:
            return float(cfid.compute_fid(gen_dir, dataset_name=dataset_name,
                                          dataset_split="custom", mode=mode,
                                          num_workers=num_workers,
                                          batch_size=batch_size, device=dev))
        return float(cfid.compute_fid(gen_dir, ref_dir, mode=mode,
                                      num_workers=num_workers,
                                      batch_size=batch_size, device=dev))


# --------------------------------------------------------------------------- #
#  MMD (RBF kernel, unbiased U-statistic, subset-averaged)
# --------------------------------------------------------------------------- #
def _pairwise_sq_dists(a, b):
    """[m, d], [n, d] -> [m, n] squared Euclidean distances."""
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True).t()
    d2 = a2 + b2 - 2.0 * (a @ b.t())
    return d2.clamp_min_(0.0)


def median_bandwidth(x, y, max_points=2000, seed=0):
    """Median-heuristic sigma^2 on the pooled sample."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pooled = torch.cat([x, y], dim=0)
    if pooled.shape[0] > max_points:
        idx = torch.randperm(pooled.shape[0], generator=g)[:max_points]
        pooled = pooled[idx.to(pooled.device)]
    d2 = _pairwise_sq_dists(pooled, pooled)
    iu = torch.triu_indices(d2.shape[0], d2.shape[0], offset=1)
    med = d2[iu[0], iu[1]].median()
    return float(med.clamp_min(1e-8))


def mmd2_rbf_unbiased(x, y, sigma2_list):
    """Unbiased MMD^2 estimate with a sum of RBF kernels."""
    m, n = x.shape[0], y.shape[0]
    dxx = _pairwise_sq_dists(x, x)
    dyy = _pairwise_sq_dists(y, y)
    dxy = _pairwise_sq_dists(x, y)

    kxx = torch.zeros_like(dxx)
    kyy = torch.zeros_like(dyy)
    kxy = torch.zeros_like(dxy)
    for s2 in sigma2_list:
        kxx += torch.exp(-dxx / (2.0 * s2))
        kyy += torch.exp(-dyy / (2.0 * s2))
        kxy += torch.exp(-dxy / (2.0 * s2))

    kxx.fill_diagonal_(0.0)
    kyy.fill_diagonal_(0.0)
    term_xx = kxx.sum() / (m * (m - 1))
    term_yy = kyy.sum() / (n * (n - 1))
    term_xy = kxy.sum() / (m * n)
    return float(term_xx + term_yy - 2.0 * term_xy)


def compute_mmd(feats_real, feats_fake, device="cuda", subset_size=2000,
                n_subsets=20, bandwidth_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
                standardize=True, seed=0):
    """Subset-averaged unbiased RBF-MMD between two feature sets.

    Returns a dict with ``mmd`` (sqrt of the clamped MMD^2), ``mmd2``, their
    standard deviations across subsets, and the estimation settings.
    """
    x = torch.as_tensor(feats_real, dtype=torch.float32)
    y = torch.as_tensor(feats_fake, dtype=torch.float32)

    if standardize:
        # Whiten with the *real* statistics so the kernel is not dominated by
        # the few Inception dimensions with the largest raw scale.
        mu = x.mean(0, keepdim=True)
        sd = x.std(0, keepdim=True).clamp_min(1e-6)
        x = (x - mu) / sd
        y = (y - mu) / sd

    dev = torch.device(device if torch.cuda.is_available()
                       and "cuda" in str(device) else "cpu")
    x, y = x.to(dev), y.to(dev)

    sigma2 = median_bandwidth(x, y, seed=seed)
    sigma2_list = [sigma2 * s for s in bandwidth_scales]

    m = min(subset_size, x.shape[0], y.shape[0])
    if m < 4:
        raise ValueError("need at least 4 samples per set to estimate MMD")

    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        xi = torch.randperm(x.shape[0], generator=g)[:m].to(dev)
        yi = torch.randperm(y.shape[0], generator=g)[:m].to(dev)
        vals.append(mmd2_rbf_unbiased(x[xi], y[yi], sigma2_list))

    vals = np.asarray(vals, dtype=np.float64)
    # The kernel is a sum of len(scales) RBFs; normalise so MMD^2 is in [0, 1]-ish
    vals = vals / float(len(sigma2_list))
    mmd2 = float(vals.mean())
    return {
        "mmd2": mmd2,
        "mmd2_std": float(vals.std()),
        "mmd": float(np.sqrt(max(mmd2, 0.0))),
        "mmd_std": float(np.sqrt(np.maximum(vals, 0.0)).std()),
        "kernel": "rbf-multiscale",
        "sigma2_median": float(sigma2),
        "bandwidth_scales": list(bandwidth_scales),
        "subset_size": int(m),
        "n_subsets": int(n_subsets),
        "standardized": bool(standardize),
    }


# --------------------------------------------------------------------------- #
#  KID (polynomial-kernel MMD^2) -- cheap companion to FID, reported alongside
# --------------------------------------------------------------------------- #
def compute_kid(feats_real, feats_fake, subset_size=1000, n_subsets=100,
                device="cpu", seed=0):
    x = torch.as_tensor(feats_real, dtype=torch.float64)
    y = torch.as_tensor(feats_fake, dtype=torch.float64)
    dev = torch.device(device if torch.cuda.is_available()
                       and "cuda" in str(device) else "cpu")
    x, y = x.to(dev), y.to(dev)
    d = x.shape[1]
    m = min(subset_size, x.shape[0], y.shape[0])
    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        xi = torch.randperm(x.shape[0], generator=g)[:m].to(dev)
        yi = torch.randperm(y.shape[0], generator=g)[:m].to(dev)
        a, b = x[xi], y[yi]
        kxx = (a @ a.t() / d + 1.0) ** 3
        kyy = (b @ b.t() / d + 1.0) ** 3
        kxy = (a @ b.t() / d + 1.0) ** 3
        kxx.fill_diagonal_(0.0)
        kyy.fill_diagonal_(0.0)
        vals.append(float(kxx.sum() / (m * (m - 1))
                          + kyy.sum() / (m * (m - 1))
                          - 2.0 * kxy.mean()))
    vals = np.asarray(vals)
    return {"kid": float(vals.mean()), "kid_std": float(vals.std()),
            "subset_size": int(m), "n_subsets": int(n_subsets)}


# --------------------------------------------------------------------------- #
#  t-SNE
# --------------------------------------------------------------------------- #
def tsne_embedding(feats_real, feats_fake, n_samples=2000, pca_dim=50,
                   perplexity=30.0, seed=0, n_iter=1000):
    """PCA -> t-SNE joint embedding of real and generated Inception features."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.RandomState(seed)
    nr = min(n_samples, feats_real.shape[0])
    nf = min(n_samples, feats_fake.shape[0])
    ir = rng.permutation(feats_real.shape[0])[:nr]
    if_ = rng.permutation(feats_fake.shape[0])[:nf]

    X = np.concatenate([feats_real[ir], feats_fake[if_]], axis=0).astype(np.float64)
    is_real = np.concatenate([np.ones(nr, bool), np.zeros(nf, bool)])

    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    pca_dim = min(pca_dim, X.shape[0] - 1, X.shape[1])
    X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    perplexity = float(min(perplexity, max(5.0, (X.shape[0] - 1) / 3.0)))
    try:
        ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate="auto", random_state=seed, max_iter=n_iter)
    except TypeError:  # scikit-learn < 1.5 uses n_iter
        ts = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate="auto", random_state=seed, n_iter=n_iter)
    emb = ts.fit_transform(X)
    return emb, is_real, ir, if_


def plot_tsne(emb, is_real, out_path, title="t-SNE of Inception features",
              real_label="Real", fake_label="GFM samples", dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(emb[is_real, 0], emb[is_real, 1], s=5, alpha=0.55,
               c="#2E5FA3", linewidths=0, label=real_label)
    ax.scatter(emb[~is_real, 0], emb[~is_real, 1], s=5, alpha=0.55,
               c="#D1495B", linewidths=0, label=fake_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=9, markerscale=2.5, frameon=False, loc="best")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tsne_by_class(emb, is_real, labels_real, class_names, out_path,
                       title="t-SNE by class", dpi=200):
    """Real points coloured by ground-truth class, generated points in grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(emb[~is_real, 0], emb[~is_real, 1], s=5, alpha=0.35,
               c="#9AA0A6", linewidths=0, label="GFM samples")
    cmap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")
    real_emb = emb[is_real]
    for c, name in enumerate(class_names):
        sel = labels_real == c
        if not sel.any():
            continue
        ax.scatter(real_emb[sel, 0], real_emb[sel, 1], s=5, alpha=0.7,
                   color=cmap(c % cmap.N), linewidths=0, label=name)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, markerscale=2.0, frameon=False,
              loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    return path
