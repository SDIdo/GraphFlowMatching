"""
ImageNet stored as HuggingFace-style parquet shards, e.g.

    data/train-00000-of-00294.parquet
    data/train-00001-of-00294.parquet
    ...

rather than as a ``train/<wnid>/*.JPEG`` folder tree. This is what
``load_dataset("imagenet-1k")`` writes to disk, and it is increasingly how
ImageNet is staged on shared cluster storage.

Each row is ``{"image": {"bytes": <encoded JPEG>, "path": <str>}, "label": int}``
(column names vary a little between dumps; the scanner adapts).

Design
------
The expensive part of a parquet dataset is decoding. Two things keep it cheap:

* **Index building reads only the leaf columns it needs** (``label`` and, when
  present, the filename), never the image bytes. Parquet is columnar, so
  scanning 1.28M labels across 294 shards costs seconds, not the 150 GB the
  shards occupy.
* **Row groups are cached LRU**, and the batch sampler keeps a batch inside one
  row group, so decoding is amortised -- the same trick the latent shards use.

For ImageNet-LT, :func:`match_split_to_rows` maps the official
``ImageNet_LT_train.txt`` onto parquet rows by filename. If the shards carry no
filenames, that is impossible and the caller must fall back to reconstructing a
long-tailed subset from the labels, which is *not* the official split.
"""

import glob
import io
import os
import random
from collections import Counter, OrderedDict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from datasets.imagenet_dataset import center_crop_arr

_EMPTY = torch.empty(0, dtype=torch.float32)

# Column names seen in the wild, most-likely first.
_IMAGE_COLS = ("image", "img", "jpeg", "picture")
_LABEL_COLS = ("label", "labels", "class", "target", "fine_label")
_NAME_LEAVES = ("path", "filename", "file_name", "name", "image_id", "id")


def is_parquet_dir(root, pattern="*.parquet"):
    """True if ``root`` looks like a parquet dump rather than a folder tree."""
    if not root or not os.path.isdir(root):
        return False
    if glob.glob(os.path.join(root, pattern)):
        return True
    # HF often nests the shards one level down, in data/.
    return bool(glob.glob(os.path.join(root, "data", pattern)))


def find_parquet_shards(root, split="train", pattern="*.parquet"):
    """Return the sorted shard paths for ``split`` (falls back to all files)."""
    files = sorted(glob.glob(os.path.join(root, pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "data", pattern)))
    if not files:
        raise FileNotFoundError(f"no {pattern} under {root} or {root}/data")
    sel = [f for f in files if split in os.path.basename(f).lower()]
    return sel or files


def _pick(names, candidates):
    lowered = {n.lower(): n for n in names}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def describe_schema(shard):
    """Return (image_col, label_col, name_leaf) for one shard."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard)
    names = pf.schema_arrow.names
    image_col = _pick(names, _IMAGE_COLS)
    label_col = _pick(names, _LABEL_COLS)
    # NB: ParquetSchema has no .num_columns -- len() gives the leaf count,
    # and .column(i).path gives the dotted leaf path (e.g. 'image.bytes').
    _sch = pf.metadata.schema
    leaves = [_sch.column(i).path for i in range(len(_sch))]
    name_leaf = None
    for leaf in leaves:
        if leaf.split(".")[-1].lower() in _NAME_LEAVES:
            # Prefer one nested under the image column.
            if image_col and leaf.startswith(image_col + "."):
                name_leaf = leaf
                break
            name_leaf = name_leaf or leaf
    if image_col is None:
        raise ValueError(f"no image column in {shard}: saw {names}")
    if label_col is None:
        raise ValueError(f"no label column in {shard}: saw {names}")
    return image_col, label_col, name_leaf


def _leaf_values(column, leaf):
    """Flatten a column selected by dotted leaf path.

    ``read_row_group(columns=["image.path"])`` reads only that leaf (verified:
    ~0.5% of the bytes of the full struct) but hands it back re-wrapped as
    ``struct<path: string>``, so the Python values are ``{"path": ...}`` dicts.
    Unwrap them to the scalar the caller asked for.
    """
    vals = column.to_pylist()
    if vals and isinstance(vals[0], dict):
        key = leaf.split(".")[-1]
        return [v.get(key) if isinstance(v, dict) else v for v in vals]
    return vals


def build_index(shards, want_names=True, verbose=True):
    """Scan shard metadata + label (and filename) columns only.

    Returns ``(rows, image_col, label_col, name_leaf)`` where ``rows`` is a list
    of ``(shard_idx, row_group, row_in_group, label, name_or_None)``.
    """
    import pyarrow.parquet as pq

    image_col, label_col, name_leaf = describe_schema(shards[0])
    cols = [label_col] + ([name_leaf] if (want_names and name_leaf) else [])

    rows = []
    for si, shard in enumerate(shards):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=cols)
            labels = _leaf_values(tbl.column(0), cols[0])
            names = (_leaf_values(tbl.column(1), cols[1])
                     if len(cols) > 1 else [None] * len(labels))
            for i, (lab, nm) in enumerate(zip(labels, names)):
                rows.append((si, rg, i, int(lab), nm))
        if verbose and (si % 25 == 0 or si == len(shards) - 1):
            print(f"  indexed shard {si + 1}/{len(shards)} "
                  f"({len(rows):,} rows)", flush=True)
    return rows, image_col, label_col, name_leaf


def match_split_to_rows(rows, split_samples, verbose=True):
    """Map an ImageNet-LT split file onto parquet rows by filename.

    ``split_samples`` is the ``[(relpath, label), ...]`` list from
    ``read_lt_split_file``. Matching is on basename, which is unique across
    ImageNet (``n01440764_190.JPEG``).

    Returns ``(selected_rows, n_missing)``.
    """
    by_name = {}
    for r in rows:
        nm = r[4]
        if nm:
            by_name[os.path.basename(str(nm))] = r
    if not by_name:
        raise ValueError("parquet shards carry no filenames; cannot match the "
                         "official ImageNet-LT split")

    selected, missing = [], 0
    for relpath, label in split_samples:
        r = by_name.get(os.path.basename(relpath))
        if r is None:
            missing += 1
            continue
        # Trust the split file's label (it is the canonical wnid ordering).
        selected.append((r[0], r[1], r[2], int(label), r[4]))
    if verbose:
        print(f"[parquet] matched {len(selected):,}/{len(split_samples):,} "
              f"split entries ({missing:,} missing)")
    return selected, missing


class ImageNetParquetDataset(Dataset):
    """Map-style access over parquet shards, returning ``(x0, x1, y)``.

    ``rows`` is the index produced by :func:`build_index` (optionally filtered,
    e.g. to an ImageNet-LT subset).
    """

    def __init__(self, shards, rows, image_col, image_size=256,
                 use_horizontal_flips=False, return_label=True,
                 cache_row_groups=2):
        self.shards = list(shards)
        self.rows = rows
        self.image_col = image_col
        self.image_size = image_size
        self.horizontal_flip = use_horizontal_flips
        self.return_label = return_label
        self.cache_row_groups = max(1, cache_row_groups)
        self._cache = OrderedDict()
        self._files = {}

    # -- introspection -----------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def class_counts(self):
        return Counter(r[3] for r in self.rows)

    # -- io ----------------------------------------------------------------
    def _file(self, si):
        import pyarrow.parquet as pq
        pf = self._files.get(si)
        if pf is None:
            pf = pq.ParquetFile(self.shards[si])
            self._files[si] = pf
        return pf

    def _row_group(self, si, rg):
        key = (si, rg)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        tbl = self._file(si).read_row_group(rg, columns=[self.image_col])
        col = tbl.column(0)
        self._cache[key] = col
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_row_groups:
            self._cache.popitem(last=False)
        return col

    @staticmethod
    def _to_bytes(cell):
        """Pull the encoded image out of a struct/binary parquet cell."""
        if isinstance(cell, dict):
            for k in ("bytes", "data", "content"):
                if k in cell and cell[k] is not None:
                    return cell[k]
            raise ValueError(f"no image bytes in struct with keys {list(cell)}")
        return cell

    def __getitem__(self, idx):
        si, rg, i, label, _ = self.rows[idx]
        col = self._row_group(si, rg)
        raw = self._to_bytes(col[i].as_py())
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            im = center_crop_arr(im, self.image_size)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        if self.horizontal_flip and random.random() < 0.5:
            x = torch.flip(x, dims=[2])
        if self.return_label:
            return _EMPTY, x, int(label)
        return _EMPTY, x


class RowGroupBatchSampler(Sampler):
    """Keep each batch inside one parquet row group.

    Without this, a shuffled sampler makes every item land in a different row
    group and the LRU cache decodes the same groups over and over -- turning a
    sequential columnar read into random IO over 150 GB.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False,
                 seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        base = getattr(dataset, "dataset", dataset)
        indices = getattr(dataset, "indices", None)
        groups = {}
        if indices is None:
            for pos, r in enumerate(base.rows):
                groups.setdefault((r[0], r[1]), []).append(pos)
        else:
            for pos, real in enumerate(indices):
                r = base.rows[real]
                groups.setdefault((r[0], r[1]), []).append(pos)
        self.groups = [v for _, v in sorted(groups.items())]

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        order = list(range(len(self.groups)))
        if self.shuffle:
            rng.shuffle(order)
        for gi in order:
            idxs = list(self.groups[gi])
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(g) // self.batch_size for g in self.groups)
        return sum((len(g) + self.batch_size - 1) // self.batch_size
                   for g in self.groups)
