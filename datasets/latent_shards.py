"""
Sharded storage for pre-encoded Stable-Diffusion VAE latents.

The repository already pre-encodes LSUN into ``chunk_00000.pt`` files; this
module generalises that idea so the same on-disk layout can hold CIFAR-10,
ImageNet and ImageNet-LT latents *together with their class labels*, which the
existing LSUN loader does not carry.

On-disk layout (``<root>/``):

    metadata.json          {"total_samples", "num_chunks", "chunk_size",
                            "latent_shape", "dataset", "image_size", ...}
    labels.npy             int64 [total_samples]  (optional but always written)
    chunk_00000.pt         {"latents": FloatTensor[n, C, S, S],
                            "labels":  LongTensor[n],
                            "start_idx": int}
    chunk_00001.pt
    ...

``ShardedLatentDataset`` keeps a small LRU cache of chunks in RAM and returns
``(x0, x1, y)`` triples, matching the convention of the image datasets.  Pair it
with ``ChunkAwareBatchSampler`` so a batch never straddles two chunks -- that is
what keeps the I/O cost of the LSUN-style layout low.
"""

import json
import os
import random
from collections import Counter, OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

_EMPTY = torch.empty(0, dtype=torch.float32)


def _load_shard(path):
    """Load a chunk file. Shards hold only tensors, so weights_only=True is
    both safe and forward-compatible with torch's changing default."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location="cpu")


class LatentShardWriter:
    """Accumulates latents in memory and flushes fixed-size chunks to disk."""

    def __init__(self, root, chunk_size=8192, storage_dtype=torch.float16,
                 metadata=None):
        self.root = root
        self.chunk_size = chunk_size
        self.storage_dtype = storage_dtype
        self.metadata = dict(metadata or {})
        os.makedirs(root, exist_ok=True)

        self._buf_lat = []
        self._buf_lab = []
        self._buffered = 0
        self.num_chunks = 0
        self.total = 0
        self.all_labels = []
        self.latent_shape = None

    def add(self, latents, labels):
        """latents: [B, C, S, S] on any device; labels: [B] ints."""
        latents = latents.detach().to("cpu", self.storage_dtype)
        labels = torch.as_tensor(labels).detach().cpu().to(torch.int64).view(-1)
        if self.latent_shape is None:
            self.latent_shape = list(latents.shape[1:])
        self._buf_lat.append(latents)
        self._buf_lab.append(labels)
        self._buffered += latents.shape[0]
        self.all_labels.append(labels.numpy())
        while self._buffered >= self.chunk_size:
            self._flush(self.chunk_size)

    def _flush(self, n):
        lat = torch.cat(self._buf_lat, dim=0)
        lab = torch.cat(self._buf_lab, dim=0)
        out_lat, rest_lat = lat[:n], lat[n:]
        out_lab, rest_lab = lab[:n], lab[n:]
        path = os.path.join(self.root, f"chunk_{self.num_chunks:05d}.pt")
        torch.save({"latents": out_lat, "labels": out_lab,
                    "start_idx": self.total}, path)
        self.total += out_lat.shape[0]
        self.num_chunks += 1
        self._buf_lat = [rest_lat] if rest_lat.numel() else []
        self._buf_lab = [rest_lab] if rest_lab.numel() else []
        self._buffered = rest_lat.shape[0] if rest_lat.numel() else 0

    def close(self):
        if self._buffered > 0:
            self._flush(self._buffered)
        labels = np.concatenate(self.all_labels) if self.all_labels else np.zeros(0, np.int64)
        np.save(os.path.join(self.root, "labels.npy"), labels)
        meta = {
            "total_samples": self.total,
            "num_chunks": self.num_chunks,
            "chunk_size": self.chunk_size,
            "latent_shape": self.latent_shape,
            "storage_dtype": str(self.storage_dtype).replace("torch.", ""),
        }
        meta.update(self.metadata)
        with open(os.path.join(self.root, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta


class ShardedLatentDataset(Dataset):
    """Reader for the layout written by :class:`LatentShardWriter`.

    Returns ``(x0, x1, y)`` with ``x0`` an empty placeholder, ``x1`` the latent
    cast to float32, and ``y`` the class index (``-1`` when unlabelled).
    """

    def __init__(self, root, cache_chunks=2, return_label=True,
                 to_float32=True):
        self.root = root
        self.cache_chunks = max(1, cache_chunks)
        self.return_label = return_label
        self.to_float32 = to_float32

        meta_path = os.path.join(root, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{meta_path} not found. Run datasets/encode_new_datasets.py first.")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.total_samples = int(self.meta["total_samples"])
        self.num_chunks = int(self.meta["num_chunks"])
        self.latent_shape = tuple(self.meta.get("latent_shape", (4, 32, 32)))

        # Trust the first chunk over the metadata, as the LSUN loader does.
        first = _load_shard(os.path.join(root, "chunk_00000.pt"))
        self.full_chunk_size = int(first["latents"].shape[0])
        self.chunk_size = self.full_chunk_size

        labels_path = os.path.join(root, "labels.npy")
        self.labels = (np.load(labels_path) if os.path.exists(labels_path)
                       else np.full(self.total_samples, -1, dtype=np.int64))

        self._cache = OrderedDict()

    # -- introspection -----------------------------------------------------
    def class_counts(self):
        return Counter(int(y) for y in self.labels if int(y) >= 0)

    @property
    def latent_size(self):
        return int(self.latent_shape[-1])

    @property
    def latent_channels(self):
        return int(self.latent_shape[0])

    # -- data --------------------------------------------------------------
    def __len__(self):
        return self.total_samples

    def _load_chunk(self, chunk_idx):
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]
        path = os.path.join(self.root, f"chunk_{chunk_idx:05d}.pt")
        data = _load_shard(path)
        self._cache[chunk_idx] = data
        self._cache.move_to_end(chunk_idx)
        if len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total_samples
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for {self.total_samples}")
        chunk_idx = idx // self.full_chunk_size
        chunk = self._load_chunk(chunk_idx)
        local = idx - chunk_idx * self.full_chunk_size
        x1 = chunk["latents"][local]
        if self.to_float32:
            x1 = x1.float()
        if not self.return_label:
            return _EMPTY, x1
        lab = chunk.get("labels")
        y = int(lab[local]) if lab is not None else int(self.labels[idx])
        return _EMPTY, x1, y


class ChunkAwareBatchSampler(Sampler):
    """Yields batches drawn from a single chunk so the LRU cache never thrashes.

    Generalisation of ``LSUN_Bedrooms_ChunkAwareBatchSampler`` that also works
    on a ``torch.utils.data.Subset`` wrapping a :class:`ShardedLatentDataset`.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False,
                 seed=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        base = getattr(dataset, "dataset", dataset)  # unwrap Subset
        chunk_size = getattr(base, "full_chunk_size", None)
        if chunk_size is None:
            raise ValueError("dataset has no full_chunk_size; not a shard dataset")

        indices = getattr(dataset, "indices", None)
        if indices is None:
            indices = range(len(dataset))
            # Direct dataset: positions are the underlying indices.
            groups = {}
            for pos in indices:
                groups.setdefault(pos // chunk_size, []).append(pos)
        else:
            # Subset: sampler must yield *positions into the Subset*.
            groups = {}
            for pos, real in enumerate(indices):
                groups.setdefault(real // chunk_size, []).append(pos)

        self.groups = [v for _, v in sorted(groups.items())]
        self.n = sum(len(g) for g in self.groups)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(None if self.seed is None else self.seed + self._epoch)
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
