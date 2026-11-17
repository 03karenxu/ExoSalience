"""
build train.npz / test.npz from the TCE catalogue and Kepler light curves

    uv run python -m src.preprocessing.build_dataset                         # full catalogue from S3
    uv run python -m src.preprocessing.build_dataset --mast data/test       # local stars only
    uv run python -m src.preprocessing.build_dataset --limit 50 --workers 4
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.datatypes import ModelInput
from src.preprocessing.load import from_dir, from_s3
from src.preprocessing.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"PC": 1, "AFP": 0, "NTP": 0}
CHUNK_SIZE = 500


def _split_stars(kepids: list[int], test_fraction: float = 0.3, seed: int = 123) -> tuple[list[int], list[int]]:
    """shuffle stars with a fixed seed and split them, so all of a star's TCEs land on the same side"""
    shuffled = np.random.default_rng(seed).permutation(kepids).tolist()
    n_test = round(len(shuffled) * test_fraction)
    return shuffled[n_test:], shuffled[:n_test]


def _process_star(kepid: int, tces: pd.DataFrame, mast_dir: Path | None) -> list[ModelInput]:
    """views for every TCE on the star (all are masked), keeping only the labelled ones"""
    quarters = from_dir(kepid, mast_dir) if mast_dir else from_s3(kepid)
    labels = dict(zip(tces.tce_plnt_num, tces.av_training_set))
    return [
        ModelInput(kepid=int(kepid), tce_plnt_num=int(n), global_view=g, local_view=l, label=LABELS[labels[n]])
        for n, g, l in preprocess(kepid, quarters, tces)
        if labels[n] in LABELS
    ]


def _try_process_star(kepid: int, tces: pd.DataFrame, mast_dir: Path | None) -> tuple[list[ModelInput], str | None]:
    try:
        return _process_star(kepid, tces, mast_dir), None
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def _save(examples: list[ModelInput], path: Path):
    """one array per ModelInput field; fields that are None (e.g. tabular) are left out"""
    arrays = {}
    for f in fields(ModelInput):
        values = [getattr(e, f.name) for e in examples]
        if examples and all(v is not None for v in values):
            arrays[f.name] = np.stack(values)
    np.savez_compressed(path, **arrays)


def _merge(chunk_paths: list[Path], path: Path):
    chunks = [dict(np.load(p)) for p in chunk_paths]
    chunks = [c for c in chunks if c]
    np.savez_compressed(path, **{key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]})


def main(tce_file: Path, mast_dir: Path | None = None, limit: int | None = None, workers: int = 8,
         out_dir: Path = ROOT / "data" / "processed"):
    catalogue = pd.read_csv(tce_file)
    kepids = sorted(catalogue[catalogue.av_training_set.isin(LABELS)].kepid.unique().tolist())
    if mast_dir:
        on_disk = {int(d.name) for d in Path(mast_dir).glob("*/*")}
        kepids = [k for k in kepids if k in on_disk]
    kepids = kepids[:limit]
    star_tces = dict(tuple(catalogue[catalogue.kepid.isin(kepids)].groupby("kepid")))

    run_id = hashlib.md5(f"{kepids}{mast_dir}".encode()).hexdigest()[:8]
    chunk_dir = Path(out_dir) / "chunks" / run_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(kepids)} stars, chunks in {chunk_dir}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, stars in zip(("train", "test"), _split_stars(kepids)):
            chunk_paths = []
            for i in range(0, len(stars), CHUNK_SIZE):
                path = chunk_dir / f"{split}_{i // CHUNK_SIZE:04d}.npz"
                chunk_paths.append(path)
                if path.exists():
                    continue
                batch = stars[i:i + CHUNK_SIZE]
                results = list(pool.map(_try_process_star, batch, [star_tces[k] for k in batch], [mast_dir] * len(batch)))
                failed = [(k, err) for k, (_, err) in zip(batch, results) if err]
                if failed:
                    pd.DataFrame(failed, columns=["kepid", "error"]).to_csv(
                        chunk_dir / "failed.csv", mode="a", index=False, header=not (chunk_dir / "failed.csv").exists())
                _save([e for examples, _ in results for e in examples], path)
                print(f"{split} chunk {i // CHUNK_SIZE}: {len(batch)} stars, {len(failed)} failed")
            _merge(chunk_paths, Path(out_dir) / f"{split}.npz")
            print(f"wrote {Path(out_dir) / f'{split}.npz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tce", type=Path, default=ROOT / "data" / "dr24-tce-2.csv", help="TCE catalogue csv")
    parser.add_argument("--mast", type=Path, default=None, help="local MAST-layout dir; default streams from S3")
    parser.add_argument("--limit", type=int, default=None, help="only the first N labelled stars")
    parser.add_argument("--workers", type=int, default=8, help="stars processed in parallel")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed", help="output dir")
    args = parser.parse_args()
    main(args.tce, args.mast, args.limit, args.workers, args.out)
