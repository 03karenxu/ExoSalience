"""
build train.npz / test.npz from a catalogue of signals (DR24 TCE catalogue or KOI table) and Kepler light curves on S3

usage: uv run python -m src.preprocessing.build_dataset --input <csv path inside data/> [--name <name>] [--limit <n>] [--workers <n>]
writes to data/processed/<name>/, where name defaults to the input file name:
    --input tce_catalogue.csv  ->  data/processed/tce_catalogue/train.npz, test.npz
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DATA_DIR, PROCESSED_DIR
from src.preprocessing.datatypes import ModelInput
from src.preprocessing.load import from_s3
from src.preprocessing.preprocess import preprocess

LABELS = {
    "av_training_set": {"PC": 1, "AFP": 0, "NTP": 0},          # DR24 TCE catalogue; drops UNK
    "koi_disposition": {"CONFIRMED": 1, "FALSE POSITIVE": 0},  # KOI table; drops CANDIDATE
}
# each catalogue's columns renamed to the generic names
COLUMNS = {
    "tce_period": "period", "tce_time0bk": "t0", "tce_duration": "duration", "tce_plnt_num": "planet_num", # TCE
    "koi_period": "period", "koi_time0bk": "t0", "koi_duration": "duration", "koi_tce_plnt_num": "planet_num",  # KOI
}
REQUIRED = ["kepid", "period", "t0", "duration", "planet_num"]
CHUNK_SIZE = 500


def load_catalogue(path: Path) -> pd.DataFrame:
    """
        reads a csv catalogue into a cleaned pandas dataframe
        - renames columns to generic ones
        - asserts all required columns exist
        - maps labels to 1/0 -> some labels are dropped to NaN, to be filtered out later
    """
    catalogue = pd.read_csv(path, comment="#").rename(columns=COLUMNS)
    missing = [c for c in REQUIRED if c not in catalogue]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    for label_column, label_map in LABELS.items():
        if label_column in catalogue:
            catalogue["is_exoplanet"] = catalogue[label_column].map(label_map)  # 1, 0, or NaN = unlabelled
            return catalogue
    raise ValueError(f"{path} has none of the label columns {list(LABELS)}")


def _select_stars(catalogue: pd.DataFrame, limit: int | None) -> list[int]:
    """sorted ids of stars with at least one labelled signal"""
    kepids = sorted(catalogue.loc[catalogue.is_exoplanet.notna(), "kepid"].unique().tolist())
    return kepids[:limit]


def _split_stars(kepids: list[int], test_fraction: float = 0.3, seed: int = 123) -> tuple[list[int], list[int]]:
    """shuffle stars, then split such that all of a star's signals land in the same set"""
    shuffled = np.random.default_rng(seed).permutation(kepids).tolist()
    n_test = round(len(shuffled) * test_fraction)
    return shuffled[n_test:], shuffled[:n_test]


def _process_star(kepid: int, signals: pd.DataFrame) -> tuple[list[ModelInput], str | None]:
    """
    create views for a star's labelled signals. one preprocessing step requires all of a star's
    signals, even unlabelled ones. unlabelled signals are dropped once the step is done.
    """
    try:
        views = preprocess(kepid, from_s3(kepid), signals)
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

    # add is_exoplanet labelled, joined on planet_num
    is_exoplanet = dict(zip(signals.planet_num, signals.is_exoplanet))
    model_inputs = []
    for planet_num, global_view, local_view in views:
        # do not send to model if is_exoplanet label is unknown
        if pd.notna(is_exoplanet[planet_num]):
            model_inputs.append(ModelInput(
                kepid=int(kepid),
                planet_num=int(planet_num),
                global_view=global_view,
                local_view=local_view,
                label=int(is_exoplanet[planet_num])
            ))
    return model_inputs, None


def _process_chunk(pool: ProcessPoolExecutor, kepids: list[int], star_signals: dict
                   ) -> tuple[list[ModelInput], list[tuple[int, str]]]:
    """process stars in parallel; returns all their examples and a (kepid, error) per failed star"""
    results = pool.map(_process_star, kepids, [star_signals[k] for k in kepids])
    examples, failed = [], []
    for kepid, (star_examples, error) in zip(kepids, results):
        examples += star_examples
        if error:
            failed.append((kepid, error))
    return examples, failed


def _log_failures(failed: list[tuple[int, str]], path: Path):
    """append failed stars to a csv, writing the header only when the file is new"""
    if failed:
        pd.DataFrame(failed, columns=["kepid", "error"]).to_csv(path, mode="a", index=False, header=not path.exists())


def _save(examples: list[ModelInput], path: Path):
    """one array per ModelInput field (tabular is never set, so it's left out)"""
    if not examples:
        np.savez_compressed(path)  # empty file still marks the chunk as done
        return
    np.savez_compressed(
        path,
        kepid=np.array([e.kepid for e in examples]),
        planet_num=np.array([e.planet_num for e in examples]),
        global_view=np.stack([e.global_view for e in examples]),
        local_view=np.stack([e.local_view for e in examples]),
        label=np.array([e.label for e in examples]),
    )


def _merge(chunk_paths: list[Path], path: Path):
    """concatenate chunk files into one npz, skipping empty chunks"""
    chunks = [np.load(p) for p in chunk_paths]
    chunks = [c for c in chunks if c.files]
    np.savez_compressed(path, **{key: np.concatenate([c[key] for c in chunks]) for key in chunks[0].files})


def main(input_file: Path, name: str | None = None, limit: int | None = None, workers: int = 8,
         out_dir: Path | None = None):
    """input_file is relative to data/; out_dir overrides data/processed/<name> (used by tests)"""
    catalogue = load_catalogue(DATA_DIR / input_file)
    out_dir = out_dir or PROCESSED_DIR / (name or Path(input_file).stem)

    kepids = _select_stars(catalogue, limit)
    star_signals = dict(list(catalogue.groupby("kepid")))

    # chunks are cached per star list, so a rerun with the same stars resumes instead of starting over
    run_id = hashlib.md5(f"{kepids}".encode()).hexdigest()[:8]
    chunk_dir = Path(out_dir) / "chunks" / run_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(kepids)} stars, chunks in {chunk_dir}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for split, stars in zip(("train", "test"), _split_stars(kepids)):
            chunks = [stars[i:i + CHUNK_SIZE] for i in range(0, len(stars), CHUNK_SIZE)]
            chunk_paths = [chunk_dir / f"{split}_{n:04d}.npz" for n in range(len(chunks))]

            for n, (chunk, path) in enumerate(zip(chunks, chunk_paths)):
                if path.exists():
                    continue  # finished in an earlier run
                examples, failed = _process_chunk(pool, chunk, star_signals)
                _log_failures(failed, chunk_dir / "failed.csv")
                _save(examples, path)
                print(f"{split} chunk {n}: {len(chunk)} stars, {len(failed)} failed")

            out_path = Path(out_dir) / f"{split}.npz"
            _merge(chunk_paths, out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="csv of signals inside data/, e.g. tce_catalogue.csv (DR24 TCE catalogue or KOI table)")
    parser.add_argument("--name", default=None, help="output folder inside data/processed; default: input file name")
    parser.add_argument("--limit", type=int, default=None, help="only the first N labelled stars")
    parser.add_argument("--workers", type=int, default=8, help="stars processed in parallel")
    args = parser.parse_args()
    main(args.input, args.name, args.limit, args.workers)
