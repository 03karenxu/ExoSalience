from dataclasses import dataclass
import numpy as np

@dataclass
class Quarter:
    quarter: int
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    quality: np.ndarray

@dataclass
class StarCurve:
    kepid: int
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray

@dataclass
class FoldedCurve:
    kepid: int
    phase: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray

@dataclass
class ProcessedData:
    kepid: int
    global_view: np.ndarray
    local_view: np.ndarray
    label: str

    