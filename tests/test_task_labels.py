from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.label_schema import APPLICATION_LABELS  # noqa: E402
from model.task_labels import select_task_labels  # noqa: E402


def _mapping(values):
    return {"label_to_id": {name: idx for idx, name in enumerate(values)}}


def test_application8_excludes_unknown_and_keeps_tor_and_nontor():
    primary = np.asarray([0, 1, 0])
    application = np.asarray([0, 1, 2])
    mappings = {
        "primary": _mapping(["TOR", "NONTOR"]),
        "secondary": _mapping(["BROWSING", "EMAIL", "UNKNOWN"]),
    }
    selection = select_task_labels("application8", primary, application, mappings)
    assert selection.class_names == list(APPLICATION_LABELS)
    assert selection.keep_mask.tolist() == [True, True, False]
    assert selection.labels.tolist() == [0, 1]


def test_tor_binary_maps_nontor_to_zero_and_tor_to_one():
    primary = np.asarray([0, 1, 2])
    application = np.asarray([0, 0, 0])
    mappings = {
        "primary": _mapping(["TOR", "NONTOR", "OTHER"]),
        "secondary": _mapping(["CHAT"]),
    }
    selection = select_task_labels("tor_binary", primary, application, mappings)
    assert selection.class_names == ["NONTOR", "TOR"]
    assert selection.keep_mask.tolist() == [True, True, False]
    assert selection.labels.tolist() == [1, 0]

