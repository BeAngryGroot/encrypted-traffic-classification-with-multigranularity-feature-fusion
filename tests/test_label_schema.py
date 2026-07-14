from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.label_schema import APPLICATION_LABELS, infer_labels  # noqa: E402


def test_iscxtor_eight_application_labels():
    cases = {
        "Tor/Browsing_packets.csv": ("TOR", "BROWSING"),
        "Tor/Email_packets.csv": ("TOR", "EMAIL"),
        "Tor/P2P_packets.csv": ("TOR", "P2P"),
        "NonTor/Video-Streaming_packets.csv": ("NONTOR", "VIDEO"),
        "Non-Tor/FileTransfer_packets.csv": ("NONTOR", "FILE"),
        "Tor/VOIP_packets.csv": ("TOR", "VOIP"),
    }
    for path, expected in cases.items():
        info = infer_labels(Path(path))
        assert (info.primary, info.application) == expected


def test_application_schema_is_fixed_to_official_eight_classes():
    assert APPLICATION_LABELS == (
        "BROWSING", "EMAIL", "CHAT", "AUDIO",
        "VIDEO", "FILE", "VOIP", "P2P",
    )


def test_nontor_is_checked_before_tor():
    assert infer_labels(Path("NonTor/Chat_packets.csv")).primary == "NONTOR"

