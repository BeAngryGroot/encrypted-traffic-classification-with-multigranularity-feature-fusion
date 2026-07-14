"""ISCXTor 数据集的统一标签定义。

主实验固定为八类应用分类；Tor/Non-Tor 只作为辅助任务。这里保留 VPN、
QUIC 等旧数据标识的兼容解析，但它们不会进入 ISCXTor 两个正式任务。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


APPLICATION_LABELS = (
    "BROWSING",
    "EMAIL",
    "CHAT",
    "AUDIO",
    "VIDEO",
    "FILE",
    "VOIP",
    "P2P",
)

_APPLICATION_ALIASES = {
    "BROWSING": {"BROWSING", "BROWSER", "WEB"},
    "EMAIL": {"EMAIL", "MAIL"},
    "CHAT": {"CHAT", "IM", "MESSAGING"},
    "AUDIO": {"AUDIO", "AUDIOSTREAMING"},
    "VIDEO": {"VIDEO", "VIDEOSTREAMING", "STREAMING"},
    "FILE": {"FILE", "FTP", "FILETRANSFER", "TRANSFER"},
    "VOIP": {"VOIP", "VOICE", "SKYPE"},
    "P2P": {"P2P", "BITTORRENT", "TORRENT"},
}


@dataclass(frozen=True)
class LabelInfo:
    """单个源文件对应的传输类型与应用类型。"""

    primary: str
    application: str

    @property
    def combined(self) -> str:
        return f"{self.primary}:{self.application}"


def _normalized_text(path: Path) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(path.with_suffix("")).upper())


def _tokens(path: Path) -> set[str]:
    text = str(path.with_suffix("")).upper()
    return {token for token in re.split(r"[^A-Z0-9]+", text) if token}


def infer_labels(path: Path) -> LabelInfo:
    """从目录和文件名推断标签，未知值显式标记而不做猜测。

    `NONTOR` 中包含 `TOR`，因此必须先识别 Non-Tor，避免辅助任务标签泄漏。
    """

    compact = _normalized_text(path)
    tokens = _tokens(path)
    if "NONTOR" in compact:
        primary = "NONTOR"
    elif "TOR" in tokens or compact.startswith("TOR"):
        primary = "TOR"
    elif "NONVPN" in compact:
        primary = "NONVPN"
    elif "VPN" in tokens or "VPN" in compact:
        primary = "VPN"
    elif "QUIC" in tokens:
        primary = "QUIC"
    else:
        primary = "OTHER"

    application = "UNKNOWN"
    for canonical in APPLICATION_LABELS:
        aliases = _APPLICATION_ALIASES[canonical]
        # 短别名（例如 IM）只允许完整词命中，避免在 EXPERIMENT 等路径词中误报。
        if any(alias in tokens or (len(alias) >= 4 and alias in compact) for alias in aliases):
            application = canonical
            break
    return LabelInfo(primary=primary, application=application)
