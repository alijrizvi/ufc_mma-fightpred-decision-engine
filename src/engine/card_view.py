"""FIFA-style rating cards as pure HTML (no Streamlit import, so tests can check it)."""
from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.engine.features import ATTRIBUTES

DEFAULT_DIR = "artifacts/cards"
_BADGE = {"STALE": "#d98e04", "INACTIVE": "#8a8a8a"}
_BOX = "border:2px solid #c9a227;border-radius:14px;padding:14px;background:#161a23;color:#fff;"


def load_cards(directory: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Read cards.json into {fighter: card}; empty dict if not published yet."""
    path = Path(directory or os.environ.get("CARDS_DIR", DEFAULT_DIR)) / "cards.json"
    if not path.exists():
        return {}
    return {c["fighter"]: c for c in json.loads(path.read_text())}


def _initials(name: str) -> str:
    return "".join(p[0] for p in name.split()[:2]).upper() or "?"


def _column(card: Mapping[str, Any], attrs: list) -> str:
    rows = "".join(
        f'<div style="margin:6px 0" title="{"Behavioral proxy" if a == "PSY" else a}">'
        f'<b style="font-size:22px">{int(card[a])}</b><br>'
        f'<span style="font-size:11px;opacity:.7">{a}</span></div>'
        for a in attrs
    )
    return f'<div style="flex:1;text-align:center">{rows}</div>'


def render_card_html(card: Optional[Mapping[str, Any]], name: str = "") -> str:
    """One card: avatar, name, badges, then two columns of three stats."""
    if card is None:
        return (f'<div style="{_BOX}text-align:center">{escape(name)}<br>'
                '<small>No rating card yet (not in the showcase roster).</small></div>')
    fighter = str(card["fighter"])
    badges = ""
    status = str(card.get("status", "ACTIVE"))
    if status in _BADGE:
        badges += (f'<span style="background:{_BADGE[status]};border-radius:6px;'
                   f'padding:1px 8px;font-size:11px;margin-right:4px">{status}</span>')
    if card.get("is_provisional"):
        badges += ('<span style="background:#3b6fd4;border-radius:6px;padding:1px 8px;'
                   'font-size:11px">PROVISIONAL</span>')
    avatar = ('<div style="width:72px;height:72px;border-radius:50%;background:#2b3245;'
              'margin:0 auto 6px;line-height:72px;font-size:26px;text-align:center">'
              f'{escape(_initials(fighter))}</div>')
    cols = _column(card, ATTRIBUTES[:3]) + _column(card, ATTRIBUTES[3:])
    return (f'<div style="{_BOX}">{avatar}'
            f'<div style="text-align:center"><b>{escape(fighter)}</b><br>'
            f'<small>{escape(str(card.get("division", "")))}</small><br>{badges}</div>'
            f'<div style="display:flex;margin-top:8px">{cols}</div></div>')
