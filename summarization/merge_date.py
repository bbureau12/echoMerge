# services/date_merge.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import datetime as dt
import re

import dateparser
from dateparser.search import search_dates

MORNING = (9, 0)     # 09:00
AFTERNOON = (15, 0)  # 15:00
EVENING = (19, 0)    # 19:00
NIGHT = (22, 0)      # 22:00

# light phrase cues -> time anchors (only used if phrase present and no time parsed)
PHRASE_TO_TIME = [
    (re.compile(r"\b(this\s+)?morning\b", re.I), MORNING),
    (re.compile(r"\b(this\s+)?afternoon\b", re.I), AFTERNOON),
    (re.compile(r"\b(this\s+)?evening\b", re.I), EVENING),
    (re.compile(r"\btonight\b", re.I), NIGHT),
]

ISO_RE = re.compile(
    r"\b(?P<date>\d{4}-\d{2}-\d{2})(?:[ T](?P<time>\d{2}:\d{2}(?::\d{2})?))?\b"
)

@dataclass
class ProximityResult:
    same_day: bool
    within_hours: bool
    best_pair: Optional[Tuple[dt.datetime, dt.datetime]]
    a_dates: List[dt.datetime]
    b_dates: List[dt.datetime]

def _to_aware(d: dt.datetime, tz: dt.tzinfo) -> dt.datetime:
    if d.tzinfo is None:
        return d.replace(tzinfo=tz)
    return d.astimezone(tz)

def _parse_all(text: str, ref: dt.datetime, tz: dt.tzinfo) -> List[dt.datetime]:
    """Return all parsed datetimes found in text, normalized to tz."""
    results: List[dt.datetime] = []

    # 1) dateparser.search (handles 'today', 'this morning', 'Sept 7', '09:00', etc.)
    settings = {
        "RELATIVE_BASE": ref,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "current_period",
        "PREFER_DAY_OF_MONTH": "current",
        "SKIP_TOKENS": ["at", "on", "the"],  # helps some edge cases
    }
    try:
        found = search_dates(text, settings=settings) or []
        for _, d in found:
            results.append(_to_aware(d, tz))
    except Exception:
        pass

    # 2) ISO regex fallback for super explicit patterns
    for m in ISO_RE.finditer(text):
        date_s = m.group("date")
        time_s = m.group("time")
        try:
            if time_s:
                d = dt.datetime.fromisoformat(f"{date_s} {time_s}")
            else:
                d = dt.datetime.fromisoformat(date_s)
            results.append(_to_aware(d, tz))
        except Exception:
            continue

    # 3) If we saw an unspecific cue like “this morning” but parser didn’t add time,
    #    synthesize one on the REF DATE.
    lowered = text.lower()
    if any(k in lowered for k in ["today", "this morning", "this afternoon", "this evening", "tonight"]):
        # ensure at least one anchor on the same day:
        # If nothing got parsed to the same calendar day, create one.
        ref_day_any = any((r.date() == ref.date()) for r in results)
        if not ref_day_any:
            # pick a time by phrase, default to 09:00
            hh, mm = MORNING
            for rx, (H, M) in PHRASE_TO_TIME:
                if rx.search(text):
                    hh, mm = H, M
                    break
            synth = ref.replace(hour=hh, minute=mm, second=0, microsecond=0)
            results.append(_to_aware(synth, tz))

    # Deduplicate (keep earliest instance per identical timestamp)
    # sort for stability
    uniq = sorted(set(r.replace(microsecond=0) for r in results))
    return uniq

def same_day_or_close(
    a: str,
    b: str,
    *,
    a_ref: Optional[dt.datetime] = None,
    b_ref: Optional[dt.datetime] = None,
    hours_window: int = 12,
    tz: dt.tzinfo = dt.timezone.utc,
) -> ProximityResult:
    """
    Parse temporal cues from A and B. Decide if they refer to the same calendar day
    or at least within a small hours window.
    """
    now = dt.datetime.now(tz)
    a_ref = a_ref or now
    b_ref = b_ref or now

    A = _parse_all(a, a_ref, tz)
    B = _parse_all(b, b_ref, tz)

    # Best pair (min absolute delta)
    best_pair = None
    best_delta = None
    for da in A:
        for db in B:
            delta = abs((da - db).total_seconds())
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_pair = (da, db)

    same_day = False
    within_hours = False
    if best_pair:
        da, db = best_pair
        same_day = (da.date() == db.date())
        within_hours = (abs((da - db).total_seconds()) <= hours_window * 3600)

    return ProximityResult(
        same_day=same_day,
        within_hours=within_hours,
        best_pair=best_pair,
        a_dates=A,
        b_dates=B,
    )