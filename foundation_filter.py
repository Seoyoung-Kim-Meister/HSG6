"""
foundation_filter.py
--------------------
Hard-filter foundations for a given individual before semantic matching.

Filters applied in order:
  1. ausgelastet = 1  → foundation is full, remove immediately
  2. GESUCHSTELLENDE  → if individual needs Privatpersonen access,
                        remove institution-only foundations
  3. EINREICHUNGSTERMIN → if the individual needs funding by a specific
                          date, remove foundations whose next deadline
                          has already passed

Usage example at the bottom of this file.
"""

import re
import pandas as pd
from datetime import date, datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Load & merge data
# ---------------------------------------------------------------------------

def load_foundations(
    descriptions_path: str = "fonds_stiftungen_descriptions.csv",
    foundations_path: str  = "foundations.csv",
) -> pd.DataFrame:
    """
    Merge the two CSVs on foundation name so every row has both the
    structured binary flags (ausgelastet, category columns) and the
    free-text fields (description, GESUCHSTELLENDE, EINREICHUNGSTERMIN).
    """
    desc = pd.read_csv(descriptions_path)
    found = pd.read_csv(foundations_path)

    # Normalise the binary columns in foundations.csv (X → 1, blank → 0)
    binary_cols = [c for c in found.columns if c not in ["Name des Fonds/Stiftung", "Seite"]]
    for col in binary_cols:
        found[col] = (found[col] == "X").astype(int)

    # binary_cols already contains "ausgelastet", so no need to list it separately
    merged = desc.merge(
        found[["Name des Fonds/Stiftung"] + binary_cols],
        left_on="name",
        right_on="Name des Fonds/Stiftung",
        how="left",
    )

    # Foundations only in the descriptions file won't have ausgelastet → treat as 0
    merged["ausgelastet"] = merged["ausgelastet"].fillna(0).astype(int)

    return merged


# ---------------------------------------------------------------------------
# 2. Filter 1 — ausgelastet
# ---------------------------------------------------------------------------

def filter_ausgelastet(df: pd.DataFrame) -> pd.DataFrame:
    """Remove foundations that are marked as fully booked."""
    before = len(df)
    df = df[df["ausgelastet"] == 0].copy()
    print(f"[Filter 1 – ausgelastet]  {before - len(df):3d} removed  →  {len(df)} remaining")
    return df


# ---------------------------------------------------------------------------
# 3. Filter 2 — GESUCHSTELLENDE
# ---------------------------------------------------------------------------

# All values that indicate a foundation accepts private persons
_PRIVATPERSONEN_VALUES = {
    "privatpersonen",
    "privatpersonen, institutionen",
    "privatpersonen, aber nur über institutionen angefragt",
}

def accepts_privatpersonen(gesuchstellende: str) -> bool:
    """Return True if the foundation accepts private-person applicants."""
    if pd.isna(gesuchstellende):
        return True   # Unknown → keep (conservative)
    return gesuchstellende.strip().lower() in _PRIVATPERSONEN_VALUES


def filter_gesuchstellende(
    df: pd.DataFrame,
    needs_privatpersonen: bool,
) -> pd.DataFrame:
    """
    If the individual is a private person (needs_privatpersonen=True),
    remove institution-only foundations.
    If the individual is an institution (needs_privatpersonen=False),
    no filtering is applied here.
    """
    if not needs_privatpersonen:
        print(f"[Filter 2 – GESUCHSTELLENDE]  skipped (applicant is an institution)")
        return df

    before = len(df)
    df = df[df["GESUCHSTELLENDE"].apply(accepts_privatpersonen)].copy()
    print(f"[Filter 2 – GESUCHSTELLENDE]  {before - len(df):3d} removed  →  {len(df)} remaining")
    return df


# ---------------------------------------------------------------------------
# 4. Filter 3 — EINREICHUNGSTERMIN
# ---------------------------------------------------------------------------

# Phrases that clearly indicate the foundation accepts applications year-round
_OPEN_YEAR_ROUND_PHRASES = [
    "ganzes jahr", "jederzeit", "laufend", "ganzjährig",
]

# Month name → month number (German)
_MONTH_MAP = {
    "januar": 1, "februar": 2, "märz": 3, "april": 4,
    "mai": 5, "juni": 6, "juli": 7, "august": 8,
    "september": 9, "oktober": 10, "november": 11, "dezember": 12,
    # Abbreviations
    "jan": 1, "feb": 2, "mär": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "okt": 10, "nov": 11, "dez": 12,
}


def _extract_deadlines(text: str, reference_year: int) -> list[date]:
    """
    Extract all concrete dates from an EINREICHUNGSTERMIN string.
    Returns a list of date objects (possibly empty).

    Handles formats such as:
      - "15.5./15.10."
      - "Ende April, Ende Oktober"
      - "Februar und Oktober"
      - "Ende Februar, Mai, August, November"
      - "Bis 30.06."
      - "26. Januar 2026, 27. April 2026"
      - "1. Quartal"
    """
    dates: list[date] = []
    text_lower = text.lower()

    # --- DD.MM. or DD.MM.YYYY patterns (e.g. "15.5.", "30.06.", "26. Januar 2026") ---
    for m in re.finditer(r"\b(\d{1,2})[.\s]+(\d{1,2})\.?(?:[.\s]+(\d{4}))?", text):
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else reference_year
        try:
            dates.append(date(year, month, day))
        except ValueError:
            pass   # invalid date like 31.11

    # --- "Ende <month>" or bare month names ---
    for word, month_num in _MONTH_MAP.items():
        # Look for "Ende April" → treat as last day of that month
        if re.search(rf"\bende\s+{word}\b", text_lower):
            last_day = (date(reference_year, month_num % 12 + 1, 1) - pd.Timedelta(days=1)).day \
                       if month_num < 12 else 31
            dates.append(date(reference_year, month_num, last_day))
        # Bare month name without a day → treat as 1st of that month
        elif re.search(rf"(?<!\w){word}(?!\w)", text_lower):
            dates.append(date(reference_year, month_num, 1))

    # --- "1. Quartal" / "2. Quartal" etc. ---
    q_match = re.search(r"(\d)\.\s*quartal", text_lower)
    if q_match:
        q = int(q_match.group(1))
        month = q * 3   # Q1→Mar, Q2→Jun, Q3→Sep, Q4→Dec
        dates.append(date(reference_year, month, 31 if month in (3,12) else 30))

    return dates


def has_open_deadline(einreichungstermin: str, needed_by: date) -> bool:
    """
    Return True if the foundation still has at least one deadline
    on or after `needed_by` (i.e. it is still reachable).

    Logic:
      - NaN / missing → keep (conservative)
      - Contains a year-round phrase → keep
      - Otherwise extract concrete dates; keep if ANY date >= needed_by
      - If no dates could be parsed → keep (conservative)
    """
    if pd.isna(einreichungstermin):
        return True

    text = einreichungstermin.strip()
    text_lower = text.lower()

    # Year-round foundations are always open
    if any(phrase in text_lower for phrase in _OPEN_YEAR_ROUND_PHRASES):
        return True

    ref_year = needed_by.year
    deadlines = _extract_deadlines(text, ref_year)

    # If nothing could be parsed, don't remove the foundation
    if not deadlines:
        return True

    # Keep if at least one deadline is still in the future
    return any(d >= needed_by for d in deadlines)


def filter_einreichungstermin(
    df: pd.DataFrame,
    needed_by: Optional[date],
) -> pd.DataFrame:
    """
    Remove foundations whose next deadline falls before `needed_by`.
    If `needed_by` is None, this filter is skipped.
    """
    if needed_by is None:
        print(f"[Filter 3 – EINREICHUNGSTERMIN]  skipped (no date constraint given)")
        return df

    before = len(df)
    mask = df["EINREICHUNGSTERMIN"].apply(
        lambda x: has_open_deadline(x, needed_by)
    )
    df = df[mask].copy()
    print(f"[Filter 3 – EINREICHUNGSTERMIN]  {before - len(df):3d} removed  →  {len(df)} remaining")
    return df


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def apply_hard_filters(
    df: pd.DataFrame,
    needs_privatpersonen: bool = True,
    needed_by: Optional[date] = None,
) -> pd.DataFrame:
    """
    Run all three hard filters in sequence and return the surviving foundations.

    Parameters
    ----------
    df                  : merged foundation DataFrame from load_foundations()
    needs_privatpersonen: True  → individual is a private person
                          False → individual is an institution
    needed_by           : individual needs funding by this date;
                          None  → skip deadline filter
    """
    print(f"\nStarting with {len(df)} foundations")
    print("-" * 50)
    df = filter_ausgelastet(df)
    df = filter_gesuchstellende(df, needs_privatpersonen)
    df = filter_einreichungstermin(df, needed_by)
    print("-" * 50)
    print(f"Final candidate pool: {len(df)} foundations\n")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    foundations = load_foundations(
        descriptions_path="fonds_stiftungen_descriptions.csv",
        foundations_path="foundations.csv",
    )

    # --- Example individual ---
    # Private person who needs funding by 1 June 2026
    candidates = apply_hard_filters(
        foundations,
        needs_privatpersonen=True,
        needed_by=date(2026, 6, 1),
    )

    print("Sample of surviving foundations:")
    print(candidates[["name", "GESUCHSTELLENDE", "EINREICHUNGSTERMIN"]].head(10).to_string(index=False))
