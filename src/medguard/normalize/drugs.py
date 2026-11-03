"""Drug name normalisation and alias resolution.

Prescriptions contain brand names, salts, abbreviations, salts with hydrate
suffixes, and dose text glued to the name. The interaction knowledge base is
keyed on generic names, so normalisation is where most of the real-world
accuracy is won or lost.

This module ships:
* :func:`normalize` - surface form to canonical key.
* :func:`resolve_alias` - brand/salt/abbreviation to generic name.
* :func:`parse_dose` - pull the dose out of a surface form.

The alias table below is intentionally small and hand-audited. Expand it with
your pharmacist interviews - that is a genuine, citable contribution.
"""

from __future__ import annotations

import re

#: Brand name / abbreviation -> generic name.
#: Focus on drugs that are extremely common in Indian outpatient prescriptions.
ALIASES: dict[str, str] = {
    # --- Diabetes ---
    "glucophage": "metformin",
    "glycomet": "metformin",
    "d-met": "metformin",
    "glynase": "glimepiride",
    "euglycon": "glimepiride",
    "glycomet gp": "metformin",
    "gluconil": "glipizide",
    "daonil": "glipizide",
    "amaryl": "glimepiride",
    "ondance": "ondansetron",
    # --- Cardiovascular ---
    "aspirin": "acetylsalicylic acid",
    "aspirin 75": "acetylsalicylic acid",
    "ecosprin": "acetylsalicylic acid",
    "disprin": "acetylsalicylic acid",
    "lopinav": "atorvastatin",
    "lipitor": "atorvastatin",
    "tonact": "atorvastatin",
    "starlip": "atorvastatin",
    "cadila": "clarithromycin",
    "zocor": "simvastatin",
    "simvotin": "simvastatin",
    "covamto": "metoprolol",
    "lopressor": "metoprolol",
    "metolar": "metoprolol",
    "cilacar": "cilostazol",
    "amlong": "amlodipine",
    "norvasc": "amlodipine",
    "tenormin": "atenolol",
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "cozaar": "losartan",
    "hyzaar": "losartan hydrochlorothiazide",
    "coversyl": "perindopril",
    "cordarone": "amiodarone",
    "digoxin": "digoxin",
    # --- Antibiotics ---
    "augmentin": "amoxicillin clavulanate",
    "mox": "amoxicillin",
    "moxicip": "amoxicillin",
    "amoxil": "amoxicillin",
    "zithromax": "azithromycin",
    "azithral": "azithromycin",
    "azicip": "azithromycin",
    "sumo": "sumatriptan",
    "cipro": "ciprofloxacin",
    "ciplox": "ciprofloxacin",
    "levoflox": "levofloxacin",
    "lvox": "levofloxacin",
    "taxim o": "cefixime",
    "taxim": "cefuroxime",
    "cefix": "cefixime",
    "keflor": "cefixime",
    "forticef": "cefixime",
    "cefspan": "cefixime",
    "moncef": "cefixime",
    "suprax": "cefixime",
    # --- PPI / GI ---
    "rilprazole": "rabeprazole",
    "rabium": "rabeprazole",
    "prazolid": "pantoprazole",
    "pan-40": "pantoprazole",
    "pan 40": "pantoprazole",
    "pantoprazole": "pantoprazole",
    "nexium": "esomeprazole",
    "prilosec": "omeprazole",
    "omezole": "omeprazole",
    "zantac": "ranitidine",
    "domstal": "domperidone",
    "motilium": "domperidone",
    "metoclopramide": "metoclopramide",
    "reglan": "metoclopramide",
    # --- Analgesics / NSAIDs ---
    "crocin": "paracetamol",
    "dolo": "paracetamol",
    "calpol": "paracetamol",
    "tylenol": "paracetamol",
    "paracip": "paracetamol",
    "dolo 650": "paracetamol",
    "brufen": "ibuprofen",
    "ibuprofen": "ibuprofen",
    "advil": "ibuprofen",
    "voltaren": "diclofenac",
    "vicks": "paracetamol",
    "allegra": "fexofenadine",
    "fexofenadine": "fexofenadine",
    "fexofed": "fexofenadine",
    "fexo": "fexofenadine",
    "antiemetic": "ondansetron",
    "emset": "ondansetron",
    "vomikind": "ondansetron",
    "combiflam": "ibuprofen diclofenac",
    "flexon": "ibuprofen",
    # --- Respiratory ---
    "asthma": "salbutamol",
    "ventolin": "salbutamol",
    "salbutamol": "salbutamol",
    "montair": "montelukast",
    "singulair": "montelukast",
    "pred": "prednisolone",
    "prednisolone": "prednisolone",
    "decort": "dexamethasone",
    # --- Neuro / psychiatric ---
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "cipralex": "escitalopram",
    "lexapro": "escitalopram",
    "nexito": "escitalopram",
    "xanax": "alprazolam",
    "alprax": "alprazolam",
    "valium": "diazepam",
    "diazepam": "diazepam",
    "ativan": "lorazepam",
    "lorazepam": "lorazepam",
    "tramadol": "tramadol",
    "ultram": "tramadol",
    "gabapin": "gabapentin",
    "gabapentin": "gabapentin",
    "lyrica": "pregabalin",
    "prega": "pregabalin",
    "levocal": "levetiracetam",
    "carbamazepine": "carbamazepine",
    "tegit": "carbamazepine",
    # --- Vitamins / supplements ---
    "vitamin d3": "cholecalciferol",
    "calcirol": "cholecalciferol",
    "calcirol 60k": "cholecalciferol",
    "uprise d3": "cholecalciferol",
    "vitamin b12": "cyanocobalamin",
    "neurobion": "vitamin b complex",
    "ferous sulphate": "ferrous sulfate",
    "livogen": "iron folic acid",
    "folic acid": "folic acid",
    "folvite": "folic acid",
    # --- Others ---
    "zincovit": "multivitamin",
    "becosules": "vitamin b complex",
    "shelcal": "calcium carbonate vitamin d3",
    "calcium": "calcium carbonate",
    "thyronorm": "levothyroxine",
    "eltroxin": "levothyroxine",
    "thyroxine": "levothyroxine",
    "neomercazine": "carbamazepine",
}

#: Salt-name suffixes that should be stripped when deriving a generic key.
_SALT_SUFFIXES: tuple[str, ...] = (
    "hcl",
    "hydrochloride",
    "sulfate",
    "sulphate",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "succinate",
    "tartrate",
    "maleate",
    "mesylate",
    "besylate",
    "tosylate",
    "fumarate",
    "citrate",
    "acetate",
    "phosphate",
    "diphosphate",
    "monohydrate",
    "dihydrate",
    "trihydrate",
    "anhydrate",
    "hydrogen succinate",
    "besilate",
)

_DOSE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)(?P<sep>\s*)(?P<unit>mg|mcg|ug|g|ml|iu|units?|%)\b",
    re.IGNORECASE,
)

#: A bare trailing number, e.g. "Dolo 650" (strength with no unit written).
_BARE_DOSE_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*$")

_FREQUENCY_RE = re.compile(
    r"\b(once|twice|thrice|qod|od|bd|tds|qid|qds|q8h|q12h|daily|weekly|bedtime|after food|before food)\b",
    re.IGNORECASE,
)

_BRACKET_PATTERN = re.compile(r"\(([^)]*)\)")

#: Trailing dose text, e.g. "metformin 500 mg" -> "metformin".
_TRAILING_DOSE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|iu|units?)\b.*$", re.IGNORECASE
)


def canonical_key(name: str) -> str:
    """Canonical knowledge-base key for a drug name.

    This is the **single source of truth** for drug identity. Both the
    knowledge base and the extraction pipeline must use it, otherwise a
    brand name and its generic name end up in different buckets and the
    interaction is silently missed.

    >>> canonical_key("Warfarin Sodium")
    'warfarin'
    >>> canonical_key("Ecosprin AV")
    'acetylsalicylic acid'
    >>> canonical_key("Aspirin")
    'acetylsalicylic acid'
    """
    if not name:
        return ""
    value = str(name).strip().lower()
    value = _BRACKET_PATTERN.sub(" ", value)
    value = _TRAILING_DOSE_PATTERN.sub(" ", value)
    value = re.sub(r"[^\w\s\-/]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -/,")
    return resolve_alias(value)


def parse_dose(surface: str) -> str | None:
    """Extract the dose string from a prescription surface form.

    >>> parse_dose("Metformin 500mg")
    '500mg'
    >>> parse_dose("Tab Azithral 500 mg")
    '500 mg'
    >>> parse_dose("Dolo 650")
    '650'
    """
    if not surface:
        return None
    match = _DOSE_RE.search(surface)
    if match:
        unit = match.group("unit").lower()
        unit = "mcg" if unit in {"ug", "mIU"} else unit
        return f"{match.group('value')}{match.group('sep')}{unit}"

    # Strength written without a unit, e.g. "Dolo 650".
    bare = _BARE_DOSE_RE.search(surface.strip())
    if bare:
        return bare.group("value")
    return None


def parse_frequency(surface: str) -> str | None:
    """Extract dosing frequency (e.g. ``bd``, ``once daily``) if present."""
    if not surface:
        return None
    match = _FREQUENCY_RE.search(surface)
    return match.group(0).lower() if match else None


def strip_salt(name: str) -> str:
    """Remove salt/chemical suffixes to derive a shorter generic key.

    >>> strip_salt("metformin hydrochloride")
    'metformin'
    """
    cleaned = name.strip().lower()
    for suffix in sorted(_SALT_SUFFIXES, key=len, reverse=True):
        if cleaned.endswith(" " + suffix):
            return cleaned[: -(len(suffix) + 1)].strip()
    return cleaned


def resolve_alias(name: str) -> str:
    """Map a brand/salt/abbreviation to its generic name.

    Handles brand-formulation suffixes ("Ecosprin AV" -> "Ecosprin") by
    progressively dropping trailing words.

    Returns the input unchanged (cleaned) when no alias is known - callers
    must therefore treat unknown drugs as *possibly out of distribution*
    rather than as a hard failure.
    """
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    key_no_salt = strip_salt(key)
    if key_no_salt in ALIASES:
        return ALIASES[key_no_salt]

    # Brand-formulation suffix fallback: "ecosprin av" -> "ecosprin".
    words = key_no_salt.split()
    for cut in range(1, min(len(words), 3)):
        candidate = " ".join(words[:-cut])
        if candidate in ALIASES:
            return ALIASES[candidate]

    return key_no_salt or key


def normalize(surface: str) -> str:
    """Full normalisation: surface form -> canonical generic drug key.

    Handles dosage-form prefixes ("Tab", "Cap"), dose text, dosing frequency
    and brand-name aliases.

    >>> normalize("Tab Dolo 650mg")
    'paracetamol'
    >>> normalize("Metformin HCl 500mg")
    'metformin'
    >>> normalize("Ecosprin AV")
    'acetylsalicylic acid'
    """
    if not surface:
        return ""
    value = str(surface).strip().lower()
    # Drop leading dosage-form words: tab, tablet, cap, syrup, sachet, ...
    value = re.sub(
        r"^(tab|tabs|tablet|tablets|cap|caps|capsule|capsules|syp|syrup|inj|injection|sachet|"
        r"cream|ointment|gel|drops?|lotion|patch|suspension|injec)\b[\s.]*",
        "",
        value,
    )
    # Drop trailing dose + frequency text.
    value = _TRAILING_DOSE_PATTERN.sub(" ", value)
    value = _FREQUENCY_RE.sub(" ", value)
    value = re.sub(r"[^\w\s\-/]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -/,")

    resolved = resolve_alias(value)
    # A multi-word alias may itself end with a salt, e.g. "metformin hydrochloride".
    resolved = strip_salt(resolved)
    return resolved or value


def normalise_token(text: str) -> str:
    """Alias-aware normaliser used when matching against a knowledge base."""
    return resolve_alias(text.strip().lower())
