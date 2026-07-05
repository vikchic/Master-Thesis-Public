from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process
from parameters import DATA_DIR

# This code was created by Gemini (generative AI tool) based on provided data

# paths
RAW_UNIQUE_CLUBS_FILE = Path(f"{DATA_DIR}/processed/club_standardization/clubs_list.xlsx")
OFFICIAL_CLUBS_FILE = Path(f"{DATA_DIR}/processed/club_standardization/official_clubs_2024.xlsx")
OUTPUT_DIR = Path(f"{DATA_DIR}/processed/club_standardization")

MAPPING_CSV = Path(f"{DATA_DIR}/processed/club_standardization/club_mapping.csv")
MAPPING_PARQUET = Path(f"{DATA_DIR}/processed/club_standardization/club_mapping.parquet")
REVIEW_CSV = Path(f"{DATA_DIR}/processed/club_standardization/club_mapping_review.csv")
OFFICIAL_CSV = Path(f"{DATA_DIR}/processed/club_standardization/official_clubs_2024_flat.csv")

# settings
RAW_CLUB_COLUMN = "home_club"
RAW_DISTRICT_COLUMN = "home_district"
RAW_ROWS_COLUMN = "rows"

FUZZY_SCORE_CUTOFF = 88

KLUBBLOS_VALUES = {
    "klubblös",
    "klubblos",
    "klubblös.",
    "klubblos.",
}

JUNK_VALUES = {
    "",
    "-",
    "--",
    "---",
    "- -",
    ".",
    "..",
    "...",
    ",",
    "0",
    "okänd",
    "okant",
    "unknown",
    "ingen klubb",
    "saknas",
    "none",
    "null",
    "vacant",
    "SWE"
}

COMBO_SEPARATORS = ["/", "+", " och ", " & ", ","]

TEAM_SUFFIX_RE = re.compile(r"\s+\d{1,2}[a-z]?$", flags=re.IGNORECASE)
TRAILING_TEAM_WORD_RE = re.compile(
    r"\s+(lag|team|stafettlag|relaylag)\s*\d{0,2}[a-z]?$",
    flags=re.IGNORECASE,
)

# concrete aliases that couldn't be matched at first
MANUAL_ALIASES = {
    "västerås sok": "Västerås Skid O OK",
    "sodertalje nykvarn of": "Södertälje-Nykvarn Orienteringsförening",
    "södertälje-nykvarn of": "Södertälje-Nykvarn Orienteringsförening",
    "jarla orientering": "Järla IF Orienteringsklubb",
    "järla orientering": "Järla IF Orienteringsklubb",
    "ifk goteborg orientering": "Idrottsföreningen Kamraterna Göteborg Orientering",
    "ifk göteborg orientering": "Idrottsföreningen Kamraterna Göteborg Orientering",
    "ifk göteborg": "Idrottsföreningen Kamraterna Göteborg Orientering",
    "ifk goteborg": "Idrottsföreningen Kamraterna Göteborg Orientering",
    "domnarvets goif": "Domnarvets Gymnastik o Idrottsförening",
    "ik hakarpspojkarna": "IKHP Huskvarna Idrottsklubb",
    "sjovalla fk": "Sjövalla Frisksportklubb",
    "sjövalla fk": "Sjövalla Frisksportklubb",
    "lidkopings vsk": "Lidköpings Vinter-Sportklubb",
    "lidköpings vsk": "Lidköpings Vinter-Sportklubb",
    "hellas orientering": "Hellas Orienteringsklubb",
    "sok aneby": "Skid- och Orienteringsklubben Aneby",
    "bredaryds sok": "Bredaryds Skid O OK",
    "sok viljan": "Skid o Orienteringsklubben Viljan",
    "ik ymer": "Idrottsklubben Ymer",
    "sol tranas": "Skid- och orienteringslöparna Tranås",
    "sol tranås": "Skid- och orienteringslöparna Tranås",
    "hagaby goif orebro": "Hagaby Gymnastik och Idrottsförening Örebro",
    "hagaby goif örebro": "Hagaby Gymnastik och Idrottsförening Örebro",
    "bjorkfors goif": "Björkfors Gymnastik O IF",
    "björkfors goif": "Björkfors Gymnastik O IF",
    "bergnasets aik": "Bergnäsets Allmänna IK",
    "bergnäsets aik": "Bergnäsets Allmänna IK",
    "valbo aif": "Valbo Allmänna Idrottsförening",
    "motala aif ol": "Motala AIF Orienteringslag",
    "fk herkules": "Frisksportklubben Herkules",
    "fellingsbro goif": "Fellingsbro Gymnastik O IF",
    "eslovs fk": "Eslövs Friluftsklubb",
    "eslövs fk": "Eslövs Friluftsklubb",
    "orkelljunga fk": "Örkelljunga Friluftsklubb",
    "örkelljunga fk": "Örkelljunga Friluftsklubb",
    "mariestads fk": "Mariestads Friluftsklubb",
    "if thor": "Idrottsföreningen Thor",
}

# helper functions
def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_accents_for_key(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def ascii_key(s: str | None) -> str | None:
    if s is None:
        return None
    out = strip_accents_for_key(s.lower())
    out = out.replace("å", "a").replace("ä", "a").replace("ö", "o")
    out = out.replace("æ", "a").replace("ø", "o")
    out = re.sub(r"[^a-z0-9 -]+", " ", out)
    out = re.sub(r"[-]+", " ", out)
    return normalize_whitespace(out) or None


def clean_club_name(name: object) -> str | None:
    if pd.isna(name):
        return None

    s = str(name).strip().lower()
    if not s:
        return None

    s = s.replace("–", "-").replace("—", "-")
    s = normalize_whitespace(s)
    s = s.strip(" ;:|")

    if s in KLUBBLOS_VALUES:
        return "klubblös"

    if s in JUNK_VALUES:
        return None

    if re.fullmatch(r"[-.\s/]+", s):
        return None

    return s or None


def split_combo_take_first(s: str) -> str:
    out = s
    for sep in COMBO_SEPARATORS:
        if sep in out:
            out = out.split(sep)[0].strip()
            break
    return out


def normalize_raw_to_candidate(name: object) -> str | None:
    s = clean_club_name(name)
    if s is None:
        return None

    if s == "klubblös":
        return s

    s = split_combo_take_first(s)
    s = TRAILING_TEAM_WORD_RE.sub("", s)
    s = TEAM_SUFFIX_RE.sub("", s)
    s = normalize_whitespace(s).strip(" ;:|,/+-")
    s = normalize_whitespace(s)

    if s in JUNK_VALUES or s == "":
        return None

    if s in KLUBBLOS_VALUES:
        return "klubblös"

    return s or None


def match_key(clean_name: str | None) -> str | None:
    """The 'Under the Hood' Matching Engine"""
    if clean_name is None:
        return None

    s = clean_name.lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = normalize_whitespace(s)

    replacements = [
        (r"\bidrottsföreningen kamraterna\b", "ifk"),
        (r"\bskid\s*[-]?\s*(o|och|&)\s*orienteringslöpar(e|na)\b", "sol"),
        (r"\bskid\s*[-]?\s*(o|och|&)\s*orienteringsklubb(en)?\b", "sok"),
        (r"\bgymnastik\s*[-]?\s*(o|och|&)\s*idrottsförening(en)?\b", "goif"),
        (r"\ballm(ä|a)nna\s*idrottsförening(en)?\b", "aif"),
        (r"\ballm(ä|a)nna\s*idrottsklubb(en)?\b", "aik"),
        (r"\ballm(ä|a)nna\s*if\b", "aif"),
        (r"\ballm(ä|a)nna\s*ik\b", "aik"),
        (r"\bskid\s*[-]?\s*(o|och|&)\s*ok\b", "sok"),
        (r"\bskid\s*[-]?\s*(o|och|&)\s*ol\b", "sol"),
        (r"\bgymnastik\s*[-]?\s*(o|och|&)\s*if\b", "goif"),
        (r"\bvinter[- ]sportklubb(en)?\b", "vsk"),
        (r"\borienteringssällskap(et)?\b", "os"),
        (r"\borienteringsklubb(en)?\b", "ok"),
        (r"\borienteringsförening(en)?\b", "of"),
        (r"\borienteringsforening(en)?\b", "of"),
        (r"\borienteringslöpar(e|na)\b", "ol"),
        (r"\borienteringslopar(e|na)\b", "ol"),
        (r"\bidrottsförening(en)?\b", "if"),
        (r"\bidrottsforening(en)?\b", "if"),
        (r"\bfrisksportklubb(en)?\b", "fk"),
        (r"\bidrottsklubb(en)?\b", "ik"),
        (r"\bfrilufts?\s*klubb(en)?\b", "fk"),
        (r"\bol[- ]klubb(en)?\b", "ol"),
        (r"\bkfuk-kfum\b", "kfum"),
        (r"\bsportklubb(en)?\b", "sk"),
        (r"\bskidklubb(en)?\b", "sk"),
        (r"\bkamraterna\b", "k"),
        (r"\bo\s*ok\b", "ok"),
        (r"\bsok\b", "sok"),
        (r"\bok\b", "ok"),
        (r"\bof\b", "of"),
        (r"\bol\b", "ol"),
        (r"\bifk\b", "ifk"),
        (r"\bik\b", "ik"),
        (r"\bsk\b", "sk"),
        (r"\bfk\b", "fk"),
        (r"\baif\b", "aif"),
        (r"\baik\b", "aik"),
        (r"\bgoif\b", "goif"),
        (r"\bvsk\b", "vsk"),
        (r"\bkfum\b", "kfum"),
    ]

    for pattern, repl in replacements:
        s = re.sub(pattern, repl, s)

    s = ascii_key(s)
    return s or None


def shorten_display_name(name: object) -> str | None:
    """The 'Visual Formatting' Engine"""
    if pd.isna(name):
        return None
        
    s = str(name)
    
    replacements = {
        r"\bidrottsföreningen\s*kamraterna\b": "IFK",
        r"\bskid\s*[-]?\s*(o|och|&)\s*orienteringslöpar(e|na)\b": "SOL",
        r"\bskid\s*[-]?\s*(o|och|&)\s*orienteringsklubb(en)?\b": "SOK",
        r"\bgymnastik\s*[-]?\s*(o|och|&)\s*idrottsförening(en)?\b": "GoIF",
        r"\ballmänna\s*idrottsförening(en)?\b": "AIF",
        r"\ballmänna\s*idrottsklubb(en)?\b": "AIK",
        r"\ballmänna\s*if\b": "AIF",
        r"\ballmänna\s*ik\b": "AIK",
        r"\bskid\s*[-]?\s*(o|och|&)\s*ok\b": "SOK",
        r"\bskid\s*[-]?\s*(o|och|&)\s*ol\b": "SOL",
        r"\bgymnastik\s*[-]?\s*(o|och|&)\s*if\b": "GoIF",
        r"\bvinter[- ]sportklubb(en)?\b": "VSK",
        r"\borienteringssällskap(et)?\b": "OS",
        r"\borienteringsklubb(en)?\b": "OK",
        r"\borienteringsförening(en)?\b": "OF",
        r"\borienteringslöpar(e|na)\b": "OL",
        r"\borienteringslopar(e|na)\b": "OL",
        r"\bidrottsförening(en)?\b": "IF",
        r"\bidrottsforening(en)?\b": "IF",
        r"\bfrisksportklubb(en)?\b": "FK",
        r"\bidrottsklubb(en)?\b": "IK",
        r"\bfrilufts?\s*klubb(en)?\b": "FK",
        r"\bskidklubb(en)?\b": "SK",
        r"\bsportklubb(en)?\b": "SK",
    }
    
    for pattern, new in replacements.items():
        s = re.sub(pattern, new, s, flags=re.IGNORECASE)
        
    return normalize_whitespace(s)


def official_key_variants(clean_name: str) -> set[str]:
    variants = set()

    base = match_key(clean_name)
    if base:
        variants.add(base)

    short_name = shorten_display_name(clean_name)
    if short_name:
        short_key = match_key(short_name)
        if short_key:
            variants.add(short_key)

        short_ascii = ascii_key(short_name)
        if short_ascii:
            variants.add(short_ascii)

    return {v for v in variants if v}


def apply_manual_alias(normalized_club: str | None) -> str | None:
    if normalized_club is None:
        return None
    return MANUAL_ALIASES.get(ascii_key(normalized_club), normalized_club)


def classify_status(normalized_club: str | None, canonical_club: object) -> str:
    if normalized_club is None:
        return "junk"
    if normalized_club == "klubblös":
        return "klubblos"
    if pd.notna(canonical_club):
        return "matched"
    return "manual_review"


# read the official club file
def load_official_clubs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None)

    district_names = df.iloc[0]
    district_ids = df.iloc[1]

    records: list[dict] = []

    for col in df.columns:
        district_name = district_names[col]
        district_id = district_ids[col]

        if pd.isna(district_name) or pd.isna(district_id):
            continue

        try:
            district_id_int = int(district_id)
        except Exception:
            continue

        for value in df.iloc[3:, col]:
            if pd.isna(value):
                continue

            canonical_club = str(value).strip()
            if canonical_club == "0":
                continue

            clean_name = clean_club_name(canonical_club)
            if clean_name is None:
                continue

            for key in official_key_variants(clean_name):
                records.append(
                    {
                        "canonical_club": canonical_club,
                        "district_name": str(district_name).strip(),
                        "district_id": district_id_int,
                        "clean_name": clean_name,
                        "key": key,
                    }
                )

    # ddd klubblös as the one allowed extra club
    for key in official_key_variants("klubblös"):
        records.append(
            {
                "canonical_club": "klubblös",
                "district_name": "klubblös",
                "district_id": 0,
                "clean_name": "klubblös",
                "key": key,
            }
        )

    official = pd.DataFrame(records)
    official = official.drop_duplicates(subset=["key"], keep="first").reset_index(drop=True)
    return official

# read cleaned unique club list
def load_clean_unique_clubs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)

    expected = {RAW_CLUB_COLUMN, RAW_DISTRICT_COLUMN, RAW_ROWS_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} must contain columns {sorted(expected)}. Missing: {sorted(missing)}"
        )

    out = df.copy()

    out["normalized_club"] = out[RAW_CLUB_COLUMN].map(normalize_raw_to_candidate)
    out["normalized_club"] = out["normalized_club"].map(apply_manual_alias)
    out["key"] = out["normalized_club"].map(match_key)

    return out

# matching
def fuzzy_match_candidates(mapping: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    official_choices = official["key"].dropna().tolist()
    official_lookup = official.set_index("key")[["canonical_club", "district_id", "district_name"]]

    unresolved = (
        mapping["canonical_club"].isna()
        & mapping["normalized_club"].notna()
        & (mapping["normalized_club"] != "klubblös")
    )

    mapping["fuzzy_key"] = None
    mapping["match_score"] = None

    for idx, key in mapping.loc[unresolved, "key"].items():
        if not key:
            continue

        match = process.extractOne(
            key,
            official_choices,
            scorer=fuzz.WRatio,
            score_cutoff=FUZZY_SCORE_CUTOFF,
        )

        if match is not None:
            matched_key, score, _ = match
            mapping.loc[idx, "fuzzy_key"] = matched_key
            mapping.loc[idx, "match_score"] = float(score)

    ok = mapping["fuzzy_key"].notna()
    mapping.loc[ok, "canonical_club"] = mapping.loc[ok, "fuzzy_key"].map(official_lookup["canonical_club"])
    mapping.loc[ok, "canonical_district"] = mapping.loc[ok, "fuzzy_key"].map(official_lookup["district_id"])
    mapping.loc[ok, "canonical_district_name"] = mapping.loc[ok, "fuzzy_key"].map(official_lookup["district_name"])
    mapping.loc[ok, "match_method"] = "fuzzy"

    return mapping


def build_mapping(raw_unique: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:

    # drop empty and duplicate keys
    official_clean = official.dropna(subset=["key"]).drop_duplicates(subset=["key"])
    official_lookup = official_clean.set_index("key")[["canonical_club", "district_id", "district_name"]]

    mapping = raw_unique.copy()

    mapping["canonical_club"] = mapping["key"].map(official_lookup["canonical_club"])
    mapping["canonical_district"] = mapping["key"].map(official_lookup["district_id"])
    mapping["canonical_district_name"] = mapping["key"].map(official_lookup["district_name"])
    mapping["match_method"] = None

    exact = mapping["canonical_club"].notna()
    mapping.loc[exact, "match_method"] = "exact"

    mapping = fuzzy_match_candidates(mapping, official_clean)

    # shorten the canonical names for visual simplicity
    mapping["canonical_club"] = mapping["canonical_club"].apply(shorten_display_name)

    mapping["status"] = mapping.apply(
        lambda row: classify_status(row["normalized_club"], row["canonical_club"]),
        axis=1,
    )

    # diagnostics only: compare with original district in clean list if it exists
    mapping["district_same_as_input"] = (
        mapping["canonical_district"].fillna(-9999).astype(int)
        == mapping[RAW_DISTRICT_COLUMN].fillna(-9999).astype(int)
    )

    return mapping

# output
def save_outputs(mapping: pd.DataFrame, official: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping.to_csv(MAPPING_CSV, index=False)
    official.to_csv(OFFICIAL_CSV, index=False)

    try:
        mapping.to_parquet(MAPPING_PARQUET, index=False)
        print(f"Wrote parquet: {MAPPING_PARQUET}")
    except Exception as e:
        print(f"Could not write parquet ({e}). CSV was still written.")

    review = mapping.loc[
        mapping["status"].isin(["manual_review", "junk"]),
        [
            RAW_CLUB_COLUMN,
            RAW_DISTRICT_COLUMN,
            RAW_ROWS_COLUMN,
            "normalized_club",
            "status",
            "canonical_club",
            "canonical_district",
            "match_method",
            "match_score",
        ],
    ].sort_values(["status", RAW_ROWS_COLUMN], ascending=[True, False])

    review.to_csv(REVIEW_CSV, index=False)


# main
def main() -> None:
    print("Loading official club reference...")
    official = load_official_clubs(OFFICIAL_CLUBS_FILE)

    supplement_path = OUTPUT_DIR / "club_supplement_template.csv"
    if supplement_path.exists():
        extra = pd.read_csv(supplement_path)
        extra = extra.rename(columns={
            "canonical_club": "canonical_club",
            "district_id": "district_id"
        })
        # build keys so they behave like official clubs
        extra["clean_name"] = extra["canonical_club"].map(clean_club_name)
        extra["key"] = extra["clean_name"].map(match_key)
        extra["district_name"] = "supplement"
        official = pd.concat([official, extra], ignore_index=True)

    print("Loading cleaned unique club list...")
    raw_unique = load_clean_unique_clubs(RAW_UNIQUE_CLUBS_FILE)

    print("Building club mapping...")
    mapping = build_mapping(raw_unique, official)

    print("Saving outputs...")
    save_outputs(mapping, official)

    print("\n=== Summary ===")
    print(mapping["status"].value_counts(dropna=False).to_string())

    total_rows = int(mapping[RAW_ROWS_COLUMN].sum())
    kept_rows = int(mapping.loc[mapping["status"].isin(["matched", "klubblos"]), RAW_ROWS_COLUMN].sum())
    coverage = (kept_rows / total_rows * 100) if total_rows else 0.0

    print(f"\nOfficial clubs in flattened reference: {len(official)}")
    print(f"Unique raw club strings:               {len(mapping)}")
    print(f"Total rows represented:               {total_rows:,}")
    print(f"Rows mapped/kept:                     {kept_rows:,} ({coverage:.2f}%)")

    print("\nTop manual review values:")
    manual = mapping.loc[
        mapping["status"] == "manual_review",
        [RAW_CLUB_COLUMN, RAW_ROWS_COLUMN, "normalized_club"],
    ]
    print(manual.sort_values(RAW_ROWS_COLUMN, ascending=False).head(30).to_string(index=False))


if __name__ == "__main__":
    main()