#!/usr/bin/env python3
"""Shorten ref.bib per advisor feedback into ref_new.bib."""
from __future__ import annotations

import re
from pathlib import Path

# Prefer the original full backup when regenerating retained fields.
_ROOT = Path("/Users/qlt315/Documents/GitHub/OMNIS")
_BACKUP = _ROOT / "ref_orig_backup.bib"
SRC = _BACKUP if _BACKUP.exists() else _ROOT / "ref.bib"
DST = _ROOT / "ref_new.bib"

BOOKTITLE_MAP = [
    (r"(?i)IEEE INFOCOM\s*2026\s*[-–—].*", "IEEE INFOCOM"),
    (r"(?i)IEEE Conference on Computer Communications\s*\(INFOCOM\)", "IEEE INFOCOM"),
    (
        r"(?i)Twenty-Fifth International Symposium on Theory, Algorithmic "
        r"Foundations, and Protocol Design for Mobile Networks and Mobile Computing",
        "ACM MobiHoc",
    ),
    (
        r"(?i)ACM/IEEE\s+\d+(?:st|nd|rd|th)?\s+International Conference on "
        r"Cyber-Physical Systems\s*\(ICCPS\)",
        "ACM/IEEE ICCPS",
    ),
    (
        r"(?i)IEEE/\s*CVF\s+Conference on Computer Vision and Pattern "
        r"Recognition\s*\(CVPR\)",
        "IEEE/CVF CVPR",
    ),
    (
        r"(?i)IEEE Conference on Computer Vision and Pattern Recognition\s*\(CVPR\)",
        "IEEE/CVF CVPR",
    ),
    (
        r"(?i)IEEE/\s*CVF\s+Winter Conference on Applications of Computer "
        r"Vision\s*\(WACV\)",
        "IEEE/CVF WACV",
    ),
    (r"(?i)IEEE Military Communications Conference\s*\(MILCOM\)", "IEEE MILCOM"),
    (
        r"(?i)IEEE\s+\d+(?:st|nd|rd|th)?\s+International Symposium on a World of "
        r"Wireless, Mobile and Multimedia Networks\s*\(WoWMoM\)",
        "IEEE WoWMoM",
    ),
    (
        r"(?i)IEEE\s+\d+(?:st|nd|rd|th)?\s+International Conference on "
        r"Distributed Computing Systems\s*\(ICDCS\)",
        "IEEE ICDCS",
    ),
    (r"(?i)IEEE International Conference on Communications\s*\(ICC\)", "IEEE ICC"),
    (r"(?i)IEEE Global Communications Conference\s*\(GLOBECOM\)", "IEEE GLOBECOM"),
    (r"(?i)European Conference on Computer Vision\s*\(\{?ECCV\}?\)", "ECCV"),
    (
        r"(?i)\d+(?:st|nd|rd|th)?\s+International Conference on Learning "
        r"Representations\s*\(ICLR\)",
        "ICLR",
    ),
    (
        r"(?i)\d+(?:st|nd|rd|th)?\s+International Conference on Neural "
        r"Information Processing Systems",
        "NeurIPS",
    ),
    (r"(?i)\d+(?:st|nd|rd|th)?\s+Picture Coding Symposium", "PCS"),
]

# Standard journal abbreviations (IEEE / common ISO-4 style)
JOURNAL_MAP = {
    "IEEE Transactions on Mobile Computing": "IEEE Trans. Mobile Comput.",
    "IEEE Transactions on Wireless Communications": "IEEE Trans. Wireless Commun.",
    "IEEE/ACM Transactions on Networking": "IEEE/ACM Trans. Netw.",
    "IEEE Journal on Selected Areas in Communications": "IEEE J. Sel. Areas Commun.",
    "IEEE Internet of Things Journal": "IEEE Internet Things J.",
    "IEEE Transactions on Vehicular Technology": "IEEE Trans. Veh. Technol.",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence": "IEEE Trans. Pattern Anal. Mach. Intell.",
    "IEEE Transactions on Circuits and Systems for Video Technology": "IEEE Trans. Circuits Syst. Video Technol.",
    "IEEE Access": "IEEE Access",
    "IEEE Wireless Communications Letters": "IEEE Wireless Commun. Lett.",
    "IEEE Transactions on Communications": "IEEE Trans. Commun.",
    "IEEE Journal of Selected Topics in Signal Processing": "IEEE J. Sel. Topics Signal Process.",
    "IEEE Transactions on Parallel and Distributed Systems": "IEEE Trans. Parallel Distrib. Syst.",
    "IEEE Transactions on Cognitive Communications and Networking": "IEEE Trans. Cogn. Commun. Netw.",
    "Communications of the {ACM}": "Commun. {ACM}",
    "Communications of the ACM": "Commun. {ACM}",
    "ACM Comput. Surv.": "ACM Comput. Surv.",
    "Journal of Machine Learning Research": "J. Mach. Learn. Res.",
    "Operations Research Letters": "Oper. Res. Lett.",
}


ACRONYMS = [
    "SEM-O-RAN",
    "OffloaDNN",
    "BottleFit",
    "Mask R-CNN",
    "Fog-RAN",
    "C-RAN",
    "O-RAN",
    "vRANs",
    "vRAN",
    "RANs",
    "RAN",
    "MU-MIMO",
    "MIMO",
    "OFDMA",
    "eMBB",
    "URLLC",
    "DNNs",
    "DNN",
    "MEC",
    "IoT",
    "D2D",
    "V2X",
    "IRS",
    "LLM",
    "JPEG",
    "COCO",
    "CVXPY",
    "PQK",
    "ReLeQ",
    "MixQuant",
    "Sionna",
    "DRL",
    "SLA",
    "QoS",
    "NR",
    "5G",
    "6G",
    "AI",
    "UL/DL",
    "ZeroQ",
    "NaviSplit",
    "Condar",
    "HydraFusion",
    "VQ-DeepSC",
    "OMNIS",
]


def parse_entries(text: str) -> list[tuple[str, str, dict[str, str]]]:
    """Return list of (type, key, fields_dict). Skips @String."""
    entries: list[tuple[str, str, dict[str, str]]] = []
    i = 0
    n = len(text)
    while i < n:
        m = re.search(r"@(\w+)\s*\{", text[i:])
        if not m:
            break
        etype = m.group(1)
        start = i + m.start()
        brace0 = i + m.end() - 1  # position of '{'
        if etype.lower() == "string":
            # skip @String{...}
            depth = 0
            j = brace0
            while j < n:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        i = j + 1
                        break
                j += 1
            else:
                break
            continue

        # read key
        j = brace0 + 1
        while j < n and text[j].isspace():
            j += 1
        key_start = j
        while j < n and text[j] not in ",}":
            j += 1
        key = text[key_start:j].strip()
        # find matching closing brace for entry
        depth = 0
        k = brace0
        while k < n:
            ch = text[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body = text[brace0 + 1 : k]
                    # strip key and first comma
                    body = re.sub(rf"^\s*{re.escape(key)}\s*,", "", body, count=1)
                    fields = parse_fields(body)
                    entries.append((etype, key, fields))
                    i = k + 1
                    break
            k += 1
        else:
            break
    return entries


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    # iterate name = value
    pos = 0
    body = body.strip()
    while pos < len(body):
        while pos < len(body) and body[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(body):
            break
        m = re.match(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*", body[pos:])
        if not m:
            break
        name = m.group(1)
        pos += m.end()
        if pos >= len(body):
            break
        if body[pos] == "{":
            depth = 0
            j = pos
            while j < len(body):
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                    if depth == 0:
                        fields[name.lower()] = body[pos + 1 : j]
                        pos = j + 1
                        break
                j += 1
            else:
                break
        elif body[pos] == '"':
            j = pos + 1
            while j < len(body) and body[j] != '"':
                j += 1
            fields[name.lower()] = body[pos + 1 : j]
            pos = j + 1
        else:
            # bare value until comma
            j = pos
            while j < len(body) and body[j] != ",":
                j += 1
            fields[name.lower()] = body[pos:j].strip()
            pos = j
    return fields


def brace_acronyms(title: str) -> str:
    parts = re.split(r"(\{[^}]*\})", title)
    acrs = sorted(set(ACRONYMS), key=len, reverse=True)
    out = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            out.append(part)
            continue
        for acr in acrs:
            pat = r"(?<!\{)\b" + re.escape(acr) + r"\b(?!\})"
            part = re.sub(pat, "{" + acr + "}", part)
        out.append(part)
    return "".join(out)


def truncate_authors(author_field: str) -> str:
    """IEEE: list all authors up to 6; if more than 6, first author + others."""
    raw = " ".join(author_field.split())
    parts = re.split(r"\s+and\s+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if parts and parts[-1].lower() == "others":
        names = parts[:-1]
    else:
        names = parts
    # Advisor asked "6 or more"; IEEE guide is "more than six".
    # Use >=6 -> first + others for compactness (IEEEtran prints "A. et al.").
    if len(names) >= 6:
        return names[0] + " and others"
    return " and ".join(names)


def shorten_journal(journal: str) -> str:
    s = " ".join(journal.split())
    if s in JOURNAL_MAP:
        return JOURNAL_MAP[s]
    # Fallback: try without protective braces in ACM
    s2 = s.replace("{ACM}", "ACM")
    if s2 in JOURNAL_MAP:
        return JOURNAL_MAP[s2]
    return s


def shorten_booktitle(bt: str) -> str:
    s = " ".join(bt.split())
    for pat, repl in BOOKTITLE_MAP:
        if re.search(pat, s):
            return repl
    m2 = re.search(r"(.+?)\s*\(([A-Za-z0-9/+-]{2,20})\)\s*$", s)
    if m2 and len(s) > 40:
        conf = m2.group(2)
        if re.search(r"IEEE", s, re.I):
            return f"IEEE {conf}"
        if re.search(r"ACM", s, re.I):
            return f"ACM {conf}"
        return conf
    if re.search(r"(?i)mobicom", s) and "ACM" in s.upper():
        return "ACM MobiCom"
    return s


PREFERRED_ORDER = [
    "author",
    "title",
    "journal",
    "booktitle",
    "howpublished",
    "institution",
    "type",
    "volume",
    "number",
    "pages",
    "month",
    "year",
    "publisher",
    "doi",
    "note",
    "series",
    "language",
    "day",
]


def format_entry(etype: str, key: str, fields: dict[str, str]) -> str:
    # Keep pages (IEEE Reference Guide includes pp.); also doi/publisher/month.
    # Drop only low-value clutter.
    for drop in ("address", "day", "language", "series", "note"):
        fields.pop(drop, None)

    if "author" in fields:
        fields["author"] = truncate_authors(fields["author"])
    if "title" in fields:
        t = fields["title"]
        if key == "Jacob2018":
            t = (
                "Quantization and Training of Neural Networks for Efficient "
                "Integer-Arithmetic-Only Inference"
            )
        fields["title"] = brace_acronyms(t)
    if "journal" in fields:
        fields["journal"] = shorten_journal(fields["journal"])
    if "booktitle" in fields:
        fields["booktitle"] = shorten_booktitle(fields["booktitle"])
        if fields["booktitle"] in {"ACM MobiHoc", "NeurIPS"}:
            fields.pop("series", None)

    lines = [f"@{etype}{{{key},"]
    seen = set()
    for name in PREFERRED_ORDER:
        if name in fields:
            lines.append(f"  {name}={{{fields[name]}}},")
            seen.add(name)
    for name, val in fields.items():
        if name not in seen:
            lines.append(f"  {name}={{{val}}},")
    # remove trailing comma on last field for cleanliness (optional; BibTeX allows)
    if lines[-1].endswith(","):
        lines[-1] = lines[-1][:-1]
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    # Keep a short header comment
    header = (
        "% Shortened bibliography (advisor feedback):\n"
        "% - no page numbers\n"
        "% - \"and others\" when >= 6 authors (first author only; IEEEtran -> et al.)\n"
        "% - keep pages (IEEE) + doi / publisher / month\n"
        "% - conference acronyms (IEEE INFOCOM, ACM MobiHoc, ...)\n"
        "% - standard journal abbreviations (IEEE Trans. Mobile Comput., ...)\n"
        "% - protect acronyms in titles with {5G}, {RAN}, {DNN}, {MEC}, ...\n"
        "% Generated from ref.bib\n\n"
    )
    entries = parse_entries(text)
    # Deduplicate by key (keep last)
    by_key: dict[str, tuple[str, str, dict[str, str]]] = {}
    for etype, key, fields in entries:
        by_key[key] = (etype, key, fields)

    blocks = [format_entry(*by_key[k]) for k in by_key]
    DST.write_text(header + "\n\n".join(blocks) + "\n", encoding="utf-8")

    out = DST.read_text(encoding="utf-8")
    print(f"Wrote {len(blocks)} entries -> {DST}")
    print("remaining pages fields:", len(re.findall(r"(?im)^\s*pages\s*=", out)))
    print("and others count:", len(re.findall(r"\band others\b", out)))
    for key in [
        "Veloso2024",
        "Qin2026",
        "Zefan2024",
        "Salvatore2021",
        "Xin2021",
        "Jacob2018",
        "Hoydis2022Sionna",
        "Nguyen2024",
    ]:
        mm = re.search(rf"@\w+\{{{key},.*?^\}}", out, re.S | re.M)
        if mm:
            print("---", key)
            print(mm.group(0))


if __name__ == "__main__":
    main()
