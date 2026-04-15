#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a canonical pancreatic vessel library from one or more segmentation sources."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


VESSEL_RENAMES = {
    "aorta": "aorta.nii.gz",
    "inferior_vena_cava": "inferior_vena_cava.nii.gz",
    "portal_vein_and_splenic_vein": "portal_vein_and_splenic_vein.nii.gz",
    "superior_mesenteric_artery": "superior_mesenteric_artery.nii.gz",
    "superior_mesenteric_vein": "superior_mesenteric_vein.nii.gz",
    "celiac_trunk": "celiac_trunk.nii.gz",
    "common_hepatic_artery": "common_hepatic_artery.nii.gz",
    "splenic_artery": "splenic_artery.nii.gz",
    "gastroduodenal_artery": "gastroduodenal_artery.nii.gz",
    "splenic_vein": "splenic_vein.nii.gz",
}

VESSEL_ALIASES = {
    "superior_mesenteric_artery": {"sma", "superior_mesenteric_artery", "superior_mesenteric_artery_mask"},
    "superior_mesenteric_vein": {"smv", "superior_mesenteric_vein", "superior_mesenteric_vein_mask"},
    "celiac_trunk": {"ca", "celiac_trunk", "celiac_axis"},
    "common_hepatic_artery": {"cha", "common_hepatic_artery"},
    "splenic_artery": {"splenic_artery", "spla"},
    "gastroduodenal_artery": {"gda", "gastroduodenal_artery"},
    "portal_vein_and_splenic_vein": {"pv", "mpv", "portal_vein_and_splenic_vein"},
    "splenic_vein": {"sv", "splenic_vein"},
    "inferior_vena_cava": {"ivc", "inferior_vena_cava"},
    "aorta": {"ao", "aorta"},
}

PREFERRED_PANCREATIC_VESSELS = {
    "superior_mesenteric_artery",
    "superior_mesenteric_vein",
    "celiac_trunk",
    "common_hepatic_artery",
    "splenic_artery",
    "gastroduodenal_artery",
    "splenic_vein",
}

BASELINE_VESSELS = {
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
}


def normalize_stem(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def canonical_key_from_name(name: str) -> str | None:
    stem = normalize_stem(name)
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]

    for canonical, aliases in VESSEL_ALIASES.items():
        if stem == canonical or stem in aliases:
            return canonical
    return None


def canonical_filename(name: str) -> str | None:
    key = canonical_key_from_name(name)
    if key is None:
        return None
    return VESSEL_RENAMES[key]


def iter_candidate_files(directory: Path) -> list[Path]:
    candidates = sorted(directory.glob("*.nii.gz"))
    candidates.extend(sorted(directory.glob("*.nii")))
    return candidates


def should_overwrite(existing_source_name: str, incoming_source_name: str, canonical_key: str) -> bool:
    if canonical_key in PREFERRED_PANCREATIC_VESSELS:
        return incoming_source_name == "dedicated"
    if canonical_key in BASELINE_VESSELS:
        return existing_source_name != "totalseg" and incoming_source_name == "totalseg"
    return False


def merge_source(source_dir: Path, target_dir: Path, source_name: str, provenance: dict[str, str]) -> None:
    if not source_dir.exists():
        return

    for source_file in iter_candidate_files(source_dir):
        canonical_key = canonical_key_from_name(source_file.name)
        if canonical_key is None:
            continue

        target_file = target_dir / VESSEL_RENAMES[canonical_key]
        if source_file.resolve() == target_file.resolve():
            provenance[target_file.name] = source_name
            print(f"Keeping {target_file.name} in place [{source_name}]")
            continue

        existing_source_name = provenance.get(target_file.name)
        if target_file.exists() and existing_source_name is not None:
            if not should_overwrite(existing_source_name, source_name, canonical_key):
                continue

        shutil.copy2(source_file, target_file)
        provenance[target_file.name] = source_name
        print(f"Published {source_file.name} -> {target_file.name} [{source_name}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge vessel masks into a canonical vessel library")
    parser.add_argument("--output-dir", required=True, help="Canonical vessel library output directory")
    parser.add_argument("--totalseg-dir", default=None, help="Optional TotalSegmentator directory")
    parser.add_argument("--dedicated-dir", default=None, help="Optional dedicated vessel model directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, str] = {}

    if args.totalseg_dir:
        merge_source(Path(args.totalseg_dir).expanduser().resolve(), output_dir, "totalseg", provenance)
    if args.dedicated_dir:
        merge_source(Path(args.dedicated_dir).expanduser().resolve(), output_dir, "dedicated", provenance)

    published = sorted(provenance)
    print("Canonical vessel library ready:")
    for name in published:
        print(f"- {name}: {provenance[name]}")
    if not published:
        print("- no canonical vessel masks found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
