#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run TotalSegmentator with optional high-resolution vessel pass."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
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
}

VESSEL_ALIASES = {
    "superior_mesenteric_artery": {"sma", "superior_mesenteric_artery", "superior_mesenteric_artery_mask"},
    "superior_mesenteric_vein": {"smv", "superior_mesenteric_vein", "superior_mesenteric_vein_mask"},
    "celiac_trunk": {"ca", "celiac_trunk", "celiac_axis"},
    "common_hepatic_artery": {"cha", "common_hepatic_artery"},
    "splenic_artery": {"splenic_artery", "spla"},
    "gastroduodenal_artery": {"gda", "gastroduodenal_artery"},
    "portal_vein_and_splenic_vein": {"pv", "mpv", "portal_vein_and_splenic_vein"},
    "inferior_vena_cava": {"ivc", "inferior_vena_cava"},
    "aorta": {"ao", "aorta"},
}


def normalize_stem(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def canonical_vessel_name(path: Path) -> str | None:
    stem = normalize_stem(path.name)
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]


    for canonical, aliases in VESSEL_ALIASES.items():
        if stem == canonical or stem in aliases:
            return VESSEL_RENAMES[canonical]
    return None


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def publish_canonical_vessels(source_dir: Path, target_dir: Path, *, copy_files: bool = False) -> None:
    if not source_dir.exists():
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.nii.gz", "*.nii"):
        for source_file in sorted(source_dir.glob(pattern)):
            canonical_name = canonical_vessel_name(source_file)
            if canonical_name is None:
                continue

            target_file = target_dir / canonical_name
            if target_file.exists():
                target_file.unlink()
            if copy_files:
                shutil.copy2(str(source_file), str(target_file))
                action = "Copied"
            else:
                shutil.move(str(source_file), str(target_file))
                action = "Moved"
            print(f"{action} {source_file.name} -> {target_file.name}")


def move_high_res_vessels(source_dir: Path, target_dir: Path) -> None:
    publish_canonical_vessels(source_dir, target_dir, copy_files=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TotalSegmentator for organ and vessel segmentation")
    parser.add_argument("-i", "--input", required=True, help="Input CT NIfTI path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--device", default="gpu", help="TotalSegmentator device, e.g. gpu, cpu, gpu:0")
    parser.add_argument("--task", default="total", help="Base TotalSegmentator task")
    parser.add_argument("--fast", action="store_true", help="Use the fast lower-resolution model")
    parser.add_argument(
        "--high-res-vessels",
        action="store_true",
        help="Run a second vessel-focused pass and normalize high-resolution vessel masks into the output directory.",
    )
    parser.add_argument(
        "--vessel-library-dir",
        default=None,
        help="Optional directory for normalized vessel masks. Defaults to the main output directory.",
    )
    parser.add_argument(
        "--publish-canonical-vessels",
        action="store_true",
        help="Normalize any recognized vessel masks from the base output into the vessel library without requiring a second pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_command = [
        "conda",
        "run",
        "-n",
        "totalseg",
        "TotalSegmentator",
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "-ta",
        args.task,
        "-d",
        args.device,
    ]
    if args.fast:
        base_command.append("--fast")

    run_command(base_command)

    vessel_library_dir = Path(args.vessel_library_dir).expanduser().resolve() if args.vessel_library_dir else output_dir
    if args.publish_canonical_vessels:
        publish_canonical_vessels(output_dir, vessel_library_dir, copy_files=True)

    if args.high_res_vessels:
        vessel_tmp_dir = output_dir / "_high_res_vessels_tmp"
        if vessel_tmp_dir.exists():
            shutil.rmtree(vessel_tmp_dir)
        vessel_tmp_dir.mkdir(parents=True, exist_ok=True)

        vessel_command = [
            "conda",
            "run",
            "-n",
            "totalseg",
            "TotalSegmentator",
            "-i",
            str(input_path),
            "-o",
            str(vessel_tmp_dir),
            "-ta",
            "vessels",
            "-d",
            args.device,
        ]
        run_command(vessel_command)
        move_high_res_vessels(vessel_tmp_dir, vessel_library_dir)
        shutil.rmtree(vessel_tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
