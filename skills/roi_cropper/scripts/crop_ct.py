#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Crop CT around pancreas mask with a fixed voxel margin."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def get_bbox_from_mask(mask: np.ndarray, outside_value: float = -900, addon: list[int] | int = 0):
    if type(addon) is int:
        addon = [addon] * 3
    if (mask > outside_value).sum() == 0:
        minzidx, maxzidx = 0, mask.shape[0]
        minxidx, maxxidx = 0, mask.shape[1]
        minyidx, maxyidx = 0, mask.shape[2]
    else:
        mask_voxel_coords = np.where(mask > outside_value)
        minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
        minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
        minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

    s = mask.shape
    minzidx = max(0, minzidx)
    maxzidx = min(s[0], maxzidx)
    minxidx = max(0, minxidx)
    maxxidx = min(s[1], maxxidx)
    minyidx = max(0, minyidx)
    maxyidx = min(s[2], maxyidx)

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox_nifti(image: nib.Nifti1Image, bbox, dtype=None) -> nib.Nifti1Image:
    data = image.get_fdata()
    data_cropped = data[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
    affine = np.copy(image.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]
    data_type = image.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(data_cropped.astype(data_type), affine)

AIR_HU = -1000
DEFAULT_MARGIN = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop CT around pancreas mask")
    parser.add_argument("--ct", required=True, help="Input CT NIfTI path")
    parser.add_argument("--pancreas-mask", required=True, help="Pancreas mask NIfTI path")
    parser.add_argument("--output", required=True, help="Output cropped CT NIfTI path")
    parser.add_argument(
        "--mask-output",
        default=None,
        help="Optional output path for the cropped pancreas mask",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help="Margin in voxels added to the mask bounding box on each side",
    )
    return parser.parse_args()


def crop_ct(ct_path: str, mask_path: str, output_path: str, mask_output_path: str | None, margin: int) -> dict:
    ct_img = nib.load(ct_path)
    mask_img = nib.load(mask_path)

    ct_data = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

    if ct_data.shape != mask_data.shape:
        raise ValueError(f"CT and mask shapes must match: {ct_data.shape} vs {mask_data.shape}")

    bbox = get_bbox_from_mask(mask_data, outside_value=0, addon=[margin, margin, margin])
    cropped_ct = crop_to_bbox_nifti(ct_img, bbox)
    cropped_mask = crop_to_bbox_nifti(mask_img, bbox)

    ct_out = cropped_ct.get_fdata()
    mask_out = cropped_mask.get_fdata()
    ct_out[np.asarray(mask_out) <= 0] = AIR_HU

    nib.save(nib.Nifti1Image(ct_out.astype(ct_img.get_data_dtype()), cropped_ct.affine, cropped_ct.header), output_path)

    if mask_output_path:
        nib.save(cropped_mask, mask_output_path)

    return {
        "ct_output": output_path,
        "mask_output": mask_output_path,
        "bbox": bbox,
        "margin": margin,
        "original_shape": ct_img.shape,
        "cropped_shape": cropped_ct.shape,
    }


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mask_output:
        Path(args.mask_output).parent.mkdir(parents=True, exist_ok=True)

    result = crop_ct(args.ct, args.pancreas_mask, args.output, args.mask_output, args.margin)
    print(result)


if __name__ == "__main__":
    main()
