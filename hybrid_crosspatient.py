#!/usr/bin/env python3
"""
Cross-patient hybrid dose pipeline (Periphocal 3D + TPS RTDOSE).

Designed for cohort studies where one source RT plan is applied to many
whole-body voxel models. The TPS dose distribution (RTDOSE) is anatomically
ported from the source planning CT to a target WB CT by translating it so
that the source isocenter lands at a user-defined target isocenter on the
WB CT. Out-of-field dose is computed with the Periphocal 3D analytical model
on the WB anatomy. Per-organ statistics are produced from either NIfTI
segmentations (typical WB cohort use case) or an RTSTRUCT DICOM (useful for
verification against the original single-patient hybrid pipeline).

All RT plan parameters are entered manually via CLI, matching the workflow
of the original P3D MATLAB script. Source isocenter (from the planning RT
plan) and target isocenter (chosen on the WB CT) must both be provided.

P3D model: Sanchez-Nieto et al., Frontiers in Oncology, 12:872752 (2022).

Usage example
-------------
python hybrid_crosspatient.py \\
    --wb-ct-dir /path/to/wb_voxelmodel_dicom \\
    --rtdose /path/to/source_RTDOSE.dcm \\
    --source-iso-mm 118.11 -244.98 10.0 \\
    --target-iso-mm 118.11 -244.98 10.0 \\
    --rx-gy 30.0 --n-fractions 15 --total-mu 4005.0 \\
    --field-area-cm2 266.7 --energy-mv 6 \\
    --rtstruct /path/to/source_RTSTRUCT.dcm \\
    --outdir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_fill_holes, affine_transform
from matplotlib.path import Path as MplPath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# ============================================================
# Periphocal 3D model -- Sanchez-Nieto et al. (2022) Eq. 2
# ============================================================
A1_MGY_CM2_PER_MU = 37.890
A2_MGY_CM_PER_MU = 0.679
A3_PER_CM = 0.007
REF_FIELD_AREA_CM2 = 149.2
REF_EU_MGY_PER_MU = 7.2
REF_LEAKAGE_MGY_PER_MU = 0.001  # Lr from the paper; used as default Lu

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def estimate_field_area_jaws(rtplan_ds) -> float:
    """JAWS method: mean of per-beam jaw-rectangle areas (cm^2).

    First-order proxy for FU. Works for any plan but overestimates the
    effective area for IMRT/VMAT. Used as fallback when RTDOSE-based
    isodose estimation isn't possible.
    """
    areas = []
    for beam in rtplan_ds.BeamSequence:
        cp0 = beam.ControlPointSequence[0]
        if not hasattr(cp0, "BeamLimitingDevicePositionSequence"):
            continue
        x_w = y_w = None
        for bld in cp0.BeamLimitingDevicePositionSequence:
            jaws = [float(x) for x in bld.LeafJawPositions]
            rt = str(bld.RTBeamLimitingDeviceType).upper()
            if rt in ("X", "ASYMX"):
                x_w = abs(jaws[1] - jaws[0]) / 10.0
            elif rt in ("Y", "ASYMY"):
                y_w = abs(jaws[1] - jaws[0]) / 10.0
        if x_w and y_w:
            areas.append(x_w * y_w)
    return float(np.mean(areas)) if areas else REF_FIELD_AREA_CM2


def estimate_field_area_isodose(
    dose_gy: np.ndarray,
    dose_x: np.ndarray,
    dose_y: np.ndarray,
    dose_z: np.ndarray,
    iso: np.ndarray,
    ref_dose_gy: float,
    isodose_level: float = 0.5,
):
    """ISODOSE method: paper-correct FU from the 50% isodose at isocenter.

    Sanchez-Nieto et al. (2022) define FU as the mean of the areas inside the
    50% isodoses in the coronal and sagittal planes through the isocenter.
    This matches the workflow Jef described (open RTDOSE in 3D Slicer, plot
    50% isodose, average coronal + sagittal). For VMAT/IMRT this can differ
    meaningfully from the jaw-based proxy.

    Assumes RTDOSE DoseSummationType == PLAN, so threshold = 0.5 * Rx.

    Returns
    -------
    field_area_cm2 : float
    details : dict with per-plane areas, threshold, isodose level
    """
    threshold = float(isodose_level) * float(ref_dose_gy)
    iy_iso = int(np.argmin(np.abs(dose_y - iso[1])))
    ix_iso = int(np.argmin(np.abs(dose_x - iso[0])))

    coronal = dose_gy[:, iy_iso, :]                 # (nz, nx)
    coronal_mask = coronal >= threshold
    if coronal_mask.any():
        coronal_mask = binary_fill_holes(coronal_mask)

    sagittal = dose_gy[:, :, ix_iso]                # (nz, ny)
    sagittal_mask = sagittal >= threshold
    if sagittal_mask.any():
        sagittal_mask = binary_fill_holes(sagittal_mask)

    dx_mm = abs(float(np.diff(dose_x).mean())) if len(dose_x) > 1 else 1.0
    dy_mm = abs(float(np.diff(dose_y).mean())) if len(dose_y) > 1 else 1.0
    dz_mm = abs(float(np.diff(dose_z).mean())) if len(dose_z) > 1 else 1.0

    coronal_area_cm2 = float(coronal_mask.sum()) * dx_mm * dz_mm / 100.0
    sagittal_area_cm2 = float(sagittal_mask.sum()) * dy_mm * dz_mm / 100.0
    field_area_cm2 = 0.5 * (coronal_area_cm2 + sagittal_area_cm2)
    return field_area_cm2, {
        "coronal_area_cm2": coronal_area_cm2,
        "sagittal_area_cm2": sagittal_area_cm2,
        "threshold_gy": threshold,
        "isodose_level": isodose_level,
    }


# ============================================================
# Default organ groups
# ============================================================
# TotalSegmentator outputs many fine-grained anatomical labels (5 heart subparts,
# 5 lung lobes, 24 vertebrae, etc.) that are typically NOT what is reported in
# OOF dose studies. These groups combine the fine labels into clinically meaningful
# whole-organ masks via logical OR. Customizable via --organ-groups JSON.
DEFAULT_ORGAN_GROUPS: Dict[str, List[str]] = {
    # Cardiothoracic
    "lungs": [
        "lung_lower_lobe_left", "lung_lower_lobe_right",
        "lung_middle_lobe_right",
        "lung_upper_lobe_left", "lung_upper_lobe_right",
    ],
    "lung_left":  ["lung_lower_lobe_left", "lung_upper_lobe_left"],
    "lung_right": ["lung_lower_lobe_right", "lung_middle_lobe_right", "lung_upper_lobe_right"],
    "heart": [
        "heart_atrium_left", "heart_atrium_right",
        "heart_myocardium",
        "heart_ventricle_left", "heart_ventricle_right",
    ],
    # Paired endocrine / urogenital
    "kidneys":  ["kidney_left", "kidney_right"],
    "adrenals": ["adrenal_gland_left", "adrenal_gland_right"],
    # Spine: TS does NOT segment the spinal cord; this groups the vertebral
    # bones as a usable proxy. Should NOT be reported as 'spinal cord'.
    "vertebral_column": [
        "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
        "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
        "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4",
        "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8",
        "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    ],
    "vertebrae_cervical":  [f"vertebrae_C{i}" for i in range(1, 8)],
    "vertebrae_thoracic":  [f"vertebrae_T{i}" for i in range(1, 13)],
    "vertebrae_lumbar":    [f"vertebrae_L{i}" for i in range(1, 6)],
    # Ribcage
    "ribs": [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)],
    # Bony skeleton (paired)
    "hips":     ["hip_left", "hip_right"],
    "scapulas": ["scapula_left", "scapula_right"],
}


@dataclass
class P3DParams:
    """Periphocal 3D model parameters."""
    epsilon: float          # EU / EU_ref
    field_factor: float     # FU / FU_ref
    leakage: float          # Lu in mGy/MU


def p3d_ppd_mgy_per_mu(
    dx_cm: np.ndarray,
    dy_cm: np.ndarray,
    dz_cm: np.ndarray,
    p: P3DParams,
) -> np.ndarray:
    """Periphocal 3D peripheral photon dose (mGy/MU).

    Coordinate convention: dx, dy, dz are displacements from the isocenter
    in cm, in DICOM patient (LPS) coordinates. This matches the P3D paper:
        x = anterior-posterior, y = left-right, z = caudal-cranial.
    """
    r = np.sqrt(dx_cm * dx_cm + dy_cm * dy_cm + dz_cm * dz_cm)
    safe_r = np.maximum(r, 1e-6)
    scatter = (
        p.epsilon * p.field_factor
        * (A1_MGY_CM2_PER_MU - A2_MGY_CM_PER_MU * np.abs(dz_cm))
        * np.exp(-A3_PER_CM * safe_r)
        / (safe_r * safe_r)
    )
    dose = np.where(
        r <= 40.0,
        scatter + (p.leakage - REF_LEAKAGE_MGY_PER_MU),
        p.leakage,
    )
    dose = np.maximum(dose, 0.0)
    dose[r < 1e-6] = np.nan
    return dose


# ============================================================
# CT DICOM loading
# ============================================================
def load_ct_volume(ct_dir: Path) -> Dict:
    """Load a CT DICOM series into a (slices, rows, cols) volume.

    Returns a dict with HU volume, axis coordinates (mm, LPS), pixel spacing,
    origin, and a few useful metadata fields.
    """
    files = []
    for f in sorted(ct_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", "").upper() == "CT":
                files.append(f)
        except Exception:
            continue
    if not files:
        raise FileNotFoundError(f"No CT slices found in {ct_dir}")

    slices = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        z = float(ds.ImagePositionPatient[2])
        slices.append((z, ds))
    slices.sort(key=lambda x: x[0])

    ds0 = slices[0][1]
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    ps = [float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])]
    origin = [float(x) for x in ds0.ImagePositionPatient]

    # SAFETY: this code assumes standard HFS orientation [1,0,0,0,1,0]. Warn if not.
    iop = [float(v) for v in getattr(ds0, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])]
    expected_iop = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    if any(abs(a - b) > 1e-3 for a, b in zip(iop, expected_iop)):
        print(f"  WARNING: non-standard ImageOrientationPatient {iop}; "
              f"results may be incorrect (code assumes HFS, IOP=[1,0,0,0,1,0])")

    # SAFETY: warn on non-uniform z spacing (affects affine and voxel volume)
    z_positions = np.array([z for z, _ in slices])
    if len(z_positions) >= 2:
        dz_values = np.diff(z_positions)
        if dz_values.std() > 0.05:  # > 0.05 mm variation
            print(f"  WARNING: non-uniform z-spacing (mean={dz_values.mean():.3f} mm, "
                  f"std={dz_values.std():.3f} mm); affine assumes uniform spacing")

    volume = np.zeros((len(slices), rows, cols), dtype=np.float32)
    z_positions = np.zeros(len(slices))
    for i, (z, ds) in enumerate(slices):
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume[i] = ds.pixel_array.astype(np.float32) * slope + intercept
        z_positions[i] = z

    x_coords = origin[0] + np.arange(cols) * ps[1]
    y_coords = origin[1] + np.arange(rows) * ps[0]

    return {
        "volume": volume,
        "x": x_coords,
        "y": y_coords,
        "z": z_positions,
        "pixel_spacing": ps,
        "origin": origin,
        "n_slices": len(slices),
        "patient_id": str(getattr(ds0, "PatientID", "Unknown")),
    }


def compute_field_area_jaw(rtplan_path: Path) -> Optional[float]:
    """Estimate field area as mean of (X_jaw_opening x Y_jaw_opening) over beams.

    First-order approximation. Reasonable for static rectangular fields,
    significantly overestimates the EFFECTIVE field area for VMAT/IMRT plans
    with strong MLC modulation. NOT what the P3D paper specifies.
    """
    rp = pydicom.dcmread(str(rtplan_path))
    areas = []
    for b in rp.BeamSequence:
        cp = b.ControlPointSequence[0]
        if not hasattr(cp, "BeamLimitingDevicePositionSequence"):
            continue
        x_open = y_open = None
        for bld in cp.BeamLimitingDevicePositionSequence:
            rt = str(bld.RTBeamLimitingDeviceType).upper()
            pos = [float(p) for p in bld.LeafJawPositions]
            opening = abs(pos[1] - pos[0])
            if rt in ("X", "ASYMX"): x_open = opening
            elif rt in ("Y", "ASYMY"): y_open = opening
        if x_open is not None and y_open is not None:
            areas.append((x_open / 10.0) * (y_open / 10.0))
    return float(np.mean(areas)) if areas else None


def extract_rtplan_parameters(rtplan_path: Path) -> Dict:
    """Auto-extract treatment parameters from RTPLAN DICOM.

    Returns dict with keys (any may be None if not present in RTPLAN):
        rx_gy : float
            Total prescribed dose to the primary target (Gy). Sum of
            TargetPrescriptionDose in DoseReferenceSequence (or 1st target).
        n_fractions : int
            NumberOfFractionsPlanned from FractionGroupSequence.
        total_mu : float
            Sum of BeamMeterset across all referenced beams in fraction group.
        energy_mv : float
            Nominal beam energy (MV) of the first photon beam.
        isocenter_mm : list of 3 floats or None
            IsocenterPosition from first ControlPoint of first beam.
        beam_count : int
        plan_label : str
    """
    rp = pydicom.dcmread(str(rtplan_path))
    out: Dict = {
        "rx_gy": None, "n_fractions": None, "total_mu": None,
        "energy_mv": None, "isocenter_mm": None,
        "beam_count": 0, "plan_label": str(getattr(rp, "RTPlanLabel", "")),
    }

    # Prescription dose: prefer DoseReferenceSequence with type=TARGET
    if hasattr(rp, "DoseReferenceSequence"):
        targets = []
        for dr in rp.DoseReferenceSequence:
            tdose = getattr(dr, "TargetPrescriptionDose", None)
            if tdose is not None:
                targets.append(float(tdose))
        if targets:
            out["rx_gy"] = float(max(targets))

    # Fractionation + MU
    if hasattr(rp, "FractionGroupSequence") and len(rp.FractionGroupSequence):
        fg = rp.FractionGroupSequence[0]
        nfx = getattr(fg, "NumberOfFractionsPlanned", None)
        if nfx is not None:
            out["n_fractions"] = int(nfx)
        if hasattr(fg, "ReferencedBeamSequence"):
            total_mu_per_fx = 0.0
            for rb in fg.ReferencedBeamSequence:
                mu = getattr(rb, "BeamMeterset", None)
                if mu is not None:
                    total_mu_per_fx += float(mu)
            if total_mu_per_fx > 0 and out["n_fractions"] is not None:
                # BeamMeterset is per-fraction; multiply by fraction count
                out["total_mu"] = total_mu_per_fx * out["n_fractions"]

    # Energy + isocenter from first beam
    if hasattr(rp, "BeamSequence") and len(rp.BeamSequence):
        out["beam_count"] = len(rp.BeamSequence)
        b0 = rp.BeamSequence[0]
        if hasattr(b0, "ControlPointSequence") and len(b0.ControlPointSequence):
            cp = b0.ControlPointSequence[0]
            ne = getattr(cp, "NominalBeamEnergy", None)
            if ne is not None:
                out["energy_mv"] = float(ne)
            ip = getattr(cp, "IsocenterPosition", None)
            if ip is not None:
                out["isocenter_mm"] = [float(v) for v in ip]
    return out


def compute_target_iso_default(
    src_ct: Dict,
    src_iso_lps_mm: np.ndarray,
    src_rtstruct_path: Path,
    wb_ct: Dict,
    wb_seg_dir: Path,
    seg_coord_system: str = "ras",
    anchor_organ_candidates: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Auto-compute target isocenter on WB-patient CT via anchor-organ centroid.

    Methodology: pick a reference organ that exists both in the source RTSTRUCT
    and in the WB-patient TS segmentations. Compute centroids in each frame.
    Place the target iso at:

        target_iso = wb_anchor_centroid + (src_iso - src_anchor_centroid)

    This preserves the anatomical iso-anchor offset across patients. Useful as
    a default for cohort OOF studies where the source plan represents a
    treatment scenario applied to multiple anatomies.

    Anchor selection: tries candidates in order; uses the first one present
    in BOTH the source RTSTRUCT and the WB-patient TS segmentations.

    Returns:
        target_iso_mm : np.ndarray of 3 floats (LPS mm)
        info : dict with anchor name, centroids, offset
    """
    if anchor_organ_candidates is None:
        # Order = preference. Larger, central, anatomically stable structures first.
        anchor_organ_candidates = [
            "brain", "heart_myocardium", "liver", "spleen",
            "stomach", "urinary_bladder", "kidney_left", "kidney_right",
        ]

    src_struct_masks = build_rtstruct_masks(
        src_rtstruct_path, src_ct["x"], src_ct["y"], src_ct["z"]
    )
    src_struct_lower = {k.lower(): k for k in src_struct_masks}

    # Map common anchor candidate names to RTSTRUCT name variations
    name_aliases: Dict[str, List[str]] = {
        "brain": ["brain", "brein", "hersenen", "wholebrain", "whole brain"],
        "heart_myocardium": ["heart", "hart", "myocardium"],
        "liver": ["liver", "lever"],
        "spleen": ["spleen", "milt"],
        "stomach": ["stomach", "maag"],
        "urinary_bladder": ["bladder", "urinary_bladder", "blaas"],
        "kidney_left": ["kidney_l", "kidney_left", "linker_nier", "nier_l", "left_kidney"],
        "kidney_right": ["kidney_r", "kidney_right", "rechter_nier", "nier_r", "right_kidney"],
    }
    ct_aff_lps_wb = _ct_voxel_to_lps_affine(wb_ct)

    for ts_name in anchor_organ_candidates:
        # Find matching RTSTRUCT name on source. Exact equality preferred over
        # substring matching to avoid 'brain' picking up 'brainstem' etc.
        candidates = name_aliases.get(ts_name, [ts_name])
        rt_match = None
        # Pass 1: exact equality (case-insensitive)
        for c in candidates:
            cl = c.lower()
            if cl in src_struct_lower:
                rt_match = src_struct_lower[cl]
                break
        # Pass 2: substring match (only if no exact match)
        if rt_match is None:
            for c in candidates:
                cl = c.lower()
                for sk_lower, sk_orig in src_struct_lower.items():
                    if cl in sk_lower:
                        rt_match = sk_orig
                        break
                if rt_match:
                    break
        if rt_match is None:
            continue

        # Find TS NIfTI on WB
        ts_path = wb_seg_dir / f"{ts_name}.nii.gz"
        if not ts_path.exists():
            ts_path = wb_seg_dir / f"{ts_name}.nii"
            if not ts_path.exists():
                continue

        # Compute centroids
        src_mask = src_struct_masks[rt_match]
        if not src_mask.any():
            continue
        idx = np.argwhere(src_mask)
        src_centroid = np.array([
            src_ct["x"][int(round(idx[:, 2].mean()))],
            src_ct["y"][int(round(idx[:, 1].mean()))],
            src_ct["z"][int(round(idx[:, 0].mean()))],
        ])

        wb_mask = _resample_one_nifti_to_ct(ts_path, wb_ct, ct_aff_lps_wb, seg_coord_system)
        if not wb_mask.any():
            continue
        idx = np.argwhere(wb_mask)
        wb_centroid = np.array([
            wb_ct["x"][int(round(idx[:, 2].mean()))],
            wb_ct["y"][int(round(idx[:, 1].mean()))],
            wb_ct["z"][int(round(idx[:, 0].mean()))],
        ])

        offset = np.asarray(src_iso_lps_mm) - src_centroid
        target_iso = wb_centroid + offset
        return target_iso, {
            "anchor_organ_ts": ts_name,
            "anchor_organ_rtstruct": rt_match,
            "src_anchor_centroid_mm": src_centroid.tolist(),
            "wb_anchor_centroid_mm": wb_centroid.tolist(),
            "src_iso_to_anchor_offset_mm": offset.tolist(),
        }
    raise RuntimeError(
        f"Could not find a usable anchor organ for target iso default. "
        f"Tried: {anchor_organ_candidates}. Available RTSTRUCT names: "
        f"{list(src_struct_masks.keys())[:10]}..."
    )


def compute_field_area_50pct_isodose(
    rtdose_path: Path,
    iso_lps_mm: np.ndarray,
    rx_gy: float,
) -> Dict:
    """Compute field area per the P3D paper: average of areas inside the 50%
    isodose in the coronal and sagittal planes through the isocenter.

    This is the methodology from Sanchez-Nieto et al. (2022). It accounts for
    actual delivered dose distribution including penumbra, scatter buildup,
    beam shaping, and MLC modulation. Recommended over jaw-based estimate.

    Returns: {'area_cm2', 'coronal_area_cm2', 'sagittal_area_cm2'}.
    """
    rtd = load_rtdose(rtdose_path)
    threshold_gy = 0.50 * rx_gy
    dose = rtd["dose_gy"]
    dx = abs(float(rtd["x"][1] - rtd["x"][0]))
    dy = abs(float(rtd["y"][1] - rtd["y"][0]))
    dz = abs(float(rtd["z"][1] - rtd["z"][0]))

    # Coronal plane: fixed y at iso, varies (x, z)
    iy = int(np.argmin(np.abs(rtd["y"] - iso_lps_mm[1])))
    coronal_mask = dose[:, iy, :] >= threshold_gy
    if coronal_mask.any():
        coronal_mask = binary_fill_holes(coronal_mask)
    a_coronal = float(coronal_mask.sum()) * dx * dz / 100.0  # mm² → cm²

    # Sagittal plane: fixed x at iso, varies (y, z)
    ix = int(np.argmin(np.abs(rtd["x"] - iso_lps_mm[0])))
    sagittal_mask = dose[:, :, ix] >= threshold_gy
    if sagittal_mask.any():
        sagittal_mask = binary_fill_holes(sagittal_mask)
    a_sagittal = float(sagittal_mask.sum()) * dy * dz / 100.0

    return {
        "area_cm2": 0.5 * (a_coronal + a_sagittal),
        "coronal_area_cm2": a_coronal,
        "sagittal_area_cm2": a_sagittal,
        "threshold_gy": threshold_gy,
        "iy_iso": iy,
        "ix_iso": ix,
    }


def load_rtdose(rtdose_path: Path) -> Dict:
    """Load an RTDOSE DICOM and return dose grid + LPS axis coordinates."""
    ds = pydicom.dcmread(str(rtdose_path))
    dose_gy = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    origin = [float(x) for x in ds.ImagePositionPatient]
    ps = [float(x) for x in ds.PixelSpacing]
    gfo = np.array([float(x) for x in ds.GridFrameOffsetVector])
    x = origin[0] + np.arange(int(ds.Columns)) * ps[1]
    y = origin[1] + np.arange(int(ds.Rows)) * ps[0]
    z = origin[2] + gfo
    return {
        "dose_gy": dose_gy,
        "x": x, "y": y, "z": z,
        "summation_type": str(getattr(ds, "DoseSummationType", "")),
        "max_gy": float(dose_gy.max()),
    }


# ============================================================
# RTSTRUCT segmentation (used for VB_HODGKIN verification)
# ============================================================
def contour_to_mask_2d(contour_data: List[float], ct_x: np.ndarray, ct_y: np.ndarray) -> np.ndarray:
    n_pts = len(contour_data) // 3
    pts = np.array(contour_data).reshape(n_pts, 3)
    col_idx = (pts[:, 0] - ct_x[0]) / (ct_x[1] - ct_x[0])
    row_idx = (pts[:, 1] - ct_y[0]) / (ct_y[1] - ct_y[0])
    polygon = np.column_stack([col_idx, row_idx])
    path = MplPath(polygon)
    cc, rr = np.meshgrid(np.arange(len(ct_x)), np.arange(len(ct_y)))
    grid = np.column_stack([cc.ravel(), rr.ravel()])
    return path.contains_points(grid).reshape(len(ct_y), len(ct_x))


def build_rtstruct_masks(
    rtstruct_path: Path,
    ct_x: np.ndarray, ct_y: np.ndarray, ct_z: np.ndarray,
) -> Dict[str, np.ndarray]:
    rs = pydicom.dcmread(str(rtstruct_path))
    name_by_num = {int(r.ROINumber): str(r.ROIName) for r in rs.StructureSetROISequence}
    masks: Dict[str, np.ndarray] = {}
    for rc in rs.ROIContourSequence:
        rname = name_by_num.get(int(rc.ReferencedROINumber), f"ROI_{rc.ReferencedROINumber}")
        if not hasattr(rc, "ContourSequence"):
            continue
        mask = np.zeros((len(ct_z), len(ct_y), len(ct_x)), dtype=bool)
        for c in rc.ContourSequence:
            cdata = [float(x) for x in c.ContourData]
            cz = cdata[2]
            zi = int(np.argmin(np.abs(ct_z - cz)))
            if abs(ct_z[zi] - cz) < 3.0:
                mask[zi] |= contour_to_mask_2d(cdata, ct_x, ct_y)
        if mask.any():
            masks[rname] = mask
    return masks


# ============================================================
# NIfTI segmentation loading (for cohort use case)
# ============================================================
def _stem_nii(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _ct_voxel_to_lps_affine(ct_info: Dict) -> np.ndarray:
    """Build the (slice, row, col) -> LPS mm affine for a CT volume.

    CT volume is stored as (n_slices, n_rows, n_cols). Per the DICOM HFS
    convention with IOP = [1, 0, 0, 0, 1, 0]:
      - col index increases in +x_LPS direction (Left), col spacing = ps[1]
      - row index increases in +y_LPS direction (Posterior), row spacing = ps[0]
      - slice index increases in +z_LPS direction (Superior), slice spacing = dz
    """
    dz = float(np.diff(ct_info["z"]).mean())
    return np.array([
        [0.0, 0.0, ct_info["pixel_spacing"][1], ct_info["x"][0]],  # x = ps[1]*col + x[0]
        [0.0, ct_info["pixel_spacing"][0], 0.0, ct_info["y"][0]],  # y = ps[0]*row + y[0]
        [dz,  0.0, 0.0,                          ct_info["z"][0]], # z = dz*slice + z[0]
        [0.0, 0.0, 0.0, 1.0],
    ])


def _resample_one_nifti_to_ct(
    nii_path: Path,
    ct_info: Dict,
    ct_aff_lps: np.ndarray,
    seg_coord_system: str,
) -> Optional[np.ndarray]:
    """Load one NIfTI and resample it to the CT grid as a uint8 boolean mask.

    Two-tier strategy:
    1. FAST PATH (~0.3s): if seg grid is just an axis permutation of CT grid
       (same physical voxels, different storage order), use np.transpose + flip.
       This is the common case for TotalSegmentator outputs generated from the
       same CT (TS uses RAS, CT DICOM uses LPS, axes (x,y,z) vs (z,y,x)).
    2. BBOX-AFFINE PATH (~3s): general affine resample on the segmentation
       bounding box (~3x speedup vs full-volume resample). Used when grids
       differ.
    """
    img = nib.load(str(nii_path))
    seg = np.asarray(img.dataobj)
    if seg.ndim == 4 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    seg_aff = np.array(img.affine, dtype=np.float64)
    if seg_coord_system.lower() == "ras":
        seg_aff = LPS_TO_RAS @ seg_aff
    nz, ny, nx = ct_info["volume"].shape

    # ----- Try fast path: axis-aligned grid match -----
    # Conditions: seg shape == (nx, ny, nz), and seg affine matches CT affine
    # under the transpose (2,1,0) and y-axis flip.
    if seg.shape == (nx, ny, nz):
        # CT affine maps (z_idx, y_idx, x_idx) -> LPS mm
        # Fast path equivalent affine maps (x_idx, y_idx, z_idx) for seg as
        # ct_voxel_to_lps_affine projected through transpose+flip
        # Equivalent test: project seg origin and unit-step through both
        # ways and compare the resulting LPS coords.
        # Simpler: the diagonal structure must satisfy:
        #   ct_aff[:3,:3] = seg_aff[:3,:3] @ P, where P is the perm+flip matrix
        # Build P: takes (z,y,x) -> seg's (x, ny-1-y, z) which means
        # column 0 of P picks seg's x axis index from CT z input -> [0,0,1]
        # etc. Just verify via origin-corner mapping.
        # Project CT voxel (0,0,0) and seg voxel (0,ny-1,0) (after y-flip), should match
        ct_origin_lps = ct_aff_lps @ np.array([0, 0, 0, 1.0])
        seg_origin_after_flip = seg_aff @ np.array([0, ny - 1, 0, 1.0])
        # Project CT (1,0,0) (z-step) and seg (0, ny-1, 1) (z-step in seg = axis 2)
        ct_zstep_lps = ct_aff_lps @ np.array([1, 0, 0, 1.0])
        seg_zstep_lps = seg_aff @ np.array([0, ny - 1, 1, 1.0])
        # Project CT (0,0,1) (x-step) and seg (1, ny-1, 0)
        ct_xstep_lps = ct_aff_lps @ np.array([0, 0, 1, 1.0])
        seg_xstep_lps = seg_aff @ np.array([1, ny - 1, 0, 1.0])

        if (np.allclose(ct_origin_lps, seg_origin_after_flip, atol=0.05) and
                np.allclose(ct_zstep_lps, seg_zstep_lps, atol=0.05) and
                np.allclose(ct_xstep_lps, seg_xstep_lps, atol=0.05)):
            return np.transpose(seg > 0, (2, 1, 0))[:, ::-1, :]

    # ----- Fallback: bbox-optimized affine resample -----
    seg_u8 = (seg > 0).astype(np.uint8)
    nz_seg, ny_seg, nx_seg = seg_u8.shape
    nonzero = np.argwhere(seg_u8)
    if nonzero.size == 0:
        return np.zeros((nz, ny, nx), dtype=bool)

    pad = 2
    z0, z1 = max(0, nonzero[:, 0].min() - pad), min(nz_seg, nonzero[:, 0].max() + 1 + pad)
    y0, y1 = max(0, nonzero[:, 1].min() - pad), min(ny_seg, nonzero[:, 1].max() + 1 + pad)
    x0, x1 = max(0, nonzero[:, 2].min() - pad), min(nx_seg, nonzero[:, 2].max() + 1 + pad)

    seg_to_ct = np.linalg.inv(ct_aff_lps) @ seg_aff
    corners = np.array([[x, y, z, 1.0]
                        for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    ct_corners = (seg_to_ct @ corners.T).T[:, :3]
    cx0 = max(0, int(np.floor(ct_corners[:, 0].min())) - pad)
    cx1 = min(nx, int(np.ceil(ct_corners[:, 0].max())) + 1 + pad)
    cy0 = max(0, int(np.floor(ct_corners[:, 1].min())) - pad)
    cy1 = min(ny, int(np.ceil(ct_corners[:, 1].max())) + 1 + pad)
    cz0 = max(0, int(np.floor(ct_corners[:, 2].min())) - pad)
    cz1 = min(nz, int(np.ceil(ct_corners[:, 2].max())) + 1 + pad)

    if cx1 <= cx0 or cy1 <= cy0 or cz1 <= cz0:
        return np.zeros((nz, ny, nx), dtype=bool)

    bbox_shape = (cz1 - cz0, cy1 - cy0, cx1 - cx0)
    ct_to_seg = np.linalg.inv(seg_aff) @ ct_aff_lps
    matrix = ct_to_seg[:3, :3]
    offset_full = ct_to_seg[:3, 3]
    offset_bbox = matrix @ np.array([cz0, cy0, cx0]) + offset_full

    sub = affine_transform(
        seg_u8, matrix=matrix, offset=offset_bbox,
        output_shape=bbox_shape, order=0, mode="constant", cval=0,
    )
    out = np.zeros((nz, ny, nx), dtype=bool)
    out[cz0:cz1, cy0:cy1, cx0:cx1] = sub.astype(bool)
    return out


class PackedMaskAccumulator:
    """Memory-efficient boolean-mask accumulator using np.packbits (8x compression).

    For a 414x512x512 grid, raw bool mask = 108 MB; packed = 13.5 MB. With many
    organ groups defined (e.g. 26 groups) the savings are essential to stay under
    the 4 GB process memory budget.

    Methods:
        or_(mask): logical-OR a new bool mask into this accumulator.
        unpack(): return the full-size bool mask (allocates 108 MB temporarily).
        count(): population count (number of True voxels) without full unpack.
        is_empty(): True if no mask has been OR'd in yet.
    """
    __slots__ = ("shape", "_packed", "_n_voxels")

    def __init__(self, shape):
        self.shape = tuple(shape)
        self._n_voxels = int(np.prod(self.shape))
        self._packed = None  # type: Optional[np.ndarray]

    def or_(self, mask: np.ndarray) -> None:
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} != accumulator shape {self.shape}")
        if self._packed is None:
            self._packed = np.packbits(mask.ravel())
        else:
            unpacked = np.unpackbits(self._packed, count=self._n_voxels).astype(bool)
            unpacked |= mask.ravel()
            self._packed = np.packbits(unpacked)
            del unpacked

    def unpack(self) -> np.ndarray:
        if self._packed is None:
            return np.zeros(self.shape, dtype=bool)
        return np.unpackbits(self._packed, count=self._n_voxels).astype(bool).reshape(self.shape)

    def count(self) -> int:
        if self._packed is None:
            return 0
        # popcount of packed bytes -- cheaper than full unpack
        return int(np.unpackbits(self._packed, count=self._n_voxels).sum())

    def is_empty(self) -> bool:
        return self._packed is None


def iter_nifti_masks_with_groups(
    seg_dir: Path,
    ct_info: Dict,
    seg_coord_system: str = "ras",
    min_voxels: int = 5,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    organ_groups: Optional[Dict[str, List[str]]] = None,
    keep_individual_members: bool = False,
):
    """Stream organ masks with optional grouping into combined clinical organs.

    Two-pass logic:
      Pass 1: Stream every NIfTI in the directory. For each:
              - If it is a member of any group, accumulate it (logical OR).
              - Then apply exclude/include regex to decide if it should be
                yielded as an individual organ.
      Pass 2: Yield each accumulated group mask.

    The KEY DESIGN POINT is that exclude_regex is applied AFTER group
    membership check. This means groups always receive their members even
    when those members would otherwise be filtered out. So you can use the
    default exclude (which hides individual ribs/vertebrae from the report)
    while still getting the 'ribs' and 'vertebral_column' grouped masks.
    """
    if organ_groups is None:
        organ_groups = {}
    member_to_groups: Dict[str, List[str]] = {}
    for gname, members in organ_groups.items():
        for m in members:
            member_to_groups.setdefault(m, []).append(gname)

    nz, ny, nx = ct_info["volume"].shape
    accumulators: Dict[str, PackedMaskAccumulator] = {
        g: PackedMaskAccumulator((nz, ny, nx)) for g in organ_groups
    }
    n_members_seen: Dict[str, int] = {g: 0 for g in organ_groups}

    inc_pat = re.compile(include_regex, re.IGNORECASE) if include_regex else None
    exc_pat = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    n_filtered_individuals = 0

    for name, mask in iter_nifti_masks_on_ct_grid(
        seg_dir, ct_info,
        seg_coord_system=seg_coord_system,
        min_voxels=min_voxels,
        # NB: we deliberately do NOT pass include/exclude here so all NIfTIs
        # are loaded; filtering happens below per-organ post-load.
    ):
        belongs_to = member_to_groups.get(name, [])
        # Accumulate into any matching group (packed)
        for g in belongs_to:
            accumulators[g].or_(mask)
            n_members_seen[g] += 1
        # Decide whether to yield individually
        yield_as_individual = True
        if not keep_individual_members and belongs_to:
            yield_as_individual = False
        if yield_as_individual and exc_pat is not None and exc_pat.search(name):
            yield_as_individual = False
            n_filtered_individuals += 1
        if yield_as_individual and inc_pat is not None and not inc_pat.search(name):
            yield_as_individual = False
            n_filtered_individuals += 1
        if yield_as_individual:
            yield name, mask, "individual"
        else:
            del mask  # release memory if we're not yielding

    if n_filtered_individuals:
        print(f"  Filtered {n_filtered_individuals} individual organs (kept in groups if applicable)")

    for g, acc in accumulators.items():
        if acc.is_empty():
            continue
        if acc.count() < min_voxels:
            continue
        yield g, acc.unpack(), f"group ({n_members_seen[g]} members)"


def iter_nifti_masks_on_ct_grid(
    seg_dir: Path,
    ct_info: Dict,
    seg_coord_system: str = "ras",
    min_voxels: int = 5,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
):
    """Yield (organ_name, mask) pairs one at a time. Streaming = memory-friendly.

    Optional include/exclude regexes are matched against the file stem (organ name).
    Exclude is applied first; then include filters in. Match is case-insensitive.
    """
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel is required for NIfTI segmentation loading")
    nii_files = sorted([p for p in seg_dir.iterdir()
                        if p.suffix == ".nii" or p.name.lower().endswith(".nii.gz")])
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files in {seg_dir}")
    inc_pat = re.compile(include_regex, re.IGNORECASE) if include_regex else None
    exc_pat = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None
    ct_aff_lps = _ct_voxel_to_lps_affine(ct_info)
    n_skipped_filter = 0
    for nii_path in nii_files:
        name = _stem_nii(nii_path)
        if exc_pat is not None and exc_pat.search(name):
            n_skipped_filter += 1
            continue
        if inc_pat is not None and not inc_pat.search(name):
            n_skipped_filter += 1
            continue
        try:
            mask = _resample_one_nifti_to_ct(nii_path, ct_info, ct_aff_lps, seg_coord_system)
        except Exception as e:
            print(f"  Skipping {nii_path.name}: load/resample error: {e}")
            continue
        if mask.sum() < min_voxels:
            continue
        yield name, mask
        del mask
    if n_skipped_filter:
        print(f"  Filter skipped {n_skipped_filter} NIfTI files")


def load_nifti_masks_to_ct_grid(
    seg_dir: Path,
    ct_info: Dict,
    seg_coord_system: str = "ras",
    min_voxels: int = 5,
) -> Dict[str, np.ndarray]:
    """Load a directory of binary NIfTI masks and resample each to the CT grid.

    Returns a dict keyed by organ name (file stem). Memory-heavy on large WB grids
    with many organs; for those use iter_nifti_masks_on_ct_grid instead.
    """
    if not HAS_NIBABEL:
        raise RuntimeError("nibabel is required for NIfTI segmentation loading")

    nii_files = sorted([p for p in seg_dir.iterdir()
                        if p.suffix == ".nii" or p.name.lower().endswith(".nii.gz")])
    if not nii_files:
        raise FileNotFoundError(f"No NIfTI files in {seg_dir}")

    nz, ny, nx = ct_info["volume"].shape
    ct_aff_lps = _ct_voxel_to_lps_affine(ct_info)

    masks: Dict[str, np.ndarray] = {}
    for nii_path in nii_files:
        try:
            mask = _resample_one_nifti_to_ct(nii_path, ct_info, ct_aff_lps, seg_coord_system)
        except Exception as e:
            print(f"  Skipping {nii_path.name}: load/resample error: {e}")
            continue
        if mask.sum() < min_voxels:
            print(f"  Skipping {nii_path.name}: only {int(mask.sum())} voxels after resample")
            continue
        masks[_stem_nii(nii_path)] = mask
    return masks


# ============================================================
# Hybrid dose pipeline
# ============================================================
def interpolate_rtdose_to_ct(
    rtdose: Dict,
    ct_info: Dict,
    shift_lps_mm: np.ndarray,
) -> np.ndarray:
    """Interpolate RTDOSE onto the CT grid after shifting RTDOSE coordinates.

    The shift is applied to the RTDOSE axis coordinates, i.e. the dose
    distribution is translated by `shift_lps_mm` in DICOM patient coordinates.
    Querying the original RTDOSE at the CT point (x, y, z) is equivalent to
    querying the shifted RTDOSE at (x - sx, y - sy, z - sz).
    """
    interp = RegularGridInterpolator(
        (rtdose["z"], rtdose["y"], rtdose["x"]),
        rtdose["dose_gy"],
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    nz, ny, nx = ct_info["volume"].shape
    dose_on_ct = np.zeros((nz, ny, nx), dtype=np.float32)
    yy, xx = np.meshgrid(ct_info["y"], ct_info["x"], indexing="ij")
    sx, sy, sz = float(shift_lps_mm[0]), float(shift_lps_mm[1]), float(shift_lps_mm[2])
    for i, zp in enumerate(ct_info["z"]):
        zz = np.full_like(xx, zp - sz)
        pts = np.stack([zz.ravel(), (yy - sy).ravel(), (xx - sx).ravel()], axis=-1)
        dose_on_ct[i] = interp(pts).reshape(ny, nx)
    return dose_on_ct


def compute_p3d_dose_on_ct(
    ct_info: Dict,
    target_iso_mm: np.ndarray,
    p3d: P3DParams,
    total_mu: float,
    return_dist: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Compute P3D absolute dose (Gy) on the CT grid, plus optional distance map (cm).

    Slice-iterative to avoid allocating full 3D dx/dy/dz arrays.
    """
    nz, ny, nx = ct_info["volume"].shape
    dose = np.zeros((nz, ny, nx), dtype=np.float32)
    dist = np.zeros((nz, ny, nx), dtype=np.float32) if return_dist else None
    # 2D dx, dy planes (do not depend on z)
    dx2 = (ct_info["x"][None, :] - target_iso_mm[0]) / 10.0  # (1, nx)
    dy2 = (ct_info["y"][:, None] - target_iso_mm[1]) / 10.0  # (ny, 1)
    dx2_b, dy2_b = np.broadcast_arrays(dx2, dy2)             # both (ny, nx)
    dx2_b = np.ascontiguousarray(dx2_b, dtype=np.float32)
    dy2_b = np.ascontiguousarray(dy2_b, dtype=np.float32)
    for k, zp in enumerate(ct_info["z"]):
        dz_val = float((zp - target_iso_mm[2]) / 10.0)
        dz2 = np.full((ny, nx), dz_val, dtype=np.float32)
        ppd = p3d_ppd_mgy_per_mu(dx2_b, dy2_b, dz2, p3d)
        dose[k] = (np.nan_to_num(ppd, nan=0.0) * total_mu / 1000.0).astype(np.float32)
        if return_dist:
            dist[k] = np.sqrt(dx2_b * dx2_b + dy2_b * dy2_b + dz_val * dz_val)
    return dose, dist


def build_5pct_mask(dose_on_ct: np.ndarray, rx_gy: float) -> np.ndarray:
    thr = 0.05 * rx_gy
    mask = dose_on_ct >= thr
    for i in range(mask.shape[0]):
        if mask[i].any():
            mask[i] = binary_fill_holes(mask[i])
    return mask


def organ_dvh(dose_gy: np.ndarray, mask: np.ndarray, n_bins: int = 200) -> pd.DataFrame:
    """Cumulative DVH as a pandas DataFrame with columns dose_gy, volume_pct."""
    vals = dose_gy[mask]
    if vals.size == 0:
        return pd.DataFrame({"dose_gy": [], "volume_pct": []})
    dmax = float(vals.max())
    edges = np.linspace(0.0, max(dmax, 1e-6), n_bins + 1)
    hist, _ = np.histogram(vals, bins=edges)
    cum = np.cumsum(hist[::-1])[::-1]
    pct = 100.0 * cum / vals.size
    return pd.DataFrame({"dose_gy": edges[:-1], "volume_pct": pct})


def organ_stats(
    dose_gy: np.ndarray,
    dist_cm: np.ndarray,
    mask: np.ndarray,
    mask_tps: np.ndarray,
    voxel_cc: float,
) -> Dict:
    n = int(mask.sum())
    if n == 0:
        return {}
    in_tps = int((mask & mask_tps).sum())
    hd = dose_gy[mask]
    dd = dist_cm[mask]
    return {
        "Volume_cc": round(n * voxel_cc, 1),
        "n_voxels": n,
        "pct_TPS": round(100.0 * in_tps / n, 1),
        "pct_P3D": round(100.0 * (n - in_tps) / n, 1),
        "Dmean_mGy": round(float(hd.mean()) * 1000, 2),
        "Dmax_mGy": round(float(hd.max()) * 1000, 2),
        "Dmin_mGy": round(float(hd.min()) * 1000, 2),
        "Dmedian_mGy": round(float(np.median(hd)) * 1000, 2),
        "D2pct_mGy": round(float(np.percentile(hd, 98)) * 1000, 2),
        "D98pct_mGy": round(float(np.percentile(hd, 2)) * 1000, 2),
        "mean_dist_cm": round(float(dd.mean()), 1),
        "min_dist_cm": round(float(dd.min()), 1),
    }


# ============================================================
# Figures
# ============================================================
def make_alignment_verification_figure(
    ct_info: Dict,
    dose_tps: np.ndarray,
    dose_hybrid: np.ndarray,
    mask_tps: np.ndarray,
    target_iso_mm: np.ndarray,
    rx_gy: float,
    seg_dir: Optional[Path],
    seg_coord_system: str,
    field_area_cm2: float,
    field_factor: float,
    out_path: Path,
    anchor_organs: Sequence[str] = ("brain", "heart_myocardium", "liver"),
) -> None:
    """3-panel alignment verification: in-field anchor + 2 OOF anchors at very
    different distances from iso. Each panel shows CT + dose overlay + organ
    contour, to give a quick visual sanity-check on:
      (1) the target_iso falls inside the in-field anchor
      (2) OOF anchors are correctly placed in the WB grid
      (3) dose magnitude drops with distance as expected
    """
    if seg_dir is None or not HAS_NIBABEL:
        return
    ct_aff = _ct_voxel_to_lps_affine(ct_info)
    panels = []
    for organ in anchor_organs:
        nii = seg_dir / f"{organ}.nii.gz"
        if not nii.exists():
            nii = seg_dir / f"{organ}.nii"
            if not nii.exists():
                continue
        mask = _resample_one_nifti_to_ct(nii, ct_info, ct_aff, seg_coord_system)
        if mask is None or not mask.any():
            continue
        ck = int(round(np.argwhere(mask)[:, 0].mean()))
        panels.append((organ, mask, ck))
    if not panels:
        return

    ext = [ct_info["x"][0], ct_info["x"][-1],
           ct_info["y"][-1], ct_info["y"][0]]
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]
    colors = ["red", "magenta", "yellow", "cyan", "orange"]

    for idx, (organ, mask, ck) in enumerate(panels):
        ax = axes[idx]
        ax.imshow(ct_info["volume"][ck], cmap="gray", vmin=-200, vmax=400, extent=ext)
        # Use TPS for the in-field organ (first panel), hybrid for OOF
        if idx == 0:
            dose_for_panel = dose_tps[ck]
            scale = "linear"
        else:
            dose_for_panel = dose_hybrid[ck]
            scale = "log"
        if scale == "linear":
            dov = np.ma.masked_where(dose_for_panel < 0.1, dose_for_panel)
            im = ax.imshow(dov, cmap="jet", alpha=0.5,
                           vmin=0, vmax=rx_gy * 1.1, extent=ext)
        else:
            dov = np.ma.masked_where(dose_for_panel < 1e-5, dose_for_panel)
            im = ax.imshow(dov, cmap="jet", alpha=0.5,
                           norm=LogNorm(vmin=1e-4, vmax=rx_gy * 1.1), extent=ext)
        if mask_tps[ck].any():
            ax.contour(ct_info["x"], ct_info["y"],
                       mask_tps[ck].astype(float), levels=[0.5],
                       colors="lime", linewidths=2)
        ax.contour(ct_info["x"], ct_info["y"], mask[ck].astype(float),
                   levels=[0.5], colors=colors[idx % len(colors)],
                   linewidths=1.5, linestyles="--")
        if idx == 0:
            ax.plot(target_iso_mm[0], target_iso_mm[1], "k+", markersize=15, mew=2)
        d_cm = abs(ct_info["z"][ck] - target_iso_mm[2]) / 10.0
        ax.set_title(f"{organ}  z={ct_info['z'][ck]:.0f}mm  ({d_cm:.0f}cm from iso)")
        ax.set_xlabel("x (mm)")
        if idx == 0:
            ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, label="Gy")

    fig.suptitle(
        f"Alignment verification (FU={field_area_cm2:.2f} cm², F={field_factor:.3f}); "
        f"5% isodose = lime, organ contour = dashed",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def make_axial_figure(
    ct_info: Dict,
    dose_tps: np.ndarray,
    dose_hybrid: np.ndarray,
    mask_tps: np.ndarray,
    target_iso_mm: np.ndarray,
    rx_gy: float,
    epsilon: float,
    field_factor: float,
    out_path: Path,
) -> None:
    iz = int(np.argmin(np.abs(ct_info["z"] - target_iso_mm[2])))
    ext = [ct_info["x"][0], ct_info["x"][-1], ct_info["y"][-1], ct_info["y"][0]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    s = ct_info["volume"][iz]

    axes[0, 0].imshow(s, cmap="gray", vmin=-200, vmax=400, extent=ext)
    axes[0, 0].set_title(f"CT (z={ct_info['z'][iz]:.0f} mm, target iso)")
    axes[0, 0].set_xlabel("x (mm)"); axes[0, 0].set_ylabel("y (mm)")

    axes[0, 1].imshow(s, cmap="gray", vmin=-200, vmax=400, extent=ext)
    dov = np.ma.masked_where(dose_tps[iz] < 0.1, dose_tps[iz])
    im = axes[0, 1].imshow(dov, cmap="jet", alpha=0.5, vmin=0, vmax=rx_gy * 1.1, extent=ext)
    axes[0, 1].contour(ct_info["x"], ct_info["y"], mask_tps[iz].astype(float),
                       levels=[0.5], colors="lime", linewidths=2)
    axes[0, 1].set_title("TPS (re-anchored) + 5% boundary"); axes[0, 1].set_xlabel("x (mm)")
    plt.colorbar(im, ax=axes[0, 1], label="Gy")

    axes[0, 2].imshow(s, cmap="gray", vmin=-200, vmax=400, extent=ext)
    hov = np.ma.masked_where(dose_hybrid[iz] < 1e-5, dose_hybrid[iz])
    im = axes[0, 2].imshow(hov, cmap="jet", alpha=0.5,
                           norm=LogNorm(vmin=1e-4, vmax=rx_gy * 1.1), extent=ext)
    axes[0, 2].contour(ct_info["x"], ct_info["y"], mask_tps[iz].astype(float),
                       levels=[0.5], colors="lime", linewidths=2)
    axes[0, 2].set_title("Hybrid (log)"); axes[0, 2].set_xlabel("x (mm)")
    plt.colorbar(im, ax=axes[0, 2], label="Gy")

    for j, off in enumerate([-20, -10, 10]):
        zi = max(0, min(len(ct_info["z"]) - 1, iz + off))
        ax = axes[1, j]
        ax.imshow(ct_info["volume"][zi], cmap="gray", vmin=-200, vmax=400, extent=ext)
        h = np.ma.masked_where(dose_hybrid[zi] < 1e-5, dose_hybrid[zi])
        im = ax.imshow(h, cmap="jet", alpha=0.5,
                       norm=LogNorm(vmin=1e-4, vmax=rx_gy * 1.1), extent=ext)
        if mask_tps[zi].any():
            ax.contour(ct_info["x"], ct_info["y"], mask_tps[zi].astype(float),
                       levels=[0.5], colors="lime", linewidths=2)
        d_iso_cm = abs(ct_info["z"][zi] - target_iso_mm[2]) / 10.0
        ax.set_title(f"z={ct_info['z'][zi]:.0f} mm ({d_iso_cm:.0f} cm from iso)")
        ax.set_xlabel("x (mm)")
        plt.colorbar(im, ax=ax, label="Gy")

    plt.suptitle(
        f"Cross-patient hybrid -- Rx={rx_gy} Gy, eps={epsilon:.3f}, F={field_factor:.3f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_zprofile_figure(
    ct_info: Dict,
    dose_tps: np.ndarray,
    dose_p3d: np.ndarray,
    dose_hybrid: np.ndarray,
    target_iso_mm: np.ndarray,
    rx_gy: float,
    out_path: Path,
) -> None:
    iy = int(np.argmin(np.abs(ct_info["y"] - target_iso_mm[1])))
    ix = int(np.argmin(np.abs(ct_info["x"] - target_iso_mm[0])))
    thr = 0.05 * rx_gy
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(ct_info["z"], np.clip(dose_tps[:, iy, ix] * 1000, 0.001, None),
                "b-", lw=2, label="TPS (re-anchored)", alpha=0.7)
    ax.semilogy(ct_info["z"], np.clip(dose_p3d[:, iy, ix] * 1000, 0.001, None),
                "r--", lw=2, label="P3D", alpha=0.7)
    ax.semilogy(ct_info["z"], np.clip(dose_hybrid[:, iy, ix] * 1000, 0.001, None),
                "k-", lw=2.5, label="Hybrid", alpha=0.9)
    ax.axhline(y=thr * 1000, color="green", ls=":", lw=1.5,
               label=f"5% ({thr * 1000:.0f} mGy)")
    ax.axvline(x=target_iso_mm[2], color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("z (mm)"); ax.set_ylabel("Dose (mGy)")
    ax.set_title("Dose profile along z through target isocenter")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.001, 50000)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_organ_figure(df: pd.DataFrame, rx_gy: float, out_path: Path,
                      target_keywords=("PTV", "CTV", "GTV")) -> None:
    excl = df["Organ"].apply(lambda s: any(k in s for k in target_keywords))
    oar = df.loc[~excl].copy()
    if oar.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
    nm = oar["Organ"].tolist()
    xp = np.arange(len(nm)); w = 0.25
    a1.bar(xp - w, oar["Dmean_mGy"], w, label="Mean", color="steelblue")
    a1.bar(xp,     oar["Dmedian_mGy"], w, label="Median", color="coral")
    a1.bar(xp + w, oar["Dmax_mGy"], w, label="Max", color="seagreen", alpha=0.7)
    a1.set_xticks(xp); a1.set_xticklabels(nm, rotation=45, ha="right")
    a1.set_ylabel("mGy"); a1.set_title("OAR hybrid doses")
    a1.legend(); a1.set_yscale("log"); a1.grid(True, alpha=0.3, axis="y")
    a2.barh(nm, oar["pct_TPS"], color="steelblue", label="TPS (>=5%)")
    a2.barh(nm, oar["pct_P3D"], left=oar["pct_TPS"], color="coral", label="P3D (<5%)")
    a2.set_xlabel("%"); a2.set_title("TPS vs P3D contribution per organ")
    a2.legend(loc="lower right"); a2.set_xlim(0, 100)
    plt.suptitle(f"OAR analysis -- Rx={rx_gy} Gy", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_dvh_plot(dvh_df: pd.DataFrame, organ: str, out_path: Path) -> None:
    if dvh_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dvh_df["dose_gy"] * 1000.0, dvh_df["volume_pct"], "b-", lw=2)
    ax.set_xlabel("Dose (mGy)"); ax.set_ylabel("Volume (%)")
    ax.set_title(f"Cumulative DVH -- {organ}")
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# Main pipeline
# ============================================================
def run(
    wb_ct_dir: Path,
    rtdose_path: Path,
    source_iso_mm: np.ndarray,
    target_iso_mm: np.ndarray,
    rx_gy: float,
    n_fractions: int,
    total_mu: float,
    field_area_cm2: float,
    energy_mv: float,
    leakage_mgy_per_mu: float,
    seg_nifti_dir: Optional[Path],
    rtstruct_path: Optional[Path],
    seg_coord_system: str,
    outdir: Path,
    write_dvhs: bool = True,
    seg_include_regex: Optional[str] = None,
    seg_exclude_regex: Optional[str] = None,
    organ_groups: Optional[Dict[str, List[str]]] = None,
    keep_individual_members: bool = False,
) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CROSS-PATIENT HYBRID DOSE PIPELINE")
    print("=" * 72)

    # --- Step 1: Load WB CT and RTDOSE ---
    print("\n[1/6] Loading WB CT and RTDOSE")
    ct = load_ct_volume(wb_ct_dir)
    print(f"  WB CT: {ct['n_slices']} slices, {ct['volume'].shape[1]}x{ct['volume'].shape[2]}, "
          f"z=[{ct['z'][0]:.1f}, {ct['z'][-1]:.1f}] mm")
    print(f"  Patient ID on WB CT: {ct['patient_id']}")

    rtd = load_rtdose(rtdose_path)
    print(f"  RTDOSE: shape={rtd['dose_gy'].shape}, max={rtd['max_gy']:.2f} Gy ({rtd['summation_type']})")

    # --- Step 2: Re-anchor RTDOSE onto WB CT grid ---
    print("\n[2/6] Re-anchoring RTDOSE on WB CT")
    shift = target_iso_mm - source_iso_mm
    print(f"  Source iso (LPS mm): {source_iso_mm.tolist()}")
    print(f"  Target iso (LPS mm): {target_iso_mm.tolist()}")
    print(f"  Shift applied:       {shift.tolist()} mm")
    dose_tps_on_ct = interpolate_rtdose_to_ct(rtd, ct, shift)
    print(f"  Max TPS on WB grid:  {dose_tps_on_ct.max():.2f} Gy")
    if dose_tps_on_ct.max() < 1e-3:
        print("  WARNING: TPS dose is essentially zero on the WB grid -- check that")
        print("           target isocenter falls within the WB CT volume.")

    # --- Step 3: P3D dose on WB CT grid ---
    print("\n[3/6] Computing P3D dose on WB CT grid")
    eu = (rx_gy / max(n_fractions, 1)) * 1000.0 / max(total_mu / max(n_fractions, 1), 1e-9)
    epsilon = eu / REF_EU_MGY_PER_MU
    field_factor = field_area_cm2 / REF_FIELD_AREA_CM2
    p3d_params = P3DParams(epsilon=epsilon, field_factor=field_factor, leakage=leakage_mgy_per_mu)
    print(f"  EU = {eu:.4f} mGy/MU,  eps = {epsilon:.4f},  F = {field_factor:.4f}")
    print(f"  Total MU = {total_mu:.0f},  Lu = {leakage_mgy_per_mu} mGy/MU")
    dose_p3d_gy, dist_cm = compute_p3d_dose_on_ct(ct, target_iso_mm, p3d_params, total_mu)
    print(f"  Sample P3D values along +z from target iso:")
    for d_cm in [5, 10, 20, 30, 50]:
        v = p3d_ppd_mgy_per_mu(np.array([0.0]), np.array([0.0]),
                               np.array([float(d_cm)]), p3d_params)
        print(f"    dz={d_cm:>2} cm: {v[0]:.4f} mGy/MU -> {v[0] * total_mu:.1f} mGy total")

    # --- Step 4: 5% mask + hybrid merge ---
    print("\n[4/6] Building 5% isodose mask and merging")
    mask_tps = build_5pct_mask(dose_tps_on_ct, rx_gy)
    print(f"  5% threshold: {0.05 * rx_gy:.2f} Gy ({0.05 * rx_gy * 1000:.0f} mGy)")
    print(f"  Voxels inside 5%: {int(mask_tps.sum())} ({100 * mask_tps.sum() / mask_tps.size:.2f}%)")
    hybrid = np.where(mask_tps, dose_tps_on_ct, dose_p3d_gy).astype(np.float32)
    print(f"  TPS max inside mask:  {dose_tps_on_ct[mask_tps].max() if mask_tps.any() else 0:.2f} Gy")
    print(f"  P3D max outside mask: {dose_p3d_gy[~mask_tps].max():.4f} Gy")
    print(f"  Hybrid max:           {hybrid.max():.2f} Gy")

    # --- Make figures FIRST while dose_tps and dose_p3d still in memory ---
    # Then free those big arrays before the organ-streaming loop, since per-organ
    # accumulators consume substantial memory and we don't need separate TPS/P3D
    # arrays for organ stats (we use the merged 'hybrid' + 'dist_cm').
    print("\n[5/6] Figures (built before organ loop to free dose arrays)")
    make_axial_figure(ct, dose_tps_on_ct, hybrid, mask_tps,
                      target_iso_mm, rx_gy, epsilon, field_factor,
                      outdir / "fig1_axial_slices.png")
    make_zprofile_figure(ct, dose_tps_on_ct, dose_p3d_gy, hybrid,
                         target_iso_mm, rx_gy,
                         outdir / "fig2_z_profile.png")
    # Alignment verification (only if WB seg dir available)
    if seg_nifti_dir is not None:
        make_alignment_verification_figure(
            ct, dose_tps_on_ct, hybrid, mask_tps, target_iso_mm, rx_gy,
            seg_nifti_dir, seg_coord_system,
            field_area_cm2, field_factor,
            outdir / "fig4_alignment_verification.png",
        )
    # Free the per-modality dose arrays; subsequent organ stats use 'hybrid' only
    del dose_tps_on_ct, dose_p3d_gy
    import gc; gc.collect()

    # --- Step 6: Organ statistics ---
    print("\n[6/6] Organ statistics")
    dz_mm = abs(float(np.diff(ct["z"]).mean()))
    voxel_cc = ct["pixel_spacing"][0] * ct["pixel_spacing"][1] * dz_mm / 1000.0

    if seg_nifti_dir is not None:
        print(f"  Will stream NIfTI segmentations from {seg_nifti_dir}")
        masks = {}  # not used in streaming branch
    elif rtstruct_path is not None:
        print(f"  Loading RTSTRUCT from {rtstruct_path}")
        masks = build_rtstruct_masks(rtstruct_path, ct["x"], ct["y"], ct["z"])
        print(f"  Built {len(masks)} ROI masks")
    else:
        print("  No segmentations provided -- skipping organ statistics")
        masks = {}

    rows = []
    dvh_dir = outdir / "dvhs"
    if write_dvhs:
        dvh_dir.mkdir(exist_ok=True)

    if seg_nifti_dir is not None:
        # Streaming path: load and process one NIfTI mask at a time, with groups
        print(f"  Streaming NIfTI segmentations from {seg_nifti_dir}")
        if organ_groups:
            print(f"  Organ groups enabled: {len(organ_groups)} groups, "
                  f"keep_individuals={keep_individual_members}")
        n_processed = 0
        for organ, mask, kind in iter_nifti_masks_with_groups(
            seg_nifti_dir, ct,
            seg_coord_system=seg_coord_system,
            include_regex=seg_include_regex,
            exclude_regex=seg_exclude_regex,
            organ_groups=organ_groups,
            keep_individual_members=keep_individual_members,
        ):
            s = organ_stats(hybrid, dist_cm, mask, mask_tps, voxel_cc)
            if not s:
                continue
            s = {"Organ": organ, "ts_kind": kind, **s}
            rows.append(s)
            if write_dvhs:
                dvh = organ_dvh(hybrid, mask)
                safe = re.sub(r'[^A-Za-z0-9_-]', '_', organ)
                dvh.to_csv(dvh_dir / f"dvh_{safe}.csv", index=False)
                write_dvh_plot(dvh, organ, dvh_dir / f"dvh_{safe}.png")
            n_processed += 1
            del mask
        print(f"  Processed {n_processed} organ masks (individual + groups)")
    else:
        # In-memory path (RTSTRUCT or empty)
        for organ, mask in masks.items():
            s = organ_stats(hybrid, dist_cm, mask, mask_tps, voxel_cc)
            if not s:
                continue
            s = {"Organ": organ, **s}
            rows.append(s)
            if write_dvhs:
                dvh = organ_dvh(hybrid, mask)
                safe = re.sub(r'[^A-Za-z0-9_-]', '_', organ)
                dvh.to_csv(dvh_dir / f"dvh_{safe}.csv", index=False)
                write_dvh_plot(dvh, organ, dvh_dir / f"dvh_{safe}.png")
    df = pd.DataFrame(rows)
    csv_path = outdir / "hybrid_organ_doses.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    # --- Final: organ figure + run info ---
    if not df.empty:
        make_organ_figure(df, rx_gy, outdir / "fig3_organ_doses.png")

    info = {
        "wb_ct_dir": str(wb_ct_dir),
        "wb_patient_id": ct["patient_id"],
        "rtdose_path": str(rtdose_path),
        "source_isocenter_mm": source_iso_mm.tolist(),
        "target_isocenter_mm": target_iso_mm.tolist(),
        "shift_mm": shift.tolist(),
        "rx_gy": rx_gy,
        "n_fractions": n_fractions,
        "total_mu": total_mu,
        "field_area_cm2": field_area_cm2,
        "energy_mv": energy_mv,
        "leakage_mgy_per_mu": leakage_mgy_per_mu,
        "epsilon": round(epsilon, 4),
        "field_factor": round(field_factor, 4),
        "p3d_constants": {"A1": A1_MGY_CM2_PER_MU, "A2": A2_MGY_CM_PER_MU, "A3": A3_PER_CM},
        "wb_ct_shape": list(ct["volume"].shape),
        "wb_ct_z_range_mm": [float(ct["z"][0]), float(ct["z"][-1])],
        "n_organs": len(rows),
        "notes": [
            "P3D model: Sanchez-Nieto et al., Frontiers in Oncology, 12:872752 (2022).",
            "TPS dose was re-anchored from source isocenter to target isocenter via "
            "rigid translation in DICOM LPS coordinates.",
            "Hybrid: TPS dose used inside the 5% isodose, P3D outside.",
            "P3D is not valid inside the 5% isodose or for skin (paper limitations).",
        ],
    }
    with open(outdir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Wrote {outdir / 'run_info.json'}")

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Rx={rx_gy} Gy | {n_fractions} fx | {total_mu:.0f} MU | "
          f"eps={epsilon:.4f} | F={field_factor:.4f}")
    print(f"Source iso: {source_iso_mm.tolist()} mm")
    print(f"Target iso: {target_iso_mm.tolist()} mm")
    print(f"Shift:      {shift.tolist()} mm\n")
    if not df.empty:
        print(f"{'Organ':<22} {'Vol':>7} {'%TPS':>6} {'%P3D':>6} "
              f"{'Dmean':>8} {'Dmax':>8} {'Dmed':>8} {'D2%':>8} {'D98%':>8}")
        print("-" * 92)
        for _, r in df.iterrows():
            print(f"{r['Organ']:<22} {r['Volume_cc']:>6.0f} "
                  f"{r['pct_TPS']:>5.1f}% {r['pct_P3D']:>5.1f}% "
                  f"{r['Dmean_mGy']:>7.0f} {r['Dmax_mGy']:>7.0f} "
                  f"{r['Dmedian_mGy']:>7.0f} {r['D2pct_mGy']:>7.0f} {r['D98pct_mGy']:>7.0f}")
        print("\nAll doses in mGy.")
    return df


# ============================================================
# CLI
# ============================================================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-patient hybrid dose pipeline (P3D + TPS RTDOSE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wb-ct-dir", type=Path, required=True,
                   help="Directory with the WB voxel model CT DICOM slices")
    p.add_argument("--rtdose", type=Path, required=True,
                   help="Source RTDOSE DICOM file (the planning dose)")
    p.add_argument("--source-iso-mm", type=float, nargs=3, required=True,
                   metavar=("X", "Y", "Z"),
                   help="Isocenter of the source plan in DICOM LPS mm "
                        "(typically taken from the source RTPLAN)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--target-iso-mm", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="Target isocenter on the WB CT in DICOM LPS mm")
    g.add_argument("--target-iso-ijk", type=int, nargs=3, metavar=("SLICE", "ROW", "COL"),
                   help="Target isocenter on the WB CT as voxel indices (slice row col)")
    p.add_argument("--rx-gy", type=float, required=True,
                   help="Total prescribed dose at isocenter in Gy")
    p.add_argument("--n-fractions", type=int, required=True,
                   help="Number of fractions")
    p.add_argument("--total-mu", type=float, required=True,
                   help="Total monitor units across all fractions")
    p.add_argument("--field-area-cm2", type=float, required=True,
                   help="Equivalent field area at isocenter (FU)")
    p.add_argument("--energy-mv", type=float, default=6.0,
                   help="Beam energy in MV (documentation only)")
    p.add_argument("--leakage-mgy-per-mu", type=float, default=REF_LEAKAGE_MGY_PER_MU,
                   help="Linac head leakage Lu in mGy/MU")
    seg = p.add_mutually_exclusive_group(required=False)
    seg.add_argument("--seg-nifti-dir", type=Path,
                     help="Directory of binary NIfTI masks (one organ per file). "
                          "Typical for cohort runs with TotalSegmentator output.")
    seg.add_argument("--rtstruct", type=Path,
                     help="RTSTRUCT DICOM file (alternative to --seg-nifti-dir; "
                          "useful for verifying against the original single-patient pipeline)")
    p.add_argument("--seg-coord-system", choices=["ras", "lps"], default="ras",
                   help="Coordinate convention of the NIfTI affine (TotalSegmentator is RAS)")
    p.add_argument("--seg-include-regex", default=None,
                   help="Regex (case-insensitive) to filter NIfTI organ names; only matching are kept")
    p.add_argument("--seg-exclude-regex", default=None,
                   help="Regex (case-insensitive) to exclude NIfTI organ names; applied first")
    p.add_argument("--no-dvh", action="store_true", help="Skip per-organ DVH CSV/PNG output")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    source_iso = np.array(args.source_iso_mm, dtype=np.float64)

    if args.target_iso_mm is not None:
        target_iso = np.array(args.target_iso_mm, dtype=np.float64)
    else:
        # Convert (slice, row, col) -> (x, y, z) in LPS mm using the WB CT grid
        ct = load_ct_volume(args.wb_ct_dir)
        s, r, c = args.target_iso_ijk
        if not (0 <= s < len(ct["z"]) and 0 <= r < len(ct["y"]) and 0 <= c < len(ct["x"])):
            print("ERROR: --target-iso-ijk out of bounds", file=sys.stderr)
            return 2
        target_iso = np.array([ct["x"][c], ct["y"][r], ct["z"][s]], dtype=np.float64)
        print(f"Resolved target ijk -> LPS mm: {target_iso.tolist()}")

    run(
        wb_ct_dir=args.wb_ct_dir,
        rtdose_path=args.rtdose,
        source_iso_mm=source_iso,
        target_iso_mm=target_iso,
        rx_gy=float(args.rx_gy),
        n_fractions=int(args.n_fractions),
        total_mu=float(args.total_mu),
        field_area_cm2=float(args.field_area_cm2),
        energy_mv=float(args.energy_mv),
        leakage_mgy_per_mu=float(args.leakage_mgy_per_mu),
        seg_nifti_dir=args.seg_nifti_dir,
        rtstruct_path=args.rtstruct,
        seg_coord_system=args.seg_coord_system,
        outdir=args.outdir,
        write_dvhs=not args.no_dvh,
        seg_include_regex=args.seg_include_regex,
        seg_exclude_regex=args.seg_exclude_regex,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
