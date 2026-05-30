#!/usr/bin/env python3
"""
Periphocal 3D out-of-field dosimetry from CT DICOM + NIfTI segmentations,
with optional CT DICOM -> NIfTI export and optional 3D full-body dose-cube export in NIfTI and RTDOSE DICOM.

Implements the 2022 Periphocal 3D analytical model and produces per-organ
DVHs, summary dose statistics, and optionally a 3D dose volume on the CT grid.

Supported segmentation inputs
-----------------------------
1) Directory of binary NIfTI masks (one organ per .nii / .nii.gz)
2) Single NIfTI file that is either:
   - a binary mask, or
   - an integer labelmap (optionally with a JSON label->name mapping)

Coordinate systems
------------------
- CT DICOM is read in DICOM patient coordinates (LPS, mm).
- NIfTI affine is usually RAS+, so the default assumes the segmentation NIfTI
  affine is RAS. Use --seg-coord-system lps if your NIfTI affine is already in
  LPS patient coordinates.

Important validity note
-----------------------
Periphocal 3D is intended for out-of-field dose estimation outside the 5%
'isodose surface' for isocentric coplanar photon treatments. It should not be
used as a replacement for TPS dose in-field or near the treated volume.

Dependencies
------------
pip install pydicom nibabel scipy pandas matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass,field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, RTDoseStorage, generate_uid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.ndimage import affine_transform

try:
    import pydicom
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pydicom. Install with `pip install pydicom`.") from exc

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: nibabel. Install with `pip install nibabel`.") from exc


# Periphocal 3D constants from Sánchez-Nieto et al. (2022)
A1_MGY_CM2_PER_MU = 37.890
A2_MGY_CM_PER_MU = 0.679
A3_PER_CM = 0.007
REFERENCE_FIELD_AREA_CM2 = 149.2
REFERENCE_EU_MGY_PER_MU = 7.2
REFERENCE_LEAKAGE_MGY_PER_MU = 0.001

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass
class CtSeries:
    volume_hu: np.ndarray  # [rows, cols, slices]
    ipp0_mm: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal_dir: np.ndarray
    row_spacing_mm: float
    col_spacing_mm: float
    slice_spacing_mm: float
    study_instance_uid: str
    frame_of_reference_uid: Optional[str]
    source_series_instance_uid: str
    sop_instance_uids: List[str]
    patient_name: str
    patient_id: str
    patient_birth_date: str
    patient_sex: str
    study_date: str
    study_time: str
    accession_number: str
    referring_physician_name: str
    study_id: str

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.volume_hu.shape)

    @property
    def affine_lps(self) -> np.ndarray:
        """Map CT voxel indices [row, col, slice, 1] to DICOM patient LPS mm."""
        aff = np.eye(4, dtype=np.float64)
        aff[:3, 0] = self.row_dir * self.row_spacing_mm
        aff[:3, 1] = self.col_dir * self.col_spacing_mm
        aff[:3, 2] = self.normal_dir * self.slice_spacing_mm
        aff[:3, 3] = self.ipp0_mm
        return aff

    @property
    def inverse_affine_lps(self) -> np.ndarray:
        return np.linalg.inv(self.affine_lps)

    @property
    def voxel_volume_cc(self) -> float:
        return (self.row_spacing_mm * self.col_spacing_mm * self.slice_spacing_mm) / 1000.0

    def index_to_patient_mm(self, indices: np.ndarray) -> np.ndarray:
        arr = np.asarray(indices, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("indices must have shape [N,3] as [row,col,slice]")
        homo = np.c_[arr, np.ones((arr.shape[0], 1), dtype=np.float64)]
        pts = (self.affine_lps @ homo.T).T[:, :3]
        return pts


@dataclass
class Roi:
    number: int
    name: str
    roi_type: str
    flat_indices_c: np.ndarray


@dataclass
class ModelInputs:
    field_area_cm2: float
    total_mu: float
    eu_mgy_per_mu: float
    leakage_mgy_per_mu: float

    @property
    def epsilon(self) -> float:
        return self.eu_mgy_per_mu / REFERENCE_EU_MGY_PER_MU

    @property
    def field_factor(self) -> float:
        return self.field_area_cm2 / REFERENCE_FIELD_AREA_CM2


@dataclass
class SegmentationResult:
    rois: List[Roi]
    source_description: str
    diagnostics: List[str] = field(default_factory=list)


def _collect_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file()])


def _nifti_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in _collect_files(root):
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            out.append(p)
    return out


def _stem_nii(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def _collect_dicom_ct_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in _collect_files(root):
        if p.suffix.lower() in {".dcm", ".dicom", ""}:
            out.append(p)
    return out


def load_ct_series(ct_dir: Path) -> CtSeries:
    slices = []
    for p in _collect_dicom_ct_files(ct_dir):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=False, force=True)
        except Exception:
            continue
        if getattr(ds, "Modality", None) != "CT":
            continue
        if not hasattr(ds, "PixelData"):
            continue
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
            continue
        slices.append(ds)

    if not slices:
        raise RuntimeError(f"No CT DICOM slices found under {ct_dir}")

    orient = np.asarray(slices[0].ImageOrientationPatient, dtype=np.float64)
    row_dir = orient[:3]
    col_dir = orient[3:]
    normal_dir = np.cross(row_dir, col_dir)
    normal_dir /= np.linalg.norm(normal_dir)

    ipp = np.array([np.asarray(ds.ImagePositionPatient, dtype=np.float64) for ds in slices])
    locs = ipp @ normal_dir
    order = np.argsort(locs)
    slices = [slices[i] for i in order]
    locs = locs[order]

    rows = int(slices[0].Rows)
    cols = int(slices[0].Columns)
    row_spacing_mm, col_spacing_mm = map(float, slices[0].PixelSpacing)
    if len(locs) > 1:
        slice_spacing_mm = float(np.median(np.abs(np.diff(locs))))
    else:
        slice_spacing_mm = float(getattr(slices[0], "SliceThickness", 1.0))

    vol = np.empty((rows, cols, len(slices)), dtype=np.int16)
    for k, ds in enumerate(slices):
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        vol[:, :, k] = np.rint(arr).astype(np.int16)

    first = slices[0]
    return CtSeries(
        volume_hu=vol,
        ipp0_mm=np.asarray(first.ImagePositionPatient, dtype=np.float64),
        row_dir=row_dir,
        col_dir=col_dir,
        normal_dir=normal_dir,
        row_spacing_mm=row_spacing_mm,
        col_spacing_mm=col_spacing_mm,
        slice_spacing_mm=slice_spacing_mm,
        study_instance_uid=str(getattr(first, "StudyInstanceUID", generate_uid())),
        frame_of_reference_uid=getattr(first, "FrameOfReferenceUID", None),
        source_series_instance_uid=str(getattr(first, "SeriesInstanceUID", generate_uid())),
        sop_instance_uids=[str(getattr(ds, "SOPInstanceUID", generate_uid())) for ds in slices],
        patient_name=str(getattr(first, "PatientName", "")),
        patient_id=str(getattr(first, "PatientID", "")),
        patient_birth_date=str(getattr(first, "PatientBirthDate", "")),
        patient_sex=str(getattr(first, "PatientSex", "")),
        study_date=str(getattr(first, "StudyDate", "")),
        study_time=str(getattr(first, "StudyTime", "")),
        accession_number=str(getattr(first, "AccessionNumber", "")),
        referring_physician_name=str(getattr(first, "ReferringPhysicianName", "")),
        study_id=str(getattr(first, "StudyID", "")),
    )


def ct_to_nifti_image(ct: CtSeries) -> nib.Nifti1Image:
    """Create a NIfTI image from the loaded DICOM CT series.

    The stored NIfTI affine is RAS+, which is the usual NIfTI world convention.
    Voxel array order is kept as [row, col, slice]; the affine encodes that layout.
    """
    affine_ras = LPS_TO_RAS @ ct.affine_lps
    img = nib.Nifti1Image(ct.volume_hu.astype(np.int16, copy=False), affine_ras)
    img.set_qform(affine_ras, code=1)
    img.set_sform(affine_ras, code=1)
    img.header.set_xyzt_units(xyz="mm")
    img.header["descrip"] = b"CT converted from DICOM by periphocal3d script"
    return img


def save_ct_as_nifti(ct: CtSeries, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(ct_to_nifti_image(ct), str(out_path))
    return out_path


def resample_nifti_to_ct_mask(
    nifti_path: Path,
    ct: CtSeries,
    seg_coord_system: str,
    label_value: Optional[int] = None,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI for {nifti_path}, got shape {data.shape}")

    header = img.header
    qform_code = int(np.asarray(header.get("qform_code", 0)).item())
    sform_code = int(np.asarray(header.get("sform_code", 0)).item())
    affine = np.asarray(img.affine, dtype=np.float64)

    # Some exported masks have no usable NIfTI world geometry at all
    # (qform_code=sform_code=0, identity-like affine, pixdim=1, same matrix shape as CT).
    # In that case, physical-space resampling is ill-posed and can silently erase ROIs.
    # Fallback: if qform and sform are both 0, treat as index-space mask on CT grid.
    # If the spatial dimensions match (rows, cols) but slice count differs, pad/crop
    # the slice axis to match the CT — this handles cases where the NIfTI was built
    # against a different CT series length than the one currently loaded.
    if qform_code == 0 and sform_code == 0:
        if tuple(data.shape) == tuple(ct.shape):
            # Perfect match — use directly
            if label_value is None:
                return np.asarray(data > float(binary_threshold), dtype=bool)
            return np.asarray(data == int(label_value), dtype=bool)
        if data.shape[0] == ct.shape[0] and data.shape[1] == ct.shape[1]:
            # Rows and cols match; pad or crop the slice axis
            ct_slices = ct.shape[2]
            nii_slices = data.shape[2]
            if nii_slices >= ct_slices:
                data_adj = data[:, :, :ct_slices]
            else:
                pad = np.zeros((data.shape[0], data.shape[1], ct_slices - nii_slices),
                               dtype=data.dtype)
                data_adj = np.concatenate([data, pad], axis=2)
            if label_value is None:
                return np.asarray(data_adj > float(binary_threshold), dtype=bool)
            return np.asarray(data_adj == int(label_value), dtype=bool)

    world_from_ct = ct.affine_lps
    if seg_coord_system.lower() == "ras":
        world_from_ct = LPS_TO_RAS @ world_from_ct
    elif seg_coord_system.lower() != "lps":
        raise ValueError("seg_coord_system must be 'ras' or 'lps'")

    seg_vox_from_world = np.linalg.inv(affine)
    seg_vox_from_ct = seg_vox_from_world @ world_from_ct

    matrix = seg_vox_from_ct[:3, :3]
    offset = seg_vox_from_ct[:3, 3]

    if label_value is None:
        src = data.astype(np.float32, copy=False)
        out = affine_transform(
            src,
            matrix=matrix,
            offset=offset,
            output_shape=ct.shape,
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return out > float(binary_threshold)

    src = (data == int(label_value)).astype(np.float32, copy=False)
    out = affine_transform(
        src,
        matrix=matrix,
        offset=offset,
        output_shape=ct.shape,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return out > 0.5


def load_labelmap_name_map(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    if isinstance(payload, dict):
        for k, v in payload.items():
            out[int(k)] = str(v)
        return out
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict) or "label" not in item or "name" not in item:
                raise ValueError("Labelmap JSON list items must contain 'label' and 'name'")
            out[int(item["label"])] = str(item["name"])
        return out
    raise ValueError("Labelmap JSON must be an object or list of objects")


def segmentation_from_nifti_dir(
    seg_dir: Path,
    ct: CtSeries,
    seg_coord_system: str,
    roi_regex: Optional[str],
    min_voxels: int,
) -> SegmentationResult:
    regex = re.compile(roi_regex, re.IGNORECASE) if roi_regex else None
    rois: List[Roi] = []
    diagnostics: List[str] = []
    files = _nifti_files(seg_dir)
    if not files:
        raise RuntimeError(f"No NIfTI files found under {seg_dir}")

    roi_num = 1
    for p in files:
        name = _stem_nii(p)
        if regex and not regex.search(name):
            diagnostics.append(f"Skipped ROI: {name} (does not match ROI regex)")
            continue
        mask = resample_nifti_to_ct_mask(p, ct, seg_coord_system=seg_coord_system)
        flat = np.flatnonzero(mask)
        if flat.size < int(min_voxels):
            diagnostics.append(
                f"Skipped ROI: {name} ({flat.size} voxels after resampling; min_voxels={int(min_voxels)})"
            )
            continue
        rois.append(Roi(number=roi_num, name=name, roi_type="OAR", flat_indices_c=flat))
        diagnostics.append(f"Loaded ROI: {name} ({flat.size} voxels on CT grid)")
        roi_num += 1

    return SegmentationResult(
        rois=rois,
        source_description=f"binary NIfTI directory: {seg_dir}",
        diagnostics=diagnostics,
    )


def segmentation_from_single_nifti(
    seg_path: Path,
    ct: CtSeries,
    seg_coord_system: str,
    roi_regex: Optional[str],
    min_voxels: int,
    labelmap_json: Optional[Path],
) -> SegmentationResult:
    regex = re.compile(roi_regex, re.IGNORECASE) if roi_regex else None
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI for {seg_path}, got shape {data.shape}")

    rois: List[Roi] = []
    diagnostics: List[str] = []
    nonzero = np.unique(data[np.isfinite(data)])
    nonzero = nonzero[nonzero != 0]

    # Binary mask case
    if nonzero.size <= 2 and np.all(np.isin(nonzero, [1, 1.0])):
        name = _stem_nii(seg_path)
        if not regex or regex.search(name):
            mask = resample_nifti_to_ct_mask(seg_path, ct, seg_coord_system=seg_coord_system)
            flat = np.flatnonzero(mask)
            if flat.size >= int(min_voxels):
                rois.append(Roi(number=1, name=name, roi_type="OAR", flat_indices_c=flat))
        diagnostics = []
        if rois:
            diagnostics.append(f"Loaded ROI: {name} ({flat.size} voxels on CT grid)")
        else:
            diagnostics.append(f"Skipped ROI: {name} ({flat.size if 'flat' in locals() else 0} voxels after resampling; min_voxels={int(min_voxels)})")
        return SegmentationResult(rois=rois, source_description=f"binary NIfTI file: {seg_path}", diagnostics=diagnostics)

    name_map = load_labelmap_name_map(labelmap_json)
    labels = sorted(int(v) for v in np.unique(data) if int(v) != 0)
    roi_num = 1
    for label in labels:
        name = name_map.get(label, f"label_{label}")
        if regex and not regex.search(name):
            diagnostics.append(f"Skipped ROI: {name} (does not match ROI regex)")
            continue
        mask = resample_nifti_to_ct_mask(
            seg_path,
            ct,
            seg_coord_system=seg_coord_system,
            label_value=label,
        )
        flat = np.flatnonzero(mask)
        if flat.size < int(min_voxels):
            continue
        rois.append(Roi(number=roi_num, name=name, roi_type="OAR", flat_indices_c=flat))
        roi_num += 1

    return SegmentationResult(rois=rois, source_description=f"labelmap NIfTI file: {seg_path}")


def derive_eu_mgy_per_mu(rx_gy: Optional[float], total_mu: float, eu_mgy_per_mu: Optional[float]) -> float:
    if eu_mgy_per_mu is not None:
        return float(eu_mgy_per_mu)
    if rx_gy is None:
        raise ValueError("Provide either --eu-mgy-per-mu or both --rx-gy and --total-mu")
    return float(rx_gy) * 1000.0 / float(total_mu)


def get_isocenter_mm(
    ct: CtSeries,
    isocenter_mm: Optional[Sequence[float]],
    isocenter_ijk: Optional[Sequence[int]],
) -> np.ndarray:
    if isocenter_mm is not None and isocenter_ijk is not None:
        raise ValueError("Use either --isocenter-mm or --isocenter-ijk, not both")
    if isocenter_mm is not None:
        arr = np.asarray(isocenter_mm, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError("--isocenter-mm needs 3 values: x y z in DICOM patient mm")
        return arr
    if isocenter_ijk is not None:
        arr = np.asarray(isocenter_ijk, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError("--isocenter-ijk needs 3 values: row col slice")
        return ct.index_to_patient_mm(arr.reshape(1, 3))[0]
    raise ValueError("Provide either --isocenter-mm or --isocenter-ijk")


def periphocal3d_ppd_mgy_per_mu(
    dx_cm: np.ndarray,
    dy_cm: np.ndarray,
    dz_cm: np.ndarray,
    model: ModelInputs,
) -> np.ndarray:
    r_cm = np.sqrt(dx_cm * dx_cm + dy_cm * dy_cm + dz_cm * dz_cm)
    safe_r = np.maximum(r_cm, 1e-6)
    scatter_term = (
        model.epsilon
        * model.field_factor
        * (A1_MGY_CM2_PER_MU - A2_MGY_CM_PER_MU * np.abs(dz_cm))
        * np.exp(-A3_PER_CM * safe_r)
        / (safe_r * safe_r)
    )
    dose = np.where(
        r_cm <= 40.0,
        scatter_term + (model.leakage_mgy_per_mu - REFERENCE_LEAKAGE_MGY_PER_MU),
        model.leakage_mgy_per_mu,
    )
    dose = np.maximum(dose, 0.0)
    dose[r_cm < 1e-6] = np.nan
    return dose


def evaluate_roi_dose_values_gy(
    ct: CtSeries,
    roi: Roi,
    isocenter_mm_lps: np.ndarray,
    model: ModelInputs,
) -> np.ndarray:
    rows, cols, slcs = np.unravel_index(roi.flat_indices_c, ct.shape, order="C")
    points_mm = ct.index_to_patient_mm(np.c_[rows, cols, slcs])
    delta_cm = (points_mm - isocenter_mm_lps[None, :]) / 10.0
    ppd_mgy_per_mu = periphocal3d_ppd_mgy_per_mu(
        dx_cm=delta_cm[:, 0],
        dy_cm=delta_cm[:, 1],
        dz_cm=delta_cm[:, 2],
        model=model,
    )
    return ppd_mgy_per_mu * model.total_mu / 1000.0


def cumulative_dvh(dose_values_gy: np.ndarray, nbins: int = 200) -> pd.DataFrame:
    vals = np.asarray(dose_values_gy, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return pd.DataFrame({"dose_Gy": [0.0], "volume_fraction": [0.0], "volume_percent": [0.0]})
    dmax = float(np.max(vals))
    if dmax <= 0.0:
        return pd.DataFrame({"dose_Gy": [0.0], "volume_fraction": [1.0], "volume_percent": [100.0]})
    bins = np.linspace(0.0, dmax, nbins)
    counts = np.array([(vals >= b).sum() for b in bins], dtype=np.float64)
    frac = counts / vals.size
    return pd.DataFrame({"dose_Gy": bins, "volume_fraction": frac, "volume_percent": frac * 100.0})


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "ROI"


def write_dvh_plot(dvh_df: pd.DataFrame, roi_name: str, out_png: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(dvh_df["dose_Gy"], dvh_df["volume_percent"])
    plt.xlabel("Dose (Gy)")
    plt.ylabel("Volume >= dose (%)")
    plt.title(roi_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def export_legacy_mat(out_path: Path, ct: CtSeries, rois: Sequence[Roi]) -> None:
    cst = np.empty((len(rois), 4), dtype=object)
    labelmap = np.zeros(ct.shape, dtype=np.int16)
    for i, roi in enumerate(rois, start=1):
        rr, cc, kk = np.unravel_index(roi.flat_indices_c, ct.shape, order="C")
        labelmap[rr, cc, kk] = i
        matlab_indices = np.ravel_multi_index((rr, cc, kk), ct.shape, order="F") + 1
        cst[i - 1, 0] = i
        cst[i - 1, 1] = roi.name
        cst[i - 1, 2] = roi.roi_type or "OAR"
        cst[i - 1, 3] = matlab_indices.astype(np.int64)

    cube_hu = np.empty((1, 1), dtype=object)
    cube = np.empty((1, 1), dtype=object)
    num_org = np.empty((1, 1), dtype=object)
    cube_hu[0, 0] = ct.volume_hu
    cube[0, 0] = ct.volume_hu
    num_org[0, 0] = labelmap
    ct_struct = {
        "cubeHU": cube_hu,
        "cube": cube,
        "numOrg": num_org,
        "resolution": {
            "x": float(ct.row_spacing_mm),
            "y": float(ct.col_spacing_mm),
            "z": float(ct.slice_spacing_mm),
        },
        "cubeDim": np.asarray(ct.shape, dtype=np.int32),
    }
    savemat(str(out_path), {"ct": ct_struct, "cst": cst}, do_compression=True)


def dose_cube_nifti_image(ct: CtSeries, dose_cube_gy: np.ndarray) -> nib.Nifti1Image:
    affine_ras = LPS_TO_RAS @ ct.affine_lps
    img = nib.Nifti1Image(np.asarray(dose_cube_gy, dtype=np.float32), affine_ras)
    img.set_qform(affine_ras, code=1)
    img.set_sform(affine_ras, code=1)
    img.header.set_xyzt_units(xyz="mm")
    img.header["descrip"] = b"Periphocal3D full-body dose cube in Gy"
    return img


def compute_full_body_dose_cube_gy(
    ct: CtSeries,
    isocenter_mm_lps: np.ndarray,
    model: ModelInputs,
    out_path: Optional[Path] = None,
    chunk_slices: int = 8,
    body_hu_threshold: Optional[float] = None,
) -> np.ndarray:
    """Compute the full 3D dose cube on the CT grid.

    The computation is chunked along z to keep memory bounded. If out_path is
    provided, a float32 memmap is used so large datasets do not need a second
    full in-memory copy before being written to NIfTI.
    """
    rows, cols, slices = ct.shape
    chunk_slices = max(1, int(chunk_slices))

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dose_cube = np.lib.format.open_memmap(
            str(out_path),
            mode="w+",
            dtype=np.float32,
            shape=ct.shape,
        )
    else:
        dose_cube = np.empty(ct.shape, dtype=np.float32)

    row_idx = np.arange(rows, dtype=np.float64)
    col_idx = np.arange(cols, dtype=np.float64)
    rr, cc = np.meshgrid(row_idx, col_idx, indexing="ij")

    aff = ct.affine_lps
    row_vec = aff[:3, 0]
    col_vec = aff[:3, 1]
    slc_vec = aff[:3, 2]
    origin = aff[:3, 3]

    base_x = origin[0] + rr * row_vec[0] + cc * col_vec[0]
    base_y = origin[1] + rr * row_vec[1] + cc * col_vec[1]
    base_z = origin[2] + rr * row_vec[2] + cc * col_vec[2]

    iso = np.asarray(isocenter_mm_lps, dtype=np.float64)

    for k0 in range(0, slices, chunk_slices):
        k1 = min(slices, k0 + chunk_slices)
        ks = np.arange(k0, k1, dtype=np.float64)

        x_mm = base_x[:, :, None] + slc_vec[0] * ks[None, None, :]
        y_mm = base_y[:, :, None] + slc_vec[1] * ks[None, None, :]
        z_mm = base_z[:, :, None] + slc_vec[2] * ks[None, None, :]

        ppd_mgy_per_mu = periphocal3d_ppd_mgy_per_mu(
            dx_cm=(x_mm - iso[0]) / 10.0,
            dy_cm=(y_mm - iso[1]) / 10.0,
            dz_cm=(z_mm - iso[2]) / 10.0,
            model=model,
        )
        dose_chunk = (ppd_mgy_per_mu * model.total_mu / 1000.0).astype(np.float32, copy=False)

        if body_hu_threshold is not None:
            body_mask = ct.volume_hu[:, :, k0:k1] > float(body_hu_threshold)
            dose_chunk = np.where(body_mask, dose_chunk, 0.0).astype(np.float32, copy=False)

        dose_cube[:, :, k0:k1] = dose_chunk

    return dose_cube


def save_full_body_dose_cube(
    ct: CtSeries,
    isocenter_mm_lps: np.ndarray,
    model: ModelInputs,
    nifti_path: Optional[Path],
    dicom_path: Optional[Path],
    npy_path: Optional[Path],
    chunk_slices: int,
    body_hu_threshold: Optional[float],
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if nifti_path is not None:
        nifti_path = Path(nifti_path)
        nifti_path.parent.mkdir(parents=True, exist_ok=True)
    if dicom_path is not None:
        dicom_path = Path(dicom_path)
        dicom_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="p3d_dosecube_") as tmpdir:
        memmap_path = Path(tmpdir) / "dose_cube.npy"
        dose_cube = compute_full_body_dose_cube_gy(
            ct=ct,
            isocenter_mm_lps=isocenter_mm_lps,
            model=model,
            out_path=memmap_path,
            chunk_slices=chunk_slices,
            body_hu_threshold=body_hu_threshold,
        )
        if nifti_path is not None:
            nib.save(dose_cube_nifti_image(ct, dose_cube), str(nifti_path))
        if dicom_path is not None:
            save_dose_cube_as_rtdose_dicom(ct, np.asarray(dose_cube, dtype=np.float32), dicom_path)

        if npy_path is not None:
            npy_path = Path(npy_path)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(npy_path), np.asarray(dose_cube, dtype=np.float32))

    return nifti_path, dicom_path, npy_path

def _safe_dicom_str(value: str) -> str:
    return str(value) if value is not None else ""


def _copy_if_present(ds: Dataset, keyword: str, value: str) -> None:
    if value not in (None, ""):
        setattr(ds, keyword, value)


def save_dose_cube_as_rtdose_dicom(ct: CtSeries, dose_cube_gy: np.ndarray, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = "P3D_PY_1"

    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.SOPClassUID = RTDoseStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTDOSE"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = ct.study_instance_uid
    if ct.frame_of_reference_uid:
        ds.FrameOfReferenceUID = ct.frame_of_reference_uid
    ds.SeriesNumber = 9001
    ds.InstanceNumber = 1

    _copy_if_present(ds, "PatientName", ct.patient_name)
    _copy_if_present(ds, "PatientID", ct.patient_id)
    _copy_if_present(ds, "PatientBirthDate", ct.patient_birth_date)
    _copy_if_present(ds, "PatientSex", ct.patient_sex)
    _copy_if_present(ds, "StudyDate", ct.study_date)
    _copy_if_present(ds, "StudyTime", ct.study_time)
    _copy_if_present(ds, "AccessionNumber", ct.accession_number)
    _copy_if_present(ds, "ReferringPhysicianName", ct.referring_physician_name)
    _copy_if_present(ds, "StudyID", ct.study_id)

    ds.Manufacturer = "OpenAI"
    ds.ManufacturerModelName = "Periphocal3D Python"
    ds.SoftwareVersions = "1.0"
    ds.SeriesDescription = "Periphocal3D full-body out-of-field dose"

    rows, cols, nframes = ct.shape
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.NumberOfFrames = int(nframes)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31

    ds.DoseUnits = "GY"
    ds.DoseType = "PHYSICAL"
    ds.DoseSummationType = "PLAN"

    ds.ImageOrientationPatient = [float(v) for v in np.r_[ct.row_dir, ct.col_dir]]
    ds.ImagePositionPatient = [float(v) for v in ct.ipp0_mm]
    ds.PixelSpacing = [float(ct.row_spacing_mm), float(ct.col_spacing_mm)]
    ds.FrameIncrementPointer = [Tag(0x3004, 0x000C)]
    ds.GridFrameOffsetVector = [float(k * ct.slice_spacing_mm) for k in range(nframes)]

    if ct.frame_of_reference_uid and ct.sop_instance_uids:
        ref_for = Dataset()
        ref_for.FrameOfReferenceUID = ct.frame_of_reference_uid
        rt_ref_study = Dataset()
        rt_ref_series = Dataset()
        rt_ref_series.SeriesInstanceUID = ct.source_series_instance_uid
        contour_items = []
        for sop_uid in ct.sop_instance_uids:
            item = Dataset()
            item.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
            item.ReferencedSOPInstanceUID = sop_uid
            contour_items.append(item)
        rt_ref_series.ContourImageSequence = DicomSequence(contour_items)
        rt_ref_study.RTReferencedSeriesSequence = DicomSequence([rt_ref_series])
        ref_for.RTReferencedStudySequence = DicomSequence([rt_ref_study])
        ds.ReferencedFrameOfReferenceSequence = DicomSequence([ref_for])

    dose = np.asarray(dose_cube_gy, dtype=np.float64)
    finite = dose[np.isfinite(dose)]
    max_dose = float(np.max(finite)) if finite.size else 0.0
    if max_dose <= 0.0:
        dose_grid_scaling = 1.0
        stored = np.zeros((nframes, rows, cols), dtype=np.uint32)
    else:
        dose_grid_scaling = max(max_dose / (np.iinfo(np.uint32).max - 1), 1e-8)
        dose_clean = np.nan_to_num(dose, nan=0.0, posinf=max_dose, neginf=0.0)
        scaled = np.rint(np.maximum(dose_clean, 0.0) / dose_grid_scaling).astype(np.uint32)
        stored = np.transpose(scaled, (2, 0, 1))
    ds.DoseGridScaling = float(dose_grid_scaling)
    ds.PixelData = np.ascontiguousarray(stored).tobytes()

    ds.save_as(str(out_path), write_like_original=False)
    return out_path



def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Periphocal 3D out-of-field dose + organ DVH calculator for CT DICOM + NIfTI segmentations, with optional CT DICOM->NIfTI export and full-body dose cube export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ct-dir", type=Path, required=True, help="Directory containing CT DICOM slices")
    parser.add_argument(
        "--save-converted-ct-nifti",
        type=Path,
        default=None,
        help="Optional path to save the loaded DICOM CT as a NIfTI (.nii or .nii.gz) while continuing the dose calculation",
    )

    seg_group = parser.add_mutually_exclusive_group(required=False)
    seg_group.add_argument("--seg-nifti-dir", type=Path, help="Directory with one binary NIfTI mask per organ")
    seg_group.add_argument("--seg-nifti", type=Path, help="Single binary-mask or labelmap NIfTI")

    parser.add_argument(
        "--seg-labelmap-json",
        type=Path,
        default=None,
        help="Optional JSON mapping for a labelmap NIfTI, e.g. {'1':'heart','2':'liver'}",
    )
    parser.add_argument(
        "--seg-coord-system",
        choices=["ras", "lps"],
        default="ras",
        help="World coordinate convention of the NIfTI affine",
    )
    parser.add_argument(
        "--isocenter-mm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Isocenter in DICOM patient coordinates (LPS mm)",
    )
    parser.add_argument(
        "--isocenter-ijk",
        nargs=3,
        type=int,
        metavar=("ROW", "COL", "SLICE"),
        help="Isocenter as 0-based CT voxel indices",
    )
    parser.add_argument("--field-area-cm2", type=float, required=True, help="FU field area at isocenter in cm^2")
    parser.add_argument("--total-mu", type=float, required=True, help="Total MU on the same basis as dose input")
    parser.add_argument("--rx-gy", type=float, default=None, help="Dose at isocenter in Gy for the same basis")
    parser.add_argument("--eu-mgy-per-mu", type=float, default=None, help="EU directly in mGy/MU")
    parser.add_argument(
        "--leakage-mgy-per-mu",
        type=float,
        default=REFERENCE_LEAKAGE_MGY_PER_MU,
        help="Lu leakage in mGy/MU",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--roi-regex", type=str, default=None, help="Only keep organs matching this regex")
    parser.add_argument("--min-voxels", type=int, default=5, help="Discard tiny masks after resampling")
    parser.add_argument("--dvh-bins", type=int, default=200, help="Number of bins for cumulative DVH")
    parser.add_argument("--export-legacy-mat", action="store_true", help="Also save legacy phantom_p3d_python.mat")
    parser.add_argument("--list-rois", action="store_true", help="Print detected ROIs and exit")
    parser.add_argument(
        "--export-full-dose-cube",
        action="store_true",
        help="Also calculate and save the full 3D dose cube on the CT grid",
    )
    parser.add_argument(
        "--full-dose-cube-nifti",
        type=Path,
        default=None,
        help="Optional output path for the 3D dose cube NIfTI; defaults to <outdir>/full_body_dose_Gy.nii.gz",
    )
    parser.add_argument(
        "--full-dose-cube-dicom",
        type=Path,
        default=None,
        help="Optional output path for the 3D dose cube as an RTDOSE DICOM file; defaults to <outdir>/full_body_dose_Gy.dcm when --export-full-dose-cube is used",
    )
    parser.add_argument(
        "--full-dose-cube-npy",
        type=Path,
        default=None,
        help="Optional output path for the 3D dose cube as a .npy array",
    )
    parser.add_argument(
        "--dose-cube-chunk-slices",
        type=int,
        default=8,
        help="Number of CT slices per chunk when building the full dose cube",
    )
    parser.add_argument(
        "--dose-cube-body-threshold-hu",
        type=float,
        default=None,
        help="Optional HU threshold. Voxels <= threshold are forced to 0 Gy in the exported dose cube",
    )

    args = parser.parse_args(argv)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ct = load_ct_series(args.ct_dir)

    saved_ct_nifti: Optional[Path] = None
    if args.save_converted_ct_nifti is not None:
        saved_ct_nifti = save_ct_as_nifti(ct, args.save_converted_ct_nifti)
        print(f"Wrote converted CT NIfTI: {saved_ct_nifti}")


    if args.seg_nifti_dir is None and args.seg_nifti is None:
        raise SystemExit("Provide either --seg-nifti-dir or --seg-nifti for dose/DVH calculation")

    if args.seg_nifti_dir is not None:
        seg = segmentation_from_nifti_dir(
            args.seg_nifti_dir,
            ct,
            seg_coord_system=args.seg_coord_system,
            roi_regex=args.roi_regex,
            min_voxels=args.min_voxels,
        )
    else:
        seg = segmentation_from_single_nifti(
            args.seg_nifti,
            ct,
            seg_coord_system=args.seg_coord_system,
            roi_regex=args.roi_regex,
            min_voxels=args.min_voxels,
            labelmap_json=args.seg_labelmap_json,
        )

    for msg in seg.diagnostics:
        print(msg)

    rois = seg.rois
    if not rois:
        raise SystemExit("No ROIs remained after loading / filtering / resampling.")

    if args.list_rois:
        for roi in rois:
            print(f"{roi.number:3d}  {roi.name}  nvox={roi.flat_indices_c.size}")
        return 0

    model = ModelInputs(
        field_area_cm2=float(args.field_area_cm2),
        total_mu=float(args.total_mu),
        eu_mgy_per_mu=derive_eu_mgy_per_mu(args.rx_gy, args.total_mu, args.eu_mgy_per_mu),
        leakage_mgy_per_mu=float(args.leakage_mgy_per_mu),
    )
    iso_mm = get_isocenter_mm(ct, args.isocenter_mm, args.isocenter_ijk)

    summary_rows: List[Dict[str, object]] = []
    for roi in rois:
        dose_gy = evaluate_roi_dose_values_gy(ct, roi, iso_mm, model)
        finite = dose_gy[np.isfinite(dose_gy)]
        if finite.size == 0:
            mean_gy = min_gy = max_gy = math.nan
        else:
            mean_gy = float(np.mean(finite))
            min_gy = float(np.min(finite))
            max_gy = float(np.max(finite))

        dvh = cumulative_dvh(dose_gy, nbins=int(args.dvh_bins))
        base = sanitize_filename(roi.name)
        dvh.to_csv(outdir / f"{base}_DVH.csv", index=False)
        write_dvh_plot(dvh, roi.name, outdir / f"{base}_DVH.png")

        summary_rows.append(
            {
                "roi_number": roi.number,
                "roi_name": roi.name,
                "roi_type": roi.roi_type,
                "n_voxels": int(roi.flat_indices_c.size),
                "voxel_volume_cc": ct.voxel_volume_cc,
                "organ_volume_cc": float(roi.flat_indices_c.size * ct.voxel_volume_cc),
                "dose_mean_Gy": mean_gy,
                "dose_min_Gy": min_gy,
                "dose_max_Gy": max_gy,
                "dvh_csv": f"{base}_DVH.csv",
                "dvh_png": f"{base}_DVH.png",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["roi_number", "roi_name"])
    summary_df.to_csv(outdir / "organ_dose_summary.csv", index=False)

    full_cube_nifti_written: Optional[Path] = None
    full_cube_dicom_written: Optional[Path] = None
    full_cube_npy_written: Optional[Path] = None
    if args.export_full_dose_cube:
        cube_nifti_path = args.full_dose_cube_nifti or (outdir / "full_body_dose_Gy.nii.gz")
        cube_dicom_path = args.full_dose_cube_dicom or (outdir / "full_body_dose_Gy.dcm")
        cube_npy_path = args.full_dose_cube_npy
        full_cube_nifti_written, full_cube_dicom_written, full_cube_npy_written = save_full_body_dose_cube(
            ct=ct,
            isocenter_mm_lps=iso_mm,
            model=model,
            nifti_path=cube_nifti_path,
            dicom_path=cube_dicom_path,
            npy_path=cube_npy_path,
            chunk_slices=int(args.dose_cube_chunk_slices),
            body_hu_threshold=args.dose_cube_body_threshold_hu,
        )
        if full_cube_nifti_written is not None:
            print(f"Wrote full-body dose cube NIfTI: {full_cube_nifti_written}")
        if full_cube_dicom_written is not None:
            print(f"Wrote full-body dose cube RTDOSE DICOM: {full_cube_dicom_written}")
        if full_cube_npy_written is not None:
            print(f"Wrote full-body dose cube NPY: {full_cube_npy_written}")

    run_info = {
        "ct_dir": str(args.ct_dir),
        "converted_ct_nifti": str(saved_ct_nifti) if saved_ct_nifti is not None else None,
        "segmentation_source": seg.source_description,
        "seg_coord_system": args.seg_coord_system,
        "ct_shape": list(ct.shape),
        "ct_spacing_mm": [ct.row_spacing_mm, ct.col_spacing_mm, ct.slice_spacing_mm],
        "isocenter_mm_lps": iso_mm.tolist(),
        "model_inputs": {
            "field_area_cm2": model.field_area_cm2,
            "total_mu": model.total_mu,
            "eu_mgy_per_mu": model.eu_mgy_per_mu,
            "epsilon": model.epsilon,
            "field_factor": model.field_factor,
            "leakage_mgy_per_mu": model.leakage_mgy_per_mu,
        },
        "periphocal3d_constants": {
            "A1_mGy_cm2_per_MU": A1_MGY_CM2_PER_MU,
            "A2_mGy_cm_per_MU": A2_MGY_CM_PER_MU,
            "A3_per_cm": A3_PER_CM,
            "reference_field_area_cm2": REFERENCE_FIELD_AREA_CM2,
            "reference_eu_mGy_per_MU": REFERENCE_EU_MGY_PER_MU,
            "reference_leakage_mGy_per_MU": REFERENCE_LEAKAGE_MGY_PER_MU,
        },
        "n_rois": len(rois),
        "roi_names": [r.name for r in rois],
        "full_body_dose_cube": {
            "exported": bool(args.export_full_dose_cube),
            "nifti_path": str(full_cube_nifti_written) if full_cube_nifti_written is not None else None,
            "dicom_rtdose_path": str(full_cube_dicom_written) if full_cube_dicom_written is not None else None,
            "npy_path": str(full_cube_npy_written) if full_cube_npy_written is not None else None,
            "chunk_slices": int(args.dose_cube_chunk_slices),
            "body_threshold_hu": args.dose_cube_body_threshold_hu,
        },
        "notes": [
            "Model validity is peripheral / out-of-field only.",
            "NIfTI masks are resampled to the CT grid using nearest-neighbor interpolation.",
            "Dose inside or near the treatment field is outside the intended validity of this model.",
            "The exported 3D dose cube is on the CT grid and can be written as NIfTI or RTDOSE DICOM for overlay.",
        ],
    }
    (outdir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    if args.export_legacy_mat:
        export_legacy_mat(outdir / "phantom_p3d_python.mat", ct, rois)

    print(f"Wrote summary: {outdir / 'organ_dose_summary.csv'}")
    print(f"ROIs processed: {len(rois)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
