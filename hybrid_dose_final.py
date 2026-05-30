#!/usr/bin/env python3
"""
Hybrid Dose Map — Periphocal 3D + TPS (RTDOSE)

Combines TPS dose (inside 5% isodose) with P3D analytical dose (outside 5%).
Uses the Sánchez-Nieto et al. (2022) Periphocal 3D model, Eq. 2.

Usage:
    python hybrid_dose_final.py /path/to/dicom/folder /path/to/output/folder

Input:  DICOM directory containing CT, RTDOSE, RTPLAN, RTSTRUCT
Output: Organ dose CSV, figures, summary
"""

import sys
import numpy as np
import pydicom
import os
import glob
from scipy.ndimage import binary_fill_holes
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as MplPath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import json
from pathlib import Path


# ============================================================
# Periphocal 3D — Sánchez-Nieto et al. (2022) Eq. 2
# Verified against original paper
# ============================================================
A1_MGY_CM2_PER_MU = 37.890
A2_MGY_CM_PER_MU  = 0.679
A3_PER_CM          = 0.007
REF_FIELD_AREA_CM2 = 149.2
REF_EU_MGY_PER_MU  = 7.2
REF_LEAKAGE        = 0.001  # mGy/MU


def p3d_ppd_mgy_per_mu(dx_cm, dy_cm, dz_cm, epsilon, field_factor, leakage):
    """
    Periphocal 3D peripheral photon dose (mGy/MU).
    
    Coordinate system (from paper): 
      x = anterior-posterior, y = left-right, z = caudal-cranial
    This matches DICOM LPS, so no coordinate transform needed.
    
    Parameters
    ----------
    dx_cm, dy_cm, dz_cm : displacement from isocenter in cm
    epsilon : EU / EU_ref  (efficiency ratio)
    field_factor : FU / FU_ref  (field area ratio)
    leakage : Lu in mGy/MU
    """
    r = np.sqrt(dx_cm**2 + dy_cm**2 + dz_cm**2)
    safe_r = np.maximum(r, 1e-6)
    
    scatter = (
        epsilon * field_factor
        * (A1_MGY_CM2_PER_MU - A2_MGY_CM_PER_MU * np.abs(dz_cm))
        * np.exp(-A3_PER_CM * safe_r)
        / (safe_r * safe_r)
    )
    
    dose = np.where(
        r <= 40.0,
        scatter + (leakage - REF_LEAKAGE),
        leakage,
    )
    dose = np.maximum(dose, 0.0)
    dose[r < 1e-6] = np.nan
    return dose


# ============================================================
# DICOM loading helpers
# ============================================================
def find_dicom_files(data_dir):
    """Auto-detect CT, RTDOSE, RTPLAN, RTSTRUCT files in a directory."""
    all_dcm = []
    for f in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, f)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            modality = getattr(ds, 'Modality', '').upper()
            all_dcm.append((modality, fpath, ds))
        except:
            continue
    
    ct_files = [(p, ds) for m, p, ds in all_dcm if m == 'CT']
    rtdose_files = [p for m, p, ds in all_dcm if m == 'RTDOSE']
    rtplan_files = [p for m, p, ds in all_dcm if m == 'RTPLAN']
    rtstruct_files = [p for m, p, ds in all_dcm if m == 'RTSTRUCT']
    
    return ct_files, rtdose_files, rtplan_files, rtstruct_files


def load_ct(ct_files):
    """Load CT slices, sorted by z position."""
    slices = []
    for fpath, ds_header in ct_files:
        ds = pydicom.dcmread(fpath)
        z = float(ds.ImagePositionPatient[2])
        slices.append((z, ds))
    slices.sort(key=lambda x: x[0])
    
    ds0 = slices[0][1]
    rows, cols = ds0.Rows, ds0.Columns
    ps = [float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])]
    origin = [float(x) for x in ds0.ImagePositionPatient]
    
    volume = np.zeros((len(slices), rows, cols), dtype=np.float32)
    z_positions = np.zeros(len(slices))
    for i, (z, ds) in enumerate(slices):
        volume[i] = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        z_positions[i] = z
    
    x_coords = origin[0] + np.arange(cols) * ps[1]
    y_coords = origin[1] + np.arange(rows) * ps[0]
    
    return volume, x_coords, y_coords, z_positions, ps, origin


def load_rtdose(rtdose_path):
    """Load RTDOSE and return dose grid + coordinates."""
    ds = pydicom.dcmread(rtdose_path)
    dose_gy = ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling)
    
    origin = [float(x) for x in ds.ImagePositionPatient]
    ps = [float(x) for x in ds.PixelSpacing]
    gfo = np.array([float(x) for x in ds.GridFrameOffsetVector])
    
    x = origin[0] + np.arange(ds.Columns) * ps[1]
    y = origin[1] + np.arange(ds.Rows) * ps[0]
    z = origin[2] + gfo
    
    return dose_gy, x, y, z, ds


def load_rtplan(rtplan_path):
    """Extract treatment parameters from RTPLAN."""
    ds = pydicom.dcmread(rtplan_path)
    
    iso = [float(x) for x in ds.BeamSequence[0].ControlPointSequence[0].IsocenterPosition]
    
    rx_gy = None
    if hasattr(ds, 'DoseReferenceSequence'):
        for dr in ds.DoseReferenceSequence:
            if hasattr(dr, 'TargetPrescriptionDose'):
                rx_gy = float(dr.TargetPrescriptionDose)
                break
    
    n_fx = int(ds.FractionGroupSequence[0].NumberOfFractionsPlanned)
    mu_per_fx = sum(float(rb.BeamMeterset) for rb in ds.FractionGroupSequence[0].ReferencedBeamSequence)
    
    # Beam energy
    energy = float(ds.BeamSequence[0].ControlPointSequence[0].NominalBeamEnergy)
    
    # Jaw-based field area always computed; isodose50 method requires the
    # interpolated dose volume and is therefore computed later in run_hybrid_dose.
    field_area_jaw = estimate_field_area_jaw(ds)
    
    return {
        'isocenter': iso,
        'rx_gy': rx_gy,
        'n_fractions': n_fx,
        'mu_per_fx': mu_per_fx,
        'total_mu': mu_per_fx * n_fx,
        'energy_mv': energy,
        'field_area_cm2_jaw': field_area_jaw,
    }


def estimate_field_area_jaw(rtplan_ds):
    """Estimate equivalent field area from jaw settings (mean over beams)."""
    areas = []
    for beam in rtplan_ds.BeamSequence:
        cp0 = beam.ControlPointSequence[0]
        if not hasattr(cp0, 'BeamLimitingDevicePositionSequence'):
            continue
        x_w = y_w = None
        for bld in cp0.BeamLimitingDevicePositionSequence:
            jaws = [float(x) for x in bld.LeafJawPositions]
            rt = bld.RTBeamLimitingDeviceType
            if rt in ('X', 'ASYMX'):
                x_w = abs(jaws[1] - jaws[0]) / 10.0
            elif rt in ('Y', 'ASYMY'):
                y_w = abs(jaws[1] - jaws[0]) / 10.0
        if x_w and y_w:
            areas.append(x_w * y_w)
    
    if areas:
        return float(np.mean(areas))
    return REF_FIELD_AREA_CM2  # fallback


def estimate_field_area_isodose50(dose_on_ct, ct_x, ct_y, ct_z, iso, rx_gy):
    """Estimate F_U as the mean of the areas inside the 50% isodose in the
    coronal and sagittal planes through the isocenter (Sanchez-Nieto 2022).

    This method matches the procedure originally used to derive the P3D model
    parameters and is used by Skadion in 3D Slicer for clinical reporting.
    Returns area in cm² + a dict with diagnostics.
    """
    threshold = 0.5 * rx_gy  # 50% of total prescribed dose
    
    # Find isocenter voxel indices on the CT grid
    ix_iso = int(np.argmin(np.abs(ct_x - iso[0])))
    iy_iso = int(np.argmin(np.abs(ct_y - iso[1])))
    iz_iso = int(np.argmin(np.abs(ct_z - iso[2])))
    
    # In-plane voxel sizes (mm)
    px_x = float(ct_x[1] - ct_x[0]) if len(ct_x) > 1 else 1.0
    px_y = float(ct_y[1] - ct_y[0]) if len(ct_y) > 1 else 1.0
    px_z = float(ct_z[1] - ct_z[0]) if len(ct_z) > 1 else 1.0
    
    # Coronal plane: y = constant (iso y), spans x and z
    # dose_on_ct has shape (z, y, x); fix y=iy_iso
    cor_slice = dose_on_ct[:, iy_iso, :]  # shape (nz, nx)
    cor_mask = cor_slice >= threshold
    n_cor = int(cor_mask.sum())
    area_cor_cm2 = n_cor * abs(px_x) * abs(px_z) / 100.0  # mm² -> cm²
    
    # Sagittal plane: x = constant (iso x), spans y and z
    sag_slice = dose_on_ct[:, :, ix_iso]  # shape (nz, ny)
    sag_mask = sag_slice >= threshold
    n_sag = int(sag_mask.sum())
    area_sag_cm2 = n_sag * abs(px_y) * abs(px_z) / 100.0
    
    area_mean = 0.5 * (area_cor_cm2 + area_sag_cm2)
    info = {
        'area_cm2': area_mean,
        'coronal_area_cm2': round(area_cor_cm2, 2),
        'sagittal_area_cm2': round(area_sag_cm2, 2),
        'threshold_gy': threshold,
        'iy_iso': iy_iso, 'ix_iso': ix_iso, 'iz_iso': iz_iso,
    }
    return area_mean, info


def contour_to_mask(contour_data, ct_x, ct_y):
    """Rasterize a single RTSTRUCT contour to a 2D binary mask."""
    n_pts = len(contour_data) // 3
    pts = np.array(contour_data).reshape(n_pts, 3)
    col_idx = (pts[:, 0] - ct_x[0]) / (ct_x[1] - ct_x[0])
    row_idx = (pts[:, 1] - ct_y[0]) / (ct_y[1] - ct_y[0])
    polygon = np.column_stack([col_idx, row_idx])
    path = MplPath(polygon)
    cc, rr = np.meshgrid(np.arange(len(ct_x)), np.arange(len(ct_y)))
    grid = np.column_stack([cc.ravel(), rr.ravel()])
    return path.contains_points(grid).reshape(len(ct_y), len(ct_x))


def build_organ_mask(roi_contour, ct_x, ct_y, ct_z):
    """Build 3D mask from RTSTRUCT contour sequence."""
    mask = np.zeros((len(ct_z), len(ct_y), len(ct_x)), dtype=bool)
    if not hasattr(roi_contour, 'ContourSequence'):
        return mask
    for c in roi_contour.ContourSequence:
        cdata = [float(x) for x in c.ContourData]
        cz = cdata[2]
        zi = np.argmin(np.abs(ct_z - cz))
        if abs(ct_z[zi] - cz) < 3.0:
            mask[zi] |= contour_to_mask(cdata, ct_x, ct_y)
    return mask


# ============================================================
# Main pipeline
# ============================================================
def run_hybrid_dose(data_dir, out_dir, field_area_method='isodose50',
                    field_area_override_cm2=None):
    os.makedirs(out_dir, exist_ok=True)
    
    # --- Handle nested directories ---
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    test_files = [f for f in os.listdir(data_dir) if f.endswith('.dcm')]
    if not test_files and len(subdirs) == 1:
        data_dir = os.path.join(data_dir, subdirs[0])
        print(f"  (Using subdirectory: {data_dir})")
    
    print("=" * 70)
    print("STEP 1: Loading DICOM data")
    print("=" * 70)
    
    ct_files, rtdose_files, rtplan_files, rtstruct_files = find_dicom_files(data_dir)
    
    if not ct_files:
        raise FileNotFoundError(f"No CT files found in {data_dir}")
    if not rtdose_files:
        raise FileNotFoundError(f"No RTDOSE found in {data_dir}")
    if not rtplan_files:
        raise FileNotFoundError(f"No RTPLAN found in {data_dir}")
    if not rtstruct_files:
        raise FileNotFoundError(f"No RTSTRUCT found in {data_dir}")
    
    ct_vol, ct_x, ct_y, ct_z, ct_ps, ct_origin = load_ct(ct_files)
    print(f"  CT: {ct_vol.shape[0]} slices, {ct_vol.shape[1]}x{ct_vol.shape[2]}")
    print(f"  CT z-range: [{ct_z[0]:.1f}, {ct_z[-1]:.1f}] mm")
    
    dose_gy, dose_x, dose_y, dose_z, rtdose_ds = load_rtdose(rtdose_files[0])
    print(f"  RTDOSE: {dose_gy.shape}, max={dose_gy.max():.2f} Gy ({rtdose_ds.DoseSummationType})")
    
    plan = load_rtplan(rtplan_files[0])
    iso = plan['isocenter']
    rx = plan['rx_gy']
    n_fx = plan['n_fractions']
    mu_fx = plan['mu_per_fx']
    total_mu = plan['total_mu']
    fa_jaw = plan['field_area_cm2_jaw']
    
    print(f"  Isocenter: {iso}")
    print(f"  Rx: {rx} Gy in {n_fx} fx, {mu_fx:.0f} MU/fx, {total_mu:.0f} MU total")
    print(f"  Energy: {plan['energy_mv']} MV")
    print(f"  Field area (jaw method): {fa_jaw:.2f} cm²")
    
    # P3D field-area-independent parameters (final F is set after method choice)
    rx_per_fx_mgy = (rx / n_fx) * 1000.0
    eu = rx_per_fx_mgy / mu_fx
    epsilon = eu / REF_EU_MGY_PER_MU
    leakage = REF_LEAKAGE
    
    rtstruct = pydicom.dcmread(rtstruct_files[0])
    roi_names = {r.ROINumber: r.ROIName for r in rtstruct.StructureSetROISequence}
    print(f"  Structures: {len(roi_names)}")
    
    # --- STEP 2: Interpolate RTDOSE onto CT grid ---
    print("\n" + "=" * 70)
    print("STEP 2: Interpolating RTDOSE onto CT grid")
    print("=" * 70)
    
    dose_interp = RegularGridInterpolator(
        (dose_z, dose_y, dose_x), dose_gy,
        method='linear', bounds_error=False, fill_value=0.0
    )
    
    dose_on_ct = np.zeros((len(ct_z), len(ct_y), len(ct_x)), dtype=np.float32)
    yy, xx = np.meshgrid(ct_y, ct_x, indexing='ij')
    for i, zp in enumerate(ct_z):
        zz = np.full_like(xx, zp)
        pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=-1)
        dose_on_ct[i] = dose_interp(pts).reshape(len(ct_y), len(ct_x))
    
    print(f"  Max TPS dose on CT grid: {dose_on_ct.max():.2f} Gy")
    
    # --- STEP 3: 5% isodose mask ---
    print("\n" + "=" * 70)
    print("STEP 3: 5% isodose boundary")
    print("=" * 70)
    
    if rx is None:
        rx = dose_gy.max()  # fallback
        print(f"  WARNING: No Rx found, using max dose {rx:.2f} Gy")
    
    thr = 0.05 * rx
    mask_tps = dose_on_ct >= thr
    for i in range(len(ct_z)):
        if mask_tps[i].any():
            mask_tps[i] = binary_fill_holes(mask_tps[i])
    
    print(f"  Threshold: {thr:.2f} Gy ({thr*1000:.0f} mGy)")
    print(f"  Voxels inside: {mask_tps.sum()} ({100*mask_tps.sum()/mask_tps.size:.1f}%)")
    
    # --- STEP 3b: Resolve field area + P3D F factor ---
    fa_info = {}
    if field_area_override_cm2 is not None:
        fa = float(field_area_override_cm2)
        fa_method_used = 'manual'
        print(f"\n  Field area: {fa:.2f} cm² (manual override)")
    elif field_area_method == 'isodose50':
        fa, fa_info = estimate_field_area_isodose50(
            dose_on_ct, ct_x, ct_y, ct_z, iso, rx)
        fa_method_used = 'isodose50'
        print(f"\n  Field area (50% isodose method, Sanchez-Nieto 2022):")
        print(f"     coronal {fa_info['coronal_area_cm2']:.2f} + "
              f"sagittal {fa_info['sagittal_area_cm2']:.2f} cm² → mean = {fa:.2f}")
    else:
        fa = fa_jaw
        fa_method_used = 'jaw'
        print(f"\n  Field area: {fa:.2f} cm² (jaw method)")
    
    ff = fa / REF_FIELD_AREA_CM2
    print(f"  P3D parameters: EU={eu:.3f} mGy/MU, ε={epsilon:.4f}, F={ff:.4f}, "
          f"leakage={leakage} mGy/MU")
    
    # --- STEP 4: P3D dose ---
    print("\n" + "=" * 70)
    print("STEP 4: P3D dose (Sánchez-Nieto et al. 2022, Eq. 2)")
    print("=" * 70)
    
    dx = (ct_x[None, None, :] - iso[0]) / 10.0
    dy = (ct_y[None, :, None] - iso[1]) / 10.0
    dz = (ct_z[:, None, None] - iso[2]) / 10.0
    
    dx3 = np.broadcast_to(dx, dose_on_ct.shape).copy()
    dy3 = np.broadcast_to(dy, dose_on_ct.shape).copy()
    dz3 = np.broadcast_to(dz, dose_on_ct.shape).copy()
    
    ppd = p3d_ppd_mgy_per_mu(dx3, dy3, dz3, epsilon, ff, leakage)
    dose_p3d = (np.nan_to_num(ppd, nan=0.0) * total_mu / 1000.0).astype(np.float32)
    
    for d_cm in [5, 10, 15, 20, 30, 50]:
        v = p3d_ppd_mgy_per_mu(
            np.array([0.0]), np.array([0.0]), np.array([float(d_cm)]),
            epsilon, ff, leakage
        )
        print(f"  dz={d_cm:>2d} cm: {v[0]:.4f} mGy/MU → {v[0]*total_mu:.1f} mGy total")
    
    # --- STEP 5: Hybrid ---
    print("\n" + "=" * 70)
    print("STEP 5: Hybrid dose map")
    print("=" * 70)
    
    hybrid = np.where(mask_tps, dose_on_ct, dose_p3d)
    print(f"  TPS max: {dose_on_ct[mask_tps].max():.2f} Gy")
    print(f"  P3D max (outside 5%): {dose_p3d[~mask_tps].max():.4f} Gy")
    print(f"  Hybrid max: {hybrid.max():.2f} Gy")
    
    # --- STEP 6: Organ doses ---
    print("\n" + "=" * 70)
    print("STEP 6: Organ doses from RTSTRUCT")
    print("=" * 70)
    
    vox_cc = ct_ps[0] * ct_ps[1] * abs(float(np.diff(ct_z).mean())) / 1000.0
    dist3d = np.sqrt(dx3**2 + dy3**2 + dz3**2)
    
    results = []
    for rc in rtstruct.ROIContourSequence:
        rnum = rc.ReferencedROINumber
        rname = roi_names.get(rnum, f"ROI_{rnum}")
        
        mask = build_organ_mask(rc, ct_x, ct_y, ct_z)
        n = mask.sum()
        if n == 0:
            continue
        
        in_tps = (mask & mask_tps).sum()
        n_oof = n - in_tps
        # Three dose vectors over the organ mask
        hd  = hybrid[mask]
        td  = dose_on_ct[mask]   # what TPS reports over the entire organ
        pd_ = dose_p3d[mask]     # what P3D reports over the entire organ
        dd  = dist3d[mask]
        
        row = {
            'Organ': rname,
            'Volume_cc': round(n * vox_cc, 1),
            'n_voxels': int(n),
            'pct_TPS': round(100 * in_tps / n, 1),
            'pct_OOF': round(100 * n_oof / n, 1),
            # Hybrid (existing default)
            'Dmean_mGy':         round(float(hd.mean())            * 1000, 2),
            'Dmax_mGy':          round(float(hd.max())             * 1000, 2),
            'Dmin_mGy':          round(float(hd.min())             * 1000, 2),
            'Dmedian_mGy':       round(float(np.median(hd))        * 1000, 2),
            'D2pct_mGy':         round(float(np.percentile(hd, 98))* 1000, 2),
            'D98pct_mGy':        round(float(np.percentile(hd, 2)) * 1000, 2),
            # TPS only (over entire organ)
            'Dmean_TPS_only_mGy':   round(float(td.mean())             * 1000, 2),
            'Dmax_TPS_only_mGy':    round(float(td.max())              * 1000, 2),
            'D2pct_TPS_only_mGy':   round(float(np.percentile(td, 98)) * 1000, 2),
            'D98pct_TPS_only_mGy':  round(float(np.percentile(td, 2))  * 1000, 2),
            # P3D only (over entire organ)
            'Dmean_P3D_only_mGy':   round(float(pd_.mean())            * 1000, 2),
            'Dmax_P3D_only_mGy':    round(float(pd_.max())             * 1000, 2),
            'D2pct_P3D_only_mGy':   round(float(np.percentile(pd_, 98))* 1000, 2),
            'D98pct_P3D_only_mGy':  round(float(np.percentile(pd_, 2)) * 1000, 2),
            'mean_dist_cm':      round(float(dd.mean()), 1),
            'min_dist_cm':       round(float(dd.min()),  1),
        }
        results.append(row)
        
        print(f"\n  {rname}:  Vol={row['Volume_cc']} cc  TPS={row['pct_TPS']}%  OOF={row['pct_OOF']}%")
        print(f"    Dmean: hybrid={row['Dmean_mGy']:.1f}  "
              f"TPS_only={row['Dmean_TPS_only_mGy']:.1f}  "
              f"P3D_only={row['Dmean_P3D_only_mGy']:.1f} mGy")
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, 'hybrid_organ_doses.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # --- STEP 7: Figures ---
    print("\n" + "=" * 70)
    print("STEP 7: Figures")
    print("=" * 70)
    
    iz = np.argmin(np.abs(ct_z - iso[2]))
    iy = np.argmin(np.abs(ct_y - iso[1]))
    ix = np.argmin(np.abs(ct_x - iso[0]))
    ext = [ct_x[0], ct_x[-1], ct_y[-1], ct_y[0]]
    
    # Fig 1: Axial
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    s = ct_vol[iz]
    
    axes[0,0].imshow(s, cmap='gray', vmin=-200, vmax=400, extent=ext)
    axes[0,0].set_title(f'CT (z={ct_z[iz]:.0f} mm, isocenter)')
    axes[0,0].set_xlabel('x (mm)'); axes[0,0].set_ylabel('y (mm)')
    
    axes[0,1].imshow(s, cmap='gray', vmin=-200, vmax=400, extent=ext)
    dov = np.ma.masked_where(dose_on_ct[iz] < 0.1, dose_on_ct[iz])
    im = axes[0,1].imshow(dov, cmap='jet', alpha=0.5, vmin=0, vmax=rx*1.1, extent=ext)
    axes[0,1].contour(ct_x, ct_y, mask_tps[iz].astype(float), levels=[0.5], colors='lime', linewidths=2)
    axes[0,1].set_title('TPS + 5% boundary'); axes[0,1].set_xlabel('x (mm)')
    plt.colorbar(im, ax=axes[0,1], label='Gy')
    
    axes[0,2].imshow(s, cmap='gray', vmin=-200, vmax=400, extent=ext)
    hov = np.ma.masked_where(hybrid[iz] < 1e-5, hybrid[iz])
    im = axes[0,2].imshow(hov, cmap='jet', alpha=0.5, norm=LogNorm(vmin=1e-4, vmax=rx*1.1), extent=ext)
    axes[0,2].contour(ct_x, ct_y, mask_tps[iz].astype(float), levels=[0.5], colors='lime', linewidths=2)
    axes[0,2].set_title('Hybrid (log)'); axes[0,2].set_xlabel('x (mm)')
    plt.colorbar(im, ax=axes[0,2], label='Gy')
    
    for j, off in enumerate([-20, -10, 10]):
        zi = max(0, min(len(ct_z)-1, iz + off))
        ax = axes[1,j]
        ax.imshow(ct_vol[zi], cmap='gray', vmin=-200, vmax=400, extent=ext)
        h = np.ma.masked_where(hybrid[zi] < 1e-5, hybrid[zi])
        im = ax.imshow(h, cmap='jet', alpha=0.5, norm=LogNorm(vmin=1e-4, vmax=rx*1.1), extent=ext)
        if mask_tps[zi].any():
            ax.contour(ct_x, ct_y, mask_tps[zi].astype(float), levels=[0.5], colors='lime', linewidths=2)
        ax.set_title(f'z={ct_z[zi]:.0f} mm ({abs(ct_z[zi]-iso[2])/10:.0f} cm from iso)')
        ax.set_xlabel('x (mm)')
        plt.colorbar(im, ax=ax, label='Gy')
    
    plt.suptitle(f'Hybrid Dose — Rx={rx} Gy, {n_fx} fx, ε={epsilon:.3f}, F={ff:.3f}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig1_path = os.path.join(out_dir, 'fig1_axial_slices.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  {fig1_path}")
    
    # Fig 2: z-profile
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(ct_z, np.clip(dose_on_ct[:, iy, ix]*1000, 0.001, None), 'b-', lw=2, label='TPS', alpha=0.7)
    ax.semilogy(ct_z, np.clip(dose_p3d[:, iy, ix]*1000, 0.001, None), 'r--', lw=2, label='P3D', alpha=0.7)
    ax.semilogy(ct_z, np.clip(hybrid[:, iy, ix]*1000, 0.001, None), 'k-', lw=2.5, label='Hybrid', alpha=0.9)
    ax.axhline(y=thr*1000, color='green', ls=':', lw=1.5, label=f'5% ({thr*1000:.0f} mGy)')
    ax.axvline(x=iso[2], color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('z (mm)'); ax.set_ylabel('Dose (mGy)')
    ax.set_title('Dose Profile along z through Isocenter'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.001, 50000)
    fig2_path = os.path.join(out_dir, 'fig2_z_profile.png')
    plt.tight_layout(); plt.savefig(fig2_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  {fig2_path}")
    
    # Fig 3: Organ bars (exclude targets)
    oar_df = df[~df['Organ'].isin(['PTV', 'CTV', 'GTV_prechemo', 'CTV_AVDB', 'Skin',
                                    'hotspots', 'avoid', 'copyinfrclaxli', 'infrclaxli',
                                    'Heart_optim'])].copy()
    if len(oar_df) > 0:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
        nm = oar_df['Organ'].tolist()
        xp = np.arange(len(nm)); w = 0.25
        a1.bar(xp-w, oar_df['Dmean_mGy'], w, label='Mean', color='steelblue')
        a1.bar(xp, oar_df['Dmedian_mGy'], w, label='Median', color='coral')
        a1.bar(xp+w, oar_df['Dmax_mGy'], w, label='Max', color='seagreen', alpha=0.7)
        a1.set_xticks(xp); a1.set_xticklabels(nm, rotation=45, ha='right')
        a1.set_ylabel('mGy'); a1.set_title('OAR Hybrid Doses')
        a1.legend(); a1.set_yscale('log'); a1.grid(True, alpha=0.3, axis='y')
        
        a2.barh(nm, oar_df['pct_TPS'], color='steelblue', label='TPS (≥5%)')
        a2.barh(nm, oar_df['pct_OOF'], left=oar_df['pct_TPS'], color='coral', label='P3D (<5%)')
        a2.set_xlabel('%'); a2.set_title('TPS vs P3D per Organ')
        a2.legend(loc='lower right'); a2.set_xlim(0, 100)
        
        plt.suptitle(f'OAR Analysis — Rx={rx} Gy, {n_fx} fx', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig3_path = os.path.join(out_dir, 'fig3_organ_doses.png')
        plt.savefig(fig3_path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  {fig3_path}")
    
    # Save run info
    info = {
        'data_dir': data_dir,
        'isocenter_mm': iso,
        'rx_gy': rx,
        'n_fractions': n_fx,
        'mu_per_fraction': mu_fx,
        'total_mu': total_mu,
        'energy_mv': plan['energy_mv'],
        'field_area_cm2': fa,
        'field_area_method': fa_method_used,
        'field_area_cm2_jaw_reference': fa_jaw,
        'field_area_isodose_details': {k: v for k, v in fa_info.items()
                                        if k != '_mask_cor' and k != '_mask_sag'},
        'epsilon': round(epsilon, 4),
        'field_factor': round(ff, 4),
        'leakage_mgy_per_mu': leakage,
        'threshold_5pct_gy': round(thr, 4),
        'p3d_constants': {'A1': A1_MGY_CM2_PER_MU, 'A2': A2_MGY_CM_PER_MU, 'A3': A3_PER_CM},
        'ct_slices': len(ct_z),
        'ct_shape': list(ct_vol.shape),
        'n_structures': len(roi_names),
        'notes': [
            'P3D model: Sánchez-Nieto et al., Frontiers in Oncology, 2022',
            'P3D is valid outside the 5% isodose only (model uncertainty ±23.2%)',
            'Hybrid: TPS inside 5% isodose, P3D outside',
            'Decomposition columns: TPS_only and P3D_only computed over the entire',
            'organ mask using each engine separately, for validation/comparison.',
            'For organs entirely OOF: Dmean_mGy ≈ Dmean_P3D_only_mGy.',
            'For organs entirely in-field: Dmean_mGy ≈ Dmean_TPS_only_mGy.',
            'Z-axis: DICOM LPS z = caudal-cranial = P3D z. No inversion needed.',
            'Skin dose: P3D not valid for skin (see paper limitations).',
        ],
    }
    with open(os.path.join(out_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Rx={rx} Gy | {n_fx} fx | {total_mu:.0f} MU | ε={epsilon:.4f} | F={ff:.4f}")
    print(f"Isocenter: {iso} mm")
    print(f"5% threshold: {thr:.2f} Gy\n")
    
    print(f"{'Organ':<20} {'Vol':>7} {'%TPS':>6} {'%OOF':>6} "
          f"{'Dmean_hyb':>10} {'Dmean_TPS':>10} {'Dmean_P3D':>10}")
    print("-" * 84)
    for _, r in df.iterrows():
        print(f"{r['Organ']:<20} {r['Volume_cc']:>6.0f} {r['pct_TPS']:>5.1f}% {r['pct_OOF']:>5.1f}% "
              f"{r['Dmean_mGy']:>9.0f}  {r['Dmean_TPS_only_mGy']:>9.0f}  "
              f"{r['Dmean_P3D_only_mGy']:>9.0f}")
    print("\nAll doses in mGy. Hybrid = TPS inside 5% mask, P3D outside.")
    print("TPS_only / P3D_only = each engine over the entire organ for comparison.")
    
    return df


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Hybrid TPS+P3D dose pipeline for a single dataset (RTSTRUCT-based).')
    p.add_argument('data_dir', type=str,
                   help='Directory containing CT, RTDOSE, RTPLAN, RTSTRUCT')
    p.add_argument('out_dir', nargs='?', type=str, default=None,
                   help='Output directory (default: <data_dir>/../hybrid_output)')
    p.add_argument('--field-area-method', choices=['isodose50', 'jaw'],
                   default='isodose50',
                   help='Field area F_U computation: 50%% isodose (Sanchez-Nieto, default) '
                        'or jaw rectangle mean')
    p.add_argument('--field-area-cm2', type=float, default=None,
                   help='Manual override for F_U; bypasses both auto methods')
    args = p.parse_args()
    
    out_dir = args.out_dir if args.out_dir else os.path.join(
        os.path.dirname(os.path.abspath(args.data_dir)), 'hybrid_output')
    
    run_hybrid_dose(args.data_dir, out_dir,
                    field_area_method=args.field_area_method,
                    field_area_override_cm2=args.field_area_cm2)
