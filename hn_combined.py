#!/usr/bin/env python3
"""
H&N (or any non-WB pCT) combined hybrid dose pipeline.

Architecture (per agreement with Jef Rutten, April 2026):

  Run 1: hybrid pipeline on the source pCT
    - TPS dose inside the 5%-isodose, P3D outside
    - Organ stats from RTSTRUCT (planner-validated contours on patient anatomy)

  Run 2: cross-patient hybrid pipeline on a WB voxel model
    - RTDOSE re-anchored from source isocenter to chosen target isocenter
    - TPS dose inside the 5%-isodose (on WB grid), P3D outside
    - Organ stats from TotalSegmentator NIfTI segmentations on the WB

  Merge (per organ):
    1. In RTSTRUCT and not truncated by pCT bounds  -> run 1 value
    2. In RTSTRUCT but truncated by pCT bounds      -> run 2 value (via name map)
    3. Not in RTSTRUCT but in WB-TS                 -> run 2 value

  Output:
    - combined_organ_doses.csv (one row per organ, "source" column = run1/run2)
    - run1_organ_doses.csv, run2_organ_doses.csv (raw outputs of each run)
    - compatibility_report.csv (organs present in both runs, side-by-side)
    - run_info.json with all parameters

Truncation detection:
    An RTSTRUCT mask is flagged as truncated if it touches the first or last
    CT slice (z-bounds), since a clipped organ would have its mask pressed
    against the pCT edge.

Name matching:
    A default mapping of common RTSTRUCT names to TotalSegmentator names is
    used (see RTSTRUCT_TS_MAP below). Can be extended via --name-map JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_fill_holes
from matplotlib.path import Path as MplPath
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Reuse the cross-patient pipeline as a module
sys.path.insert(0, str(Path(__file__).parent))
from hybrid_crosspatient import (
    P3DParams, p3d_ppd_mgy_per_mu,
    A1_MGY_CM2_PER_MU, A2_MGY_CM_PER_MU, A3_PER_CM,
    REF_FIELD_AREA_CM2, REF_EU_MGY_PER_MU, REF_LEAKAGE_MGY_PER_MU,
    load_ct_volume, load_rtdose, build_rtstruct_masks,
    load_nifti_masks_to_ct_grid,
    interpolate_rtdose_to_ct, compute_p3d_dose_on_ct,
    build_5pct_mask, organ_stats, organ_dvh, write_dvh_plot,
    make_axial_figure, make_zprofile_figure, make_organ_figure,
    run as crosspatient_run,
)


# ============================================================
# Default RTSTRUCT <-> TotalSegmentator name mapping
# ============================================================
# Lowercased on lookup; extend with --name-map JSON when needed.
RTSTRUCT_TS_MAP: Dict[str, str] = {
    # IMPORTANT: spinal_cord is NOT mapped to vertebral_column. TS does not segment
    # the spinal cord (soft-tissue nerve bundle); vertebral_column is bone. Their
    # dose tolerances differ by an order of magnitude. If the RTSTRUCT spinal_cord
    # is truncated by the source pCT, only the in-field portion is reportable;
    # the OOF portion cannot be inferred from TS.
    # cardiothoracic -- map to combined groups
    "heart":       "heart",            # group of 5 TS subparts
    "heart_optim": "heart",
    "lungs":       "lungs",            # group of 5 TS lobes
    "lung_l":      "lung_left",
    "lung_r":      "lung_right",
    "esophagus":   "esophagus",
    "trachea":     "trachea",
    "a_aorta":     "aorta",
    "aorta":       "aorta",
    # head & neck
    "brainstem":   "brainstem",
    "brain":       "brain",
    "thyroid":     "thyroid_gland",
    # eyes
    "lens_left":   "eye_lens_left",
    "lens_right":  "eye_lens_right",
    "retina_left": "eye_left",
    "retina_right":"eye_right",
    "optic_nerve_left":  "optic_nerve_left",
    "optic_nerve_right": "optic_nerve_right",
    "optic_chiasm":      "optic_chiasm",
    "lacrimal_gland_l":  "lacrimal_gland_left",
    "lacrimal_gland_r":  "lacrimal_gland_right",
}


# ============================================================
# Run 1: hybrid pipeline on the source pCT
# ============================================================
def run_on_source_pct(
    src_ct_dir: Path,
    rtdose_path: Path,
    rtstruct_path: Path,
    iso_mm: np.ndarray,
    p3d: P3DParams,
    rx_gy: float,
    total_mu: float,
    outdir: Path,
) -> Tuple[Dict, pd.DataFrame]:
    """Run 1 = the original hybrid pipeline on the source pCT.

    Equivalent to hybrid_crosspatient.run() with target_iso = source_iso and
    RTSTRUCT segmentations. Returns (auxiliary info dict, organ table).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 72)
    print("RUN 1 -- hybrid pipeline on source pCT (RTSTRUCT organs)")
    print("=" * 72)

    ct = load_ct_volume(src_ct_dir)
    print(f"  Source pCT: {ct['n_slices']} slices, "
          f"z=[{ct['z'][0]:.1f}, {ct['z'][-1]:.1f}] mm")
    rtd = load_rtdose(rtdose_path)
    print(f"  RTDOSE: {rtd['summation_type']}, max={rtd['max_gy']:.2f} Gy")

    # No shift -- target_iso = source_iso
    dose_tps = interpolate_rtdose_to_ct(rtd, ct, np.zeros(3))
    dose_p3d, dist_cm = compute_p3d_dose_on_ct(ct, iso_mm, p3d, total_mu)
    mask_5pct = build_5pct_mask(dose_tps, rx_gy)
    hybrid = np.where(mask_5pct, dose_tps, dose_p3d).astype(np.float32)
    print(f"  5% threshold: {0.05 * rx_gy:.2f} Gy, voxels inside: "
          f"{int(mask_5pct.sum())} ({100 * mask_5pct.sum() / mask_5pct.size:.2f}%)")

    masks = build_rtstruct_masks(rtstruct_path, ct["x"], ct["y"], ct["z"])
    print(f"  Built {len(masks)} RTSTRUCT masks")

    dz_mm = abs(float(np.diff(ct["z"]).mean()))
    voxel_cc = ct["pixel_spacing"][0] * ct["pixel_spacing"][1] * dz_mm / 1000.0

    rows = []
    for organ, m in masks.items():
        s = organ_stats(hybrid, dist_cm, m, mask_5pct, voxel_cc)
        if not s:
            continue
        # Truncation detection: mask touches the first or last z-slice?
        touches_top = bool(m[0].any())
        touches_bot = bool(m[-1].any())
        s = {
            "Organ": organ,
            "source": "run1",
            "truncated_top": touches_top,
            "truncated_bottom": touches_bot,
            "truncated": touches_top or touches_bot,
            **s,
        }
        rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "run1_organ_doses.csv", index=False)

    # Save figures specific to run 1
    make_axial_figure(ct, dose_tps, hybrid, mask_5pct, iso_mm, rx_gy,
                      p3d.epsilon, p3d.field_factor,
                      outdir / "run1_fig_axial.png")
    make_zprofile_figure(ct, dose_tps, dose_p3d, hybrid, iso_mm, rx_gy,
                         outdir / "run1_fig_zprofile.png")

    return {
        "ct_z_range_mm": [float(ct["z"][0]), float(ct["z"][-1])],
        "ct_shape": list(ct["volume"].shape),
        "rtdose_max_gy": rtd["max_gy"],
    }, df


# ============================================================
# Merge logic
# ============================================================
def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_ts_match(rt_name: str, ts_names: List[str], extra_map: Dict[str, str]) -> Optional[str]:
    """Find a TS organ name that matches an RTSTRUCT organ name.

    Priority:
      1. Explicit user-provided extra_map (lowercased keys)
      2. Built-in RTSTRUCT_TS_MAP
      3. Normalized exact match (strip non-alphanumeric, lowercase)
    Returns the actual TS name (case-preserving) or None.
    """
    rt_low = rt_name.lower()
    extra_low = {k.lower(): v for k, v in extra_map.items()}

    if rt_low in extra_low:
        target = extra_low[rt_low]
        for t in ts_names:
            if t.lower() == target.lower():
                return t

    if rt_low in RTSTRUCT_TS_MAP:
        target = RTSTRUCT_TS_MAP[rt_low]
        for t in ts_names:
            if t.lower() == target.lower():
                return t

    rt_norm = normalize_name(rt_name)
    for t in ts_names:
        if normalize_name(t) == rt_norm:
            return t

    return None


def merge_runs(
    df_run1: pd.DataFrame,
    df_run2: pd.DataFrame,
    extra_name_map: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the three-path merge logic.

    Returns (combined_df, compatibility_df).
    """
    print("\n" + "=" * 72)
    print("MERGE -- combining run 1 and run 2 per organ")
    print("=" * 72)

    ts_names = df_run2["Organ"].tolist() if not df_run2.empty else []
    used_ts = set()
    combined_rows: List[Dict] = []
    compat_rows: List[Dict] = []

    # --- Path 1 & 2: RTSTRUCT-driven ---
    for _, r1 in df_run1.iterrows():
        rt_name = r1["Organ"]
        ts_match = find_ts_match(rt_name, ts_names, extra_name_map)

        # Compatibility table: anytime both present, record both
        if ts_match is not None:
            r2 = df_run2[df_run2["Organ"] == ts_match].iloc[0]
            compat_rows.append({
                "RTSTRUCT_organ": rt_name,
                "TS_organ": ts_match,
                "Volume_run1_cc": r1["Volume_cc"],
                "Volume_run2_cc": r2["Volume_cc"],
                "Dmean_run1_mGy": r1["Dmean_mGy"],
                "Dmean_run2_mGy": r2["Dmean_mGy"],
                "delta_mGy": round(r2["Dmean_mGy"] - r1["Dmean_mGy"], 2),
                "delta_pct": round(100 * (r2["Dmean_mGy"] - r1["Dmean_mGy"]) /
                                   max(r1["Dmean_mGy"], 1e-6), 1),
                "rtstruct_truncated": bool(r1["truncated"]),
            })

        if r1["truncated"] and ts_match is not None:
            # Path 2: truncated -> use run 2
            r2 = df_run2[df_run2["Organ"] == ts_match].iloc[0]
            row = r2.to_dict()
            row["Organ"] = rt_name  # keep clinical name
            row["source"] = "run2 (RTSTRUCT truncated)"
            row["TS_match"] = ts_match
            combined_rows.append(row)
            used_ts.add(ts_match)
            print(f"  {rt_name:<22}  run 2 (RTSTRUCT truncated, matched TS '{ts_match}')")
        else:
            # Path 1: full RTSTRUCT -> use run 1
            row = r1.to_dict()
            row["source"] = "run1"
            row["TS_match"] = ts_match if ts_match else ""
            combined_rows.append(row)
            if ts_match is not None:
                used_ts.add(ts_match)
            tag = "complete RTSTRUCT" if not r1["truncated"] else "truncated, no TS match"
            print(f"  {rt_name:<22}  run 1 ({tag})")

    # --- Path 3: TS organs not yet used ---
    for _, r2 in df_run2.iterrows():
        ts_name = r2["Organ"]
        if ts_name in used_ts:
            continue
        row = r2.to_dict()
        row["source"] = "run2 (TS only)"
        row["truncated"] = False
        row["truncated_top"] = False
        row["truncated_bottom"] = False
        row["TS_match"] = ts_name
        combined_rows.append(row)
        print(f"  {ts_name:<22}  run 2 (TS only, no RTSTRUCT)")

    combined = pd.DataFrame(combined_rows)
    compatibility = pd.DataFrame(compat_rows)
    return combined, compatibility


# ============================================================
# Combined figure
# ============================================================
def make_combined_figure(combined: pd.DataFrame, rx_gy: float, out_path: Path) -> None:
    excl = combined["Organ"].apply(lambda s: any(k in s.upper() for k in ("PTV", "CTV", "GTV", "BOX", "SKIN")))
    df = combined.loc[~excl].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(6, 0.32 * len(df))))
    colors = {
        "run1":                       "steelblue",
        "run2 (TS only)":             "darkorange",
        "run2 (RTSTRUCT truncated)":  "firebrick",
    }
    df = df.sort_values("Dmean_mGy", ascending=True).reset_index(drop=True)
    bar_colors = [colors.get(s, "gray") for s in df["source"]]
    ax.barh(df["Organ"], df["Dmean_mGy"], color=bar_colors)
    ax.set_xscale("log")
    ax.set_xlabel("Dmean (mGy)")
    ax.set_title(f"Combined organ doses (Rx = {rx_gy} Gy)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for l, c in colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main_combined(
    src_ct_dir: Path,
    rtdose_path: Path,
    rtplan_path: Path,
    rtstruct_path: Path,
    wb_ct_dir: Path,
    wb_seg_dir: Path,
    target_iso_mm: np.ndarray,
    rx_gy: float,
    n_fractions: int,
    total_mu: float,
    field_area_cm2: float,
    energy_mv: float,
    leakage_mgy_per_mu: float,
    extra_name_map: Dict[str, str],
    seg_coord_system: str,
    outdir: Path,
    seg_include_regex: Optional[str] = None,
    seg_exclude_regex: Optional[str] = None,
    organ_groups: Optional[Dict[str, List[str]]] = None,
    keep_individual_members: bool = False,
    extra_run_info: Optional[Dict] = None,
) -> Dict:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Source isocenter from RTPLAN ---
    rp = pydicom.dcmread(str(rtplan_path))
    iso = rp.BeamSequence[0].ControlPointSequence[0].IsocenterPosition
    source_iso_mm = np.array([float(iso[0]), float(iso[1]), float(iso[2])])

    # --- P3D parameters (same for both runs since same RT plan) ---
    eu = (rx_gy / max(n_fractions, 1)) * 1000.0 / max(total_mu / max(n_fractions, 1), 1e-9)
    p3d = P3DParams(
        epsilon=eu / REF_EU_MGY_PER_MU,
        field_factor=field_area_cm2 / REF_FIELD_AREA_CM2,
        leakage=leakage_mgy_per_mu,
    )

    print(f"Source isocenter (from RTPLAN): {source_iso_mm.tolist()} mm")
    print(f"Target isocenter (on WB):       {target_iso_mm.tolist()} mm")
    print(f"P3D parameters: eps={p3d.epsilon:.4f}, F={p3d.field_factor:.4f}, "
          f"Lu={p3d.leakage} mGy/MU, total_MU={total_mu:.0f}")

    # --- Run 1 on source pCT ---
    run1_dir = outdir / "run1_pCT"
    run1_info, df_run1 = run_on_source_pct(
        src_ct_dir, rtdose_path, rtstruct_path, source_iso_mm,
        p3d, rx_gy, total_mu, run1_dir,
    )

    # --- Run 2 on WB ---
    print("\n" + "=" * 72)
    print("RUN 2 -- cross-patient hybrid pipeline on WB voxel model")
    print("=" * 72)
    run2_dir = outdir / "run2_WB"
    df_run2 = crosspatient_run(
        wb_ct_dir=wb_ct_dir,
        rtdose_path=rtdose_path,
        source_iso_mm=source_iso_mm,
        target_iso_mm=target_iso_mm,
        rx_gy=rx_gy,
        n_fractions=n_fractions,
        total_mu=total_mu,
        field_area_cm2=field_area_cm2,
        energy_mv=energy_mv,
        leakage_mgy_per_mu=leakage_mgy_per_mu,
        seg_nifti_dir=wb_seg_dir,
        rtstruct_path=None,
        seg_coord_system=seg_coord_system,
        outdir=run2_dir,
        write_dvhs=False,
        seg_include_regex=seg_include_regex,
        seg_exclude_regex=seg_exclude_regex,
        organ_groups=organ_groups,
        keep_individual_members=keep_individual_members,
    )
    df_run2["source"] = "run2"

    # --- Merge ---
    combined, compatibility = merge_runs(df_run1, df_run2, extra_name_map)

    # --- Classify each row by clinical category ---
    def classify_organ(row) -> str:
        """Categorize each organ into clinical reporting buckets.

        Categories (5):
          OAR_clinical : RTSTRUCT organs that are real OARs (used for plan eval)
          OAR_TS       : TS-derived organ (group or individual; with default settings,
                         groups have replaced their members so each anatomical
                         structure appears exactly once)
          PRV          : RTSTRUCT planning-risk-volume expansions (suffix _exp)
          target       : RTSTRUCT target volumes (PTV/CTV/GTV)
          plan_helper  : Optimization helpers (expLD*, _opt, _avoid, _ring, ...)
        """
        name = str(row["Organ"])
        nl = name.lower()
        is_run1 = (str(row.get("source", "")).startswith("run1"))
        # Targets first
        if any(t in nl for t in ["ptv", "ctv", "gtv", "itv"]):
            return "target"
        # Planning helpers
        if (nl.startswith("expld") or nl.startswith("ld") or "_opt" in nl
                or "_avoid" in nl or "_ring" in nl or "_helper" in nl
                or "boost" in nl or "couch" in nl):
            return "plan_helper"
        # PRV expansions  (e.g. retina_l_exp0.2, brainstem_exp0.3, spinal_cordex0.5)
        if (re.search(r"exp\d", nl) or nl.endswith("_prv")
                or "_prv_" in nl or nl.endswith("_prv0.5") or nl.endswith("_prv0.3")):
            return "PRV"
        # TS rows (run 2 source) -- single category whether group or individual
        if not is_run1:
            return "OAR_TS"
        # Otherwise RTSTRUCT clinical OAR
        return "OAR_clinical"

    combined["category"] = combined.apply(classify_organ, axis=1)
    # Sort: clinical OARs (RTSTRUCT) first, then TS organs, then PRVs/targets/helpers
    cat_order = {"OAR_clinical": 0, "OAR_TS": 1,
                 "PRV": 2, "target": 3, "plan_helper": 4}
    combined["_sort_cat"] = combined["category"].map(cat_order).fillna(9)
    combined = combined.sort_values(["_sort_cat", "mean_dist_cm"]).drop(columns=["_sort_cat"])

    combined.to_csv(outdir / "combined_organ_doses.csv", index=False)
    compatibility.to_csv(outdir / "compatibility_report.csv", index=False)
    df_run1.to_csv(outdir / "run1_organ_doses_full.csv", index=False)
    df_run2.to_csv(outdir / "run2_organ_doses_full.csv", index=False)

    n_by_cat = combined["category"].value_counts().to_dict()
    print(f"  Combined report: {len(combined)} rows  "
          f"(clinical OARs={n_by_cat.get('OAR_clinical',0)}, "
          f"TS={n_by_cat.get('OAR_TS',0)}, "
          f"PRV={n_by_cat.get('PRV',0)}, "
          f"target={n_by_cat.get('target',0)}, "
          f"helper={n_by_cat.get('plan_helper',0)})")

    make_combined_figure(combined, rx_gy, outdir / "fig_combined_organs.png")

    # --- Run info ---
    info = {
        "source_data_dir": str(src_ct_dir),
        "wb_ct_dir": str(wb_ct_dir),
        "wb_seg_dir": str(wb_seg_dir),
        "rtdose_path": str(rtdose_path),
        "rtplan_path": str(rtplan_path),
        "rtstruct_path": str(rtstruct_path),
        "source_isocenter_mm": source_iso_mm.tolist(),
        "target_isocenter_mm": target_iso_mm.tolist(),
        "shift_mm": (target_iso_mm - source_iso_mm).tolist(),
        "rx_gy": rx_gy,
        "n_fractions": n_fractions,
        "total_mu": total_mu,
        "field_area_cm2": field_area_cm2,
        "energy_mv": energy_mv,
        "leakage_mgy_per_mu": leakage_mgy_per_mu,
        "epsilon": round(p3d.epsilon, 4),
        "field_factor": round(p3d.field_factor, 4),
        "n_organs_combined": len(combined),
        "n_organs_run1": len(df_run1),
        "n_organs_run2": len(df_run2),
        "n_compatibility_overlap": len(compatibility),
        "extra_name_map": extra_name_map,
        "run1_info": run1_info,
        "merge_logic": [
            "Path 1: organ in RTSTRUCT, not truncated by pCT bounds -> run 1",
            "Path 2: organ in RTSTRUCT, truncated by pCT bounds -> run 2 via name map",
            "Path 3: organ not in RTSTRUCT, in TS-WB -> run 2",
        ],
    }
    if extra_run_info:
        info.update(extra_run_info)
    with open(outdir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("COMBINED OUTPUT")
    print("=" * 72)
    print(f"  combined CSV:        {outdir / 'combined_organ_doses.csv'}")
    print(f"  compatibility CSV:   {outdir / 'compatibility_report.csv'}")
    print(f"  per-run CSVs:        {outdir / 'run1_organ_doses_full.csv'}")
    print(f"                       {outdir / 'run2_organ_doses_full.csv'}")
    print(f"  figures:             {outdir / 'fig_combined_organs.png'}")
    print(f"                       {run1_dir}/run1_fig_*.png")
    print(f"                       {run2_dir}/fig*.png")
    print(f"  run info:            {outdir / 'run_info.json'}")
    return info


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Combined H&N hybrid dose pipeline (run 1 on pCT + run 2 on WB).\n"
            "Treatment parameters auto-extract from RTPLAN by default; use the "
            "explicit flags below to override."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src-ct-dir", type=Path, required=True,
                   help="Directory with the source pCT DICOM slices")
    p.add_argument("--rtdose", type=Path, required=True)
    p.add_argument("--rtplan", type=Path, required=True)
    p.add_argument("--rtstruct", type=Path, required=True)
    p.add_argument("--wb-ct-dir", type=Path, required=True,
                   help="Directory with the WB voxel model CT DICOM slices")
    p.add_argument("--wb-seg-dir", type=Path, required=True,
                   help="Directory with TotalSegmentator NIfTI masks on the WB")

    # Auto-detected by default; manually overridable
    p.add_argument("--target-iso-mm", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Target isocenter on the WB CT (DICOM LPS mm). "
                        "AUTO-DEFAULT: anchor-organ centroid + (src_iso - src_anchor_centroid). "
                        "Anchor is the first available of: brain, heart_myocardium, liver, spleen, "
                        "stomach, urinary_bladder, kidney_left, kidney_right.")
    p.add_argument("--anchor-organ", type=str, default=None,
                   help="TS organ name to use as anatomical anchor for target_iso default.")
    p.add_argument("--rx-gy", type=float, default=None,
                   help="Total prescription dose (Gy). AUTO-EXTRACT: TargetPrescriptionDose from RTPLAN.")
    p.add_argument("--n-fractions", type=int, default=None,
                   help="Number of fractions. AUTO-EXTRACT: NumberOfFractionsPlanned from RTPLAN.")
    p.add_argument("--total-mu", type=float, default=None,
                   help="Total MU summed over all beams x fractions. "
                        "AUTO-EXTRACT: sum of BeamMeterset x n_fractions from RTPLAN.")
    p.add_argument("--field-area-cm2", type=float, default=None,
                   help="Equivalent field area (cm²). AUTO: 50%% isodose method "
                        "(Sanchez-Nieto 2022): mean of areas inside 50%% isodose in coronal "
                        "and sagittal planes through source isocenter.")
    p.add_argument("--field-area-method", choices=["isodose50", "jaw"], default="isodose50",
                   help="Method to use when --field-area-cm2 is not given.")
    p.add_argument("--energy-mv", type=float, default=None,
                   help="Beam energy (MV). AUTO-EXTRACT: NominalBeamEnergy of first beam in RTPLAN.")

    p.add_argument("--leakage-mgy-per-mu", type=float, default=REF_LEAKAGE_MGY_PER_MU)
    p.add_argument("--seg-coord-system", choices=["ras", "lps"], default="ras")
    p.add_argument("--seg-include-regex", default=None,
                   help="Regex (case-insensitive) to filter NIfTI organ names; only matching are kept")
    p.add_argument("--seg-exclude-regex",
                   default=None,
                   help="Regex (case-insensitive) to exclude NIfTI organ names from individual "
                        "reporting (group accumulators are NOT affected by this filter; group members "
                        "always feed into their groups). Default: no filter (all organs reported).")
    p.add_argument("--name-map", type=Path, default=None,
                   help="Optional JSON file with extra RTSTRUCT->TS name mappings")
    p.add_argument("--organ-groups-json", type=Path, default=None,
                   help="Optional JSON file with custom organ groups (group_name -> [TS members]). "
                        "Default groups are used if not provided.")
    p.add_argument("--no-organ-groups", action="store_true",
                   help="Disable organ grouping; report every TS NIfTI individually")
    p.add_argument("--keep-individual-members", action="store_true",
                   help="When grouping, also report individual member organs alongside groups "
                        "(default: groups replace their members for a clean clinical report)")
    p.add_argument("--outdir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from hybrid_crosspatient import (
        DEFAULT_ORGAN_GROUPS, extract_rtplan_parameters, compute_target_iso_default,
        compute_field_area_50pct_isodose, compute_field_area_jaw, load_ct_volume,
    )
    args = parse_args(argv)
    extra_map: Dict[str, str] = {}
    if args.name_map is not None:
        extra_map = json.loads(args.name_map.read_text())

    # ---- AUTO-EXTRACT MISSING TREATMENT PARAMS FROM RTPLAN ----
    print(f"\n{'='*72}\nResolving treatment parameters\n{'='*72}")
    rp_params = extract_rtplan_parameters(args.rtplan)
    print(f"  RTPLAN: '{rp_params['plan_label']}' ({rp_params['beam_count']} beams)")
    print(f"  RTPLAN auto-extract: rx={rp_params['rx_gy']} Gy, n_fx={rp_params['n_fractions']}, "
          f"total_mu={rp_params['total_mu']}, energy={rp_params['energy_mv']} MV")
    print(f"  RTPLAN isocenter:   {rp_params['isocenter_mm']}")

    rx_gy = args.rx_gy if args.rx_gy is not None else rp_params["rx_gy"]
    n_fx = args.n_fractions if args.n_fractions is not None else rp_params["n_fractions"]
    total_mu = args.total_mu if args.total_mu is not None else rp_params["total_mu"]
    energy_mv = args.energy_mv if args.energy_mv is not None else rp_params["energy_mv"]
    if rx_gy is None or n_fx is None or total_mu is None or energy_mv is None:
        raise SystemExit(
            f"Could not auto-extract all treatment parameters from RTPLAN; "
            f"missing one of rx_gy/n_fx/total_mu/energy. Use explicit CLI flags."
        )
    src_iso = np.array(rp_params["isocenter_mm"], dtype=np.float64)

    # ---- AUTO-COMPUTE FIELD AREA ----
    if args.field_area_cm2 is not None:
        fa = float(args.field_area_cm2)
        fa_method = "manual"
        fa_info = {}
    elif args.field_area_method == "isodose50":
        fa_info = compute_field_area_50pct_isodose(args.rtdose, src_iso, rx_gy)
        fa = fa_info["area_cm2"]
        fa_method = "isodose50"
        print(f"  Field area (50% isodose method, Sanchez-Nieto 2022):")
        print(f"     coronal {fa_info['coronal_area_cm2']:.2f} cm² + "
              f"sagittal {fa_info['sagittal_area_cm2']:.2f} cm² → mean = {fa:.2f} cm²")
    else:
        fa = compute_field_area_jaw(args.rtplan)
        fa_method = "jaw"
        fa_info = {}
        print(f"  Field area (jaw mean): {fa:.2f} cm²")

    # ---- AUTO-COMPUTE TARGET_ISO ----
    if args.target_iso_mm is not None:
        target_iso = np.array(args.target_iso_mm, dtype=np.float64)
        ti_info = {"method": "manual"}
        print(f"  Target iso (manual): {target_iso}")
    else:
        src_ct = load_ct_volume(args.src_ct_dir)
        wb_ct = load_ct_volume(args.wb_ct_dir)
        candidates = None
        if args.anchor_organ is not None:
            candidates = [args.anchor_organ]
        target_iso, ti_info = compute_target_iso_default(
            src_ct=src_ct, src_iso_lps_mm=src_iso,
            src_rtstruct_path=args.rtstruct,
            wb_ct=wb_ct, wb_seg_dir=args.wb_seg_dir,
            seg_coord_system=args.seg_coord_system,
            anchor_organ_candidates=candidates,
        )
        ti_info["method"] = "auto"
        print(f"  Target iso (auto via anchor='{ti_info['anchor_organ_ts']}'): "
              f"{np.round(target_iso, 2).tolist()}")

    # Resolve organ groups
    if args.no_organ_groups:
        organ_groups = None
    elif args.organ_groups_json is not None:
        organ_groups = json.loads(args.organ_groups_json.read_text())
    else:
        organ_groups = dict(DEFAULT_ORGAN_GROUPS)

    main_combined(
        src_ct_dir=args.src_ct_dir,
        rtdose_path=args.rtdose,
        rtplan_path=args.rtplan,
        rtstruct_path=args.rtstruct,
        wb_ct_dir=args.wb_ct_dir,
        wb_seg_dir=args.wb_seg_dir,
        target_iso_mm=target_iso,
        rx_gy=float(rx_gy),
        n_fractions=int(n_fx),
        total_mu=float(total_mu),
        field_area_cm2=float(fa),
        energy_mv=float(energy_mv),
        leakage_mgy_per_mu=float(args.leakage_mgy_per_mu),
        extra_name_map=extra_map,
        seg_coord_system=args.seg_coord_system,
        outdir=args.outdir,
        seg_include_regex=args.seg_include_regex,
        seg_exclude_regex=args.seg_exclude_regex,
        organ_groups=organ_groups,
        keep_individual_members=args.keep_individual_members,
        # Pass extra info for run_info.json
        extra_run_info={
            "rtplan_auto": rp_params,
            "field_area_method": fa_method,
            "field_area_isodose_details": {k: v for k, v in fa_info.items()
                                           if not k.startswith("_mask")},
            "target_iso_info": ti_info,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
