"""
Lymphoma combined hybrid-dose pipeline.
Single-pCT version: Run 1 (hybrid TPS+P3D with RTSTRUCT organs on the pCT) is
combined with Run 2 (hybrid TPS+P3D with TotalSegmentator NIfTIs on the SAME
pCT), and the two organ tables are merged via the same 3-path logic that is
used in hn_combined.py. No cross-anatomy projection is needed because the
lymphoma planning CT is itself whole-body.

This script reuses the proven backend from hybrid_crosspatient.py and the
merge/classification logic from hn_combined.py.

Usage:
    python3 lymfoom_combined.py \\
        --data-dir /path/to/VB_HODGKIN \\
        --seg-dir /path/to/TS_NIfTIs/ \\
        --outdir /path/to/results/
        [--field-area-method {isodose50, jaw}]
        [--field-area-cm2 X.X]
        [--rx-gy X --total-mu X --n-fractions N --energy-mv X]
        [--keep-individual-members]
        [--name-map file.json]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def main(argv: Optional[Sequence[str]] = None) -> int:
    from hybrid_crosspatient import (
        DEFAULT_ORGAN_GROUPS,
        load_ct_volume, load_rtdose,
        extract_rtplan_parameters, compute_field_area_50pct_isodose,
        compute_field_area_jaw,
        REF_FIELD_AREA_CM2,
    )
    from hn_combined import (
        RTSTRUCT_TS_MAP, find_ts_match, merge_runs, main_combined,
    )

    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory with CT slices + RTDOSE/RTPLAN/RTSTRUCT")
    p.add_argument("--seg-dir", type=Path, required=True,
                   help="Directory with TotalSegmentator NIfTI files")
    p.add_argument("--outdir", type=Path, required=True)

    # RTPLAN overrides (auto-extracted if not given)
    p.add_argument("--rx-gy",        type=float, default=None)
    p.add_argument("--n-fractions",  type=int,   default=None)
    p.add_argument("--total-mu",     type=float, default=None)
    p.add_argument("--energy-mv",    type=float, default=None)

    p.add_argument("--field-area-cm2", type=float, default=None,
                   help="Manual override for F_U (cm²)")
    p.add_argument("--field-area-method",
                   choices=["isodose50", "jaw"], default="isodose50",
                   help="Auto-method when --field-area-cm2 is not given")

    p.add_argument("--name-map", type=Path, default=None,
                   help="JSON with extra RTSTRUCT->TS name mappings")
    p.add_argument("--keep-individual-members", action="store_true",
                   help="Report individual TS members alongside groups "
                        "(default: groups replace members)")

    p.add_argument("--seg-coord-system", choices=["ras", "lps"], default="ras",
                   help="Coordinate system of seg NIfTIs (TotalSegmentator: ras)")
    args = p.parse_args(argv)

    # ---------- Locate RT files in data-dir ----------
    data_dir = args.data_dir
    rt_dose = rt_plan = rt_struct = None
    for f in data_dir.iterdir():
        nm = f.name
        if nm.startswith("RTDOSE") and nm.endswith(".dcm"):
            rt_dose = f
        elif nm.startswith("RTPLAN") and nm.endswith(".dcm"):
            rt_plan = f
        elif nm.startswith("RTSTRUCT") and nm.endswith(".dcm"):
            rt_struct = f
    missing = [n for n, v in [("RTDOSE", rt_dose), ("RTPLAN", rt_plan),
                              ("RTSTRUCT", rt_struct)] if v is None]
    if missing:
        sys.stderr.write(f"ERROR: missing in data-dir: {missing}\n")
        return 2

    # ---------- RTPLAN auto-extract (same logic as hn_combined.py) ----------
    print("=" * 72)
    print("LYMPHOMA COMBINED PIPELINE (single pCT, RTSTRUCT + TS)")
    print("=" * 72)
    rp_params = extract_rtplan_parameters(rt_plan)
    rx_gy       = args.rx_gy       if args.rx_gy       is not None else rp_params["rx_gy"]
    n_fractions = args.n_fractions if args.n_fractions is not None else rp_params["n_fractions"]
    total_mu    = args.total_mu    if args.total_mu    is not None else rp_params["total_mu"]
    energy_mv   = args.energy_mv   if args.energy_mv   is not None else rp_params["energy_mv"]
    source_iso  = rp_params["isocenter_mm"]
    print(f"  RTPLAN: rx={rx_gy} Gy/{n_fractions}fx, MU={total_mu}, energy={energy_mv} MV")
    print(f"  Source isocenter (LPS): {source_iso}")

    # ---------- Field area ----------
    fa_info = {}
    if args.field_area_cm2 is not None:
        fa = float(args.field_area_cm2)
        fa_method = "manual"
        print(f"  Field area: {fa:.2f} cm² (manual override)")
    elif args.field_area_method == "isodose50":
        fa_info = compute_field_area_50pct_isodose(rt_dose, source_iso, rx_gy)
        fa = fa_info["area_cm2"]
        fa_method = "isodose50"
        print(f"  Field area (50% isodose, Sanchez-Nieto):")
        print(f"     coronal {fa_info['coronal_area_cm2']:.2f} + "
              f"sagittal {fa_info['sagittal_area_cm2']:.2f} cm² → mean = {fa:.2f}")
    else:
        fa = compute_field_area_jaw(rt_plan)
        fa_method = "jaw"
        print(f"  Field area (jaw mean): {fa:.2f} cm²")

    # ---------- Extra name map ----------
    extra_map = {}
    if args.name_map is not None:
        extra_map = json.loads(args.name_map.read_text())
    # IMPORTANT: spinal_cord is now valid TS structure for this dataset
    # (TS full task includes it). Add it to the mapping.
    extra_map.setdefault("spinalcord", "spinal_cord")
    extra_map.setdefault("spinal_cord", "spinal_cord")
    extra_map.setdefault("cord", "spinal_cord")

    # ---------- Call main_combined with target == source ----------
    # main_combined() expects src_ct_dir + wb_ct_dir + wb_seg_dir + target_iso.
    # For lymphoma, src == wb, target_iso == source_iso (shift = 0).
    args.outdir.mkdir(parents=True, exist_ok=True)
    organ_groups = dict(DEFAULT_ORGAN_GROUPS)
    target_iso_mm = np.asarray(source_iso, dtype=float)  # shift = 0

    result = main_combined(
        src_ct_dir=data_dir,
        rtdose_path=rt_dose,
        rtplan_path=rt_plan,
        rtstruct_path=rt_struct,
        wb_ct_dir=data_dir,           # same as source for lymphoma
        wb_seg_dir=args.seg_dir,
        target_iso_mm=target_iso_mm,
        rx_gy=rx_gy,
        n_fractions=n_fractions,
        total_mu=total_mu,
        field_area_cm2=fa,
        energy_mv=energy_mv,
        leakage_mgy_per_mu=0.001,     # paper reference value
        extra_name_map=extra_map,
        seg_coord_system=args.seg_coord_system,
        outdir=args.outdir,
        seg_include_regex=None,
        seg_exclude_regex=None,
        organ_groups=organ_groups,
        keep_individual_members=args.keep_individual_members,
        extra_run_info={
            "case_type": "lymphoma_single_pct",
            "rtplan_auto": rp_params,
            "field_area_method": fa_method,
            "field_area_isodose_details": {k: v for k, v in fa_info.items()
                                            if not k.startswith("_mask")},
            "notes": [
                "Single-pCT lymphoma case: source CT == target CT, shift = 0.",
                "spinal_cord mapping enabled (TS full task segments it).",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
