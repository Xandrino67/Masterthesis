# Out-of-field hybrid dose pipeline (TPS + Periphocal3D)

Python implementation of the out-of-field (OOF) radiotherapy dose pipeline used in the
master's dissertation *"Evaluation of the cumulative imaging and out-of-field radiation
dose in lymphoma cancer patients"* (Xander Van den Berghe, MSc Biomedical Engineering,
Ghent University / VUB, 2025–2026).

The pipeline reconstructs a **single whole-body dose map** by combining the **in-field**
component of the clinical treatment planning system (RTDOSE) with the **out-of-field**
component of the analytical **Periphocal3D** model (Sánchez-Nieto et al., 2022). The
in-field/out-of-field split is decided per voxel using the 5 % isodose contour. Per-organ
dose statistics and DVHs are extracted by overlaying organ masks (clinical RTSTRUCT
contours and/or TotalSegmentator NIfTI segmentations) onto the resulting dose cube.

> The cumulative **imaging** dose side of the thesis (CT/CBCT via ImpactMC + PET via ICRP-128)
> lives in the separate **Charlie** web application (React/TypeScript/Vite + Supabase).

---

## Files

| File | Role | Description |
|------|------|-------------|
| `hybrid_crosspatient.py` | **Core library / engine** | The shared backend for the whole pipeline. Implements the Periphocal3D peripheral-photon-dose equation (`p3d_ppd_mgy_per_mu`, `P3DParams`), automatic extraction of treatment parameters from the RTPLAN (prescribed dose, fractions, total MU, beam energy, isocenter), **both** field-area definitions — jaw aperture (`estimate_field_area_jaws`) and the 50 % isodose method in the coronal/sagittal planes (`compute_field_area_50pct_isodose`) — CT and RTDOSE loading, RTSTRUCT contour→mask conversion, and resampling of NIfTI masks onto the CT grid. Imported by the orchestrators below. |
| `lymfoom_combined.py` | **Lymphoma case (own pCT)** | CLI orchestrator for the Hodgkin lymphoma case run on the patient's own planning CT. Takes one patient folder (CT + RTDOSE + RTPLAN + RTSTRUCT) and a TotalSegmentator NIfTI directory, computes the hybrid per-organ doses, with options for the field-area method (`--field-area-method`, `--field-area-cm2`) and RTSTRUCT→TotalSegmentator name mapping. |
| `hn_combined.py` | **Cross-anatomy case** | CLI orchestrator for projecting a plan across anatomies (the head-and-neck / whole-brain case mapped onto whole-body reference voxel models). Maps a source planning-CT plan onto a target whole-body CT with TotalSegmentator masks (`run_on_source_pct`), merges the in-field (TPS) and out-of-field (P3D) runs (`merge_runs`), and renders the combined per-organ figure. |
| `hybrid_dose_final.py` | **Standalone hybrid + phantom runner** | Self-contained, single-file version of the hybrid combination (TPS RTDOSE inside the 5 % isodose, P3D outside) for one patient DICOM folder. Used among others for the **Alderson Rando phantom film-validation** runs. Entry point: `run_hybrid_dose(data_dir, out_dir)`. |
| `periphocal3d_dvh_ctdicom_nifti_rtdose.py` | **Periphocal3D-only calculator** | Standalone OOF dose + organ-DVH calculator using **only** the analytical P3D model (no TPS hybrid). Loads a CT DICOM series and NIfTI segmentations (per-organ masks **or** a labelmap), builds the P3D peripheral-dose cube and per-organ DVHs in DICOM LPS coordinates. This is the engine wrapped by the notebook. |
| `periphocal3d_organ.ipynb` | **Interactive notebook** | Jupyter/Colab front-end for the P3D calculator: loads CT + segmentations, interactive isocenter placement (sliders, voxel index, or LPS mm), and exports organ doses + DVHs to CSV. |
| `requirements.txt` | dependencies | Python package list (see below). |


---

## Dependencies

```
numpy
scipy
pydicom
nibabel
matplotlib
pandas
ipywidgets   # interactive notebook only
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Typical usage

Every script exposes `-h/--help` with the full, authoritative flag list. Representative calls:

```bash
# Lymphoma case on the patient's own planning CT
python lymfoom_combined.py \
    --patient-dir   /path/to/CT_RTDOSE_RTPLAN_RTSTRUCT \
    --seg-dir       /path/to/totalsegmentator_nifti \
    --field-area-method isodose

# Cross-anatomy (H&N / whole-brain) projected onto a whole-body voxel model
python hn_combined.py \
    --source-pct-dir /path/to/source_pCT \
    --wb-ct-dir      /path/to/wholebody_CT \
    --ts-dir         /path/to/totalsegmentator_nifti \
    --target-iso     <x> <y> <z>

# Periphocal3D-only OOF dose + DVH (no TPS hybrid)
python periphocal3d_dvh_ctdicom_nifti_rtdose.py \
    --ct-dir      /path/to/CT_dicom \
    --seg-dir     /path/to/nifti_masks \
    --isocenter-lps <x> <y> <z>

# Standalone hybrid run (also used for the Rando phantom validation)
python hybrid_dose_final.py   # configure data_dir / out_dir inside run_hybrid_dose(...)
```

All coordinates are handled in the **DICOM LPS (mm)** frame throughout the pipeline; no
image-viewer/index conventions are used outside the notebook's visualisation widgets.

---

## Data

**No patient data is included in this repository.** All clinical CT/RTDOSE/RTPLAN/RTSTRUCT
inputs were retrospectively obtained, fully anonymised data made available through the
Department of Radiotherapy of Ghent University Hospital (UZ Ghent) and fall under the
corresponding ethical approval; they are not redistributable. The curated per-organ dose
results are reported in the dissertation itself.

---

## Citation

Sánchez-Nieto, B. et al. *Periphocal3D: an analytical model for the calculation of
out-of-field photon dose in 3D*, Physica Medica, 2022.
