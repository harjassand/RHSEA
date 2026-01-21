# NHSEA Phase 0 (v1 closeout)

This repo contains the locked artifacts, reports, and reproducibility bundle for the NHSEA v1 phase 0/phase 3 closeout.

## What is here
- Final closeout report: `NHSEA_V1_FINAL_CLOSEOUT.md`
- Reproducibility bundle: `repro_bundle/`
- Phase reports and master CSVs in `runs/` and repo root
- Asymmetry analysis outputs: `asymmetry_master.csv`, `asymmetry_report.md`

## Quick start
- See the full closeout summary: `NHSEA_V1_FINAL_CLOSEOUT.md`
- Reproduce key results: run `repro_bundle/commands.sh` (requires the recorded environment in `repro_bundle/environment.txt`)

## How to run v3 (baseline-only preflight + confirmatory)
- Use the project `.venv` with torch installed. System Python does not have torch.
- Preflight (baseline-only): `./run_v3_preflight.sh` or `python -m nhsea.v3_preflight --out runs/v3_preflight/`
- Confirmatory (mechanism runs): `./run_v3_confirmatory.sh`

## Notes
- NHSEA v1 is closed out; no further sweeps are intended.
- If you continue, the intended next step is an NHSEA v2 task redesign (spec not included here).
