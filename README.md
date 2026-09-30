# GeoEarthFM — Multitask GeoAI Prototype

This project is a **student-scale GeoAI foundation-model prototype**, not a claim of a production-scale foundation model.

It combines:
1. EuroSAT RGB land-cover classification.
2. STTORM-CD Sentinel-2 temporal change detection.
3. A shared GeoAI encoder that produces a 256-dimensional representation.
4. A Streamlit demonstration UI.

The STTORM-CD dataset contains 12 training scenes and 4 test scenes, with RGB PNG and NumPy formats; its NumPy data has 14 channels, including the 13 original Sentinel-2 bands plus a cloud-mask channel, and binary change masks. The reference implementation treats the chronologically last image as the after image and earlier images as before images.

## Expected folders

Put this repository contents beside your extracted datasets:

GeoEarthFM/
- dataset/
  - train/
  - test/
  - label_colors.txt
- EuroSAT_RGB/
  - AnnualCrop/
  - Forest/
  - ...
  - SeaLake/
- geoearthfm/
- train_landcover.py
- train_change.py
- predict_change.py
- inspect_data.py
- app.py
- requirements.txt

## Windows setup

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Inspect the data

```powershell
python inspect_data.py
```

This should show the 10 EuroSAT classes and the STTORM events/sequences.

## 2. Train land-cover model

Start small:

```powershell
python train_landcover.py --epochs 5
```

Best model:
`outputs/landcover_best.pt`

## 3. Train change-detection model

For a fast prototype:

```powershell
python train_change.py --epochs 5 --max-samples 1000
```

For a stronger run:

```powershell
python train_change.py --epochs 10 --max-samples 3000
```

Best model:
`outputs/change_best.pt`

## 4. Run dashboard

```powershell
streamlit run app.py
```

## 5. Important terminology

Do not describe this as a giant foundation model trained from scratch.

Use:
"GeoAI foundation-model prototype with a shared multimodal representation and downstream Earth-observation tasks."

The architecture is intentionally lightweight so it can run on a student workstation.

## If STTORM loading fails

Run:

```powershell
python inspect_data.py
```

and send the complete output. The loader expects the official STTORM structure:

`dataset/train/<event>/<sequence>/all_bands/*.npy`
`dataset/train/<event>/<sequence>/mask.npy`

The official STTORM repository documents an example:
`./data/dataset/train/italia_emilia_romagna/1/mask.npy`.
