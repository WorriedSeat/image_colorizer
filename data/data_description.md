# Datasets
1. [Source](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
2. [Structure](#structure)
3. [Reproduce](#reproduce)

## Structure
```
data/
├── raw/                       # Raw downloaded & extracted data
│   ├── EuroSAT_RGB.zip        # Original archive (~2.7 GB)
│   └── EuroSAT_RGB/           # Extracted folder (27,000 images)
│       ├── AnnualCrop/
│       │   ├── AnnualCrop_1.jpg
│       │   └── ...
│       ├── Forest/
│       └── ...
│
└── prep/                      # Preprocessed data for training
    └── EuroSAT_RGB/
        ├── image_info.csv     # Metadata: file_name, full_path, original_class (shuffled)
        └── images/            # All images copied in shuffled order
            ├── AnnualCrop_1.jpg
            └── ...
```
## Reproduce

### Pipeline Overview (`dvc.yaml`)
```yaml
stages:
  download  → data/raw/EuroSAT_RGB.zip
  unzip     → data/raw/EuroSAT_RGB/
  prep      → data/prep/EuroSAT_RGB/{image_info.csv, images/}
  cleanup   → (optional) removes raw/zip if configured
```

### Parameters (`params.yaml`)
```yaml
data_pipeline:
  remove_zip: true      # Delete .zip after unzip
  remove_raw: true      # Delete raw folder after prep
```

> **Important**: If `remove_zip: true` or `remove_raw: true`, the corresponding files will be **deleted locally** after `dvc repro`, but remain in **DVC cache** and **remote storage**.

---

```bash
dvc repro
```
> Re-downloads, unzips, preprocesses, and cleans up (based on `params.yaml`).

---