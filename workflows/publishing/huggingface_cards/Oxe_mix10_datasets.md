# ALAM OXE video pretraining mixture

Repository ID: `Mark-ZJTang/Oxe_mix10_datasets`.

The ten converted Open X-Embodiment video datasets consumed by the released ALAM
mixed-data configuration. The payload is published as uncompressed
`data-*.tar` shards plus `ARCHIVE_MANIFEST.sha256` and `DATASET_LAYOUT.json`.
Use the ALAM downloader to verify and reconstruct the ten display-key
directories, each containing MP4 files and `video_metadata.json`:

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact oxe_mix10_datasets
```

This repository must not be made public until the publisher has reviewed and
recorded the redistribution terms and attribution requirements for every source
dataset. Dataset-specific licenses may differ.
