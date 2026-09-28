# ALAM CALVIN NPZ dataset

Repository ID: `Mark-ZJTang/CALVIN_task_ABC_D`.

Converted CALVIN RGB-static frames used by ALAM tokenizer training. The payload
is planned as uncompressed `data-*.tar` shards plus
`ARCHIVE_MANIFEST.sha256` and `DATASET_LAYOUT.json`; it is not meant to be read
directly from the tar files. The ALAM downloader verifies and reconstructs this
runtime layout:

```text
data/alam/calvin/{training,validation}/
```

Each split contains its `npz_metadata.json` and referenced NPZ files. Training
has 1,795,045 entries, validation has 99,022, and this snapshot has no separate
test split.

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact calvin_task_abc_d
```

Before making this dataset public, the publisher must verify CALVIN source-data
redistribution terms and add the appropriate license and upstream attribution.
