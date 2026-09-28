# Models and publication

Install the publishing environment and download **models only** from the repository root:

```bash
bash workflows/install/install_environments.sh --components publish
bash workflows/publishing/download_release.sh --models-only
```

| Model repository | Contents |
| --- | --- |
| [alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) | MetaWorld and LIBERO ALAM tokenizers |
| [alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) | MetaWorld policy |
| [alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) | LIBERO policy |

The [manifest](huggingface_manifest.json) records local paths and hashes. The model-only release does not upload datasets; obtain them from the [upstream sources](../../README.md#dataset-sources).

For maintainers: `upload_models.sh --dry-run` previews the Hugging Face model upload; `upload_github.sh --dry-run` checks the code push. Neither script stores credentials in the repository.
