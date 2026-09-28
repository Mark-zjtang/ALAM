# Model downloads and GitHub code publishing

From the repository root, install the publishing environment and download **models only**:

```bash
bash workflows/install/install_environments.sh --components publish
bash workflows/publishing/download_release.sh --models-only
```

| Model repository | Contents |
| --- | --- |
| [alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) | ALAM tokenizers pretrained on Mix-11: 10 OXE video sources + CALVIN |
| [alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) | π0 policy post-trained on MetaWorld demonstrations with frozen ALAM |
| [alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) | π0 policy post-trained on LIBERO demonstrations with frozen ALAM |

The [download manifest](huggingface_manifest.json) records model paths; file hashes are in the [weights manifest](../../evaluation/WEIGHTS_MANIFEST.sha256). Dataset links are in the [main README](../../README.md). Neither datasets nor model binaries are bundled with GitHub.

## Maintainer: push code to GitHub

Commit your changes first, then run:

```bash
bash workflows/publishing/upload_github.sh --dry-run
bash workflows/publishing/upload_github.sh
```

If Git asks for a username, enter your GitHub username. At the **Password** prompt, enter a GitHub personal access token with repository write access, **not** your account password. The script pushes to `Mark-zjtang/ALAM` without force-pushing.

The script fetches remote `main` before checking commit history. If GitHub has new changes, follow the synchronization command it prints and rerun the script.

Do not put your GitHub token in a command, URL, `.env`, README, or chat. The code-push script does not write it to this repository or a credential helper.
