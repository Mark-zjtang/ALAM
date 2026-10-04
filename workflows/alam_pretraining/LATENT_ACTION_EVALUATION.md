# Latent-action evaluation

We assess latent actions through algebraic consistency and image reconstruction:

- **Additivity:** compare a directly encoded transition with the sum of consecutive latent actions.
- **Reversibility:** measure how closely forward and reverse latent actions cancel each other.
- **Reconstruction:** compare decoded frames with target frames using PSNR, SSIM, and LPIPS.

Lower algebraic errors and LPIPS, and higher PSNR and SSIM, indicate better performance. Comparisons use matched video sequences, frame intervals, and preprocessing. Evaluation data are not bundled with this repository.

## Validation script

Entry point: [evaluate_validation.sh](evaluate_validation.sh). Implementation: [evaluate_validation.py](evaluate_validation.py).

After setting up the [ALAM environment](README.md#training), run from the repository root:

```bash
bash workflows/alam_pretraining/evaluate_validation.sh --gpu 0 \
  --checkpoint evaluation/checkpoints/alam/alam_pretrain_latent_action_tokenizer \
  --calvin-root data/alam/calvin --samples 100 \
  --output outputs/latent_action_validation/summary.json
```

Set `--checkpoint` to your ALAM v3 tokenizer directory (`config.yaml` and `pytorch_model.bin`) and `--calvin-root` to your prepared CALVIN data directory. Results are saved as JSON.

This script reports three-frame additivity/cycle MSE losses and reconstruction/perceptual losses. It does not output multi-step absolute errors or PSNR/SSIM/AlexNet-LPIPS scores.
