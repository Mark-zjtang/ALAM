# Latent-action evaluation

Latent-action evaluation measures additivity, reversibility, and image reconstruction. These diagnostics are separate from robot-policy success rates.

## Run the released validation workflow

Install the [ALAM environment](README.md#training), download an ALAM tokenizer, and prepare the [CALVIN validation data](README.md#data-mixture). Unlike policy evaluation, this diagnostic needs video data.

From the repository root:

```bash
bash workflows/alam_pretraining/evaluate_validation.sh \
  --gpu 0 \
  --checkpoint evaluation/checkpoints/alam/metaworld_epoch19_step58216 \
  --calvin-root data/alam/calvin \
  --samples 100 \
  --output outputs/latent_action_validation/summary.json
```

Replace `--checkpoint` with your tokenizer directory containing `config.yaml` and `pytorch_model.bin`. The released loader expects the ALAM v3 architecture and 458 state-dictionary entries. Use `evaluation/checkpoints/alam/libero_epoch16_step49024` for the other released tokenizer.

The [shell entrypoint](evaluate_validation.sh) calls [evaluate_validation.py](evaluate_validation.py), which restores the checkpoint strictly and invokes the [preserved trainer's `calculate_loss`](../../Algebraic_latent_action_model/latent_action_model/trainers/ctl_latent_action_tokenizer_trainer.py). It runs inference without updating weights.

### Sampling and outputs

- CALVIN validation triplets use a fixed frame interval of 10 and evaluation preprocessing (`train=False`).
- The script scans sample indices in order, skips specified data-loading failures within a bounded scan, and records the exact `evaluated_indices`.
- `--samples` defaults to 10; it is a sample count, not a full-validation-set evaluation.
- The JSON records checkpoint/data paths, sample count, indices, and `mean_losses`. `PASS` means the requested samples produced finite losses, not that a performance threshold was reached.

Let `a_ij` denote the continuous physical latent action encoded from frames `i` and `j`. The [model implementation](../../Algebraic_latent_action_model/latent_action_model/models/ctl_latent_action_tokenizer_v3.py) returns:

| JSON field | Definition |
| --- | --- |
| `add_loss` | Weighted mean squared error between `a_02` and `a_01 + a_12` |
| `cyc_loss` | Weighted mean squared error between `a_01 + a_10` and zero |
| `avg_recons_loss` | Weighted mean reconstruction loss for transitions `0→1`, `1→2`, and `0→2`; MSE or L1 according to the checkpoint config |
| `avg_perceptual_loss` | Weighted mean LPIPS reconstruction loss for the same three transitions, using the model's VGG network |

Weights come from the checkpoint's `loss_config`. Compare losses only with matching weights, preprocessing, data, and frame intervals. The JSON also includes auxiliary training-loss fields; the table lists the terms relevant to these diagnostics.

## Additivity and reversibility errors

For a six-frame sequence `o_0, o_k, …, o_5k`, define the following quantities at horizons `n = 1, …, 5`:

```text
d_n = encode(o_0, o_nk)                    # direct latent action
s_n = sum(encode(o_(j-1)k, o_jk), j=1..n)   # cumulative latent action
r_n = encode(o_nk, o_0)                    # reverse latent action

additivity error:   abs(d_n - s_n)
reversibility error: abs(d_n + r_n)
```

Errors are computed on continuous `phys_a_01` outputs, before re-quantizing the cumulative action for image decoding. They are **absolute errors, not MSE**, and are not normalized by latent magnitude or variance.

For each horizon, report mean, standard deviation (`torch.std` with its default sample correction), maximum, and per-coordinate mean. Reduce the overall mean/std/max over all batch and latent elements, and the per-coordinate mean over the batch. Lower errors are better. One-step additivity is a sanity check; horizons of two or more steps test composition.

## Reconstruction metrics

To assess reconstruction, compare the predicted image with the target at each horizon. The direct and cumulative routes are defined as:

| Route | Decoder input | Target |
| --- | --- | --- |
| Direct | Codes from `d_n` | `o_nk` |
| Cumulative | Codes obtained by re-quantizing `s_n` once | `o_nk` |

Both routes use the same anchor image `o_0`. Clamp predicted and target images to `[0, 1]` before calculating metrics.

| Metric | Calculation | Better |
| --- | --- | --- |
| PSNR | `10 * log10(1 / MSE)` with image range `[0, 1]` | Higher |
| SSIM | TorchMetrics structural similarity with `data_range=1.0` | Higher |
| LPIPS | TorchMetrics LPIPS with AlexNet, mapping images to `[-1, 1]` | Lower |

Report each horizon and reconstruction route separately, together with the sample count and aggregation rule. AlexNet LPIPS is distinct from the VGG perceptual training loss returned by the validation workflow.

## Workflow coverage

The command above computes three-frame validation losses. The multi-step absolute errors and PSNR/SSIM/AlexNet-LPIPS definitions describe additional diagnostics; they are not outputs of `evaluate_validation.sh`. Keep checkpoint, frames, preprocessing, and frame intervals matched when comparing models.
