# Latent-action evaluation

Latent-action diagnostics measure algebraic consistency and image reconstruction, not robot-policy success. This guide distinguishes the original multi-step evaluator from the validation-loss workflow included in this release.

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
| `avg_commit_loss` | Weighted vector-quantizer commitment loss |
| `loss` | Sum of the weighted loss terms above |
| `active_code_num` | Number of distinct codes used across the three forward transitions; a usage diagnostic, not an error |

Weights come from the checkpoint's `loss_config`. Compare losses only with matching weights, preprocessing, data, and frame intervals. Smaller algebraic errors alone do not rule out collapsed representations; inspect reconstruction and code usage as well.

## Original multi-step error evaluator

The original development-tree entrypoints were `run_local_eval.sh` and `run_dsw_eval_alam.sh`. Both invoked `train_phylam_mix.py`, whose default configuration was `configs/lam/ctl_v3_lam_calvin_mix_data_add_eval.yaml`.

The metric implementation was `LatentActionTokenizerTrainer.eval_latent_motion_reconstruction` in `uni_world_model/latent_action_model/trainers/ctl_latent_action_tokenizer_trainer_eval.py`; image metrics came from `calculate_psl_metrics_dict` in the adjacent `trainer_utils.py`.

These legacy entrypoints are not bundled as runnable public evaluation commands. The inspected version enters a training loop and performs an optimizer update before its first step-zero evaluation. Do not use it as an inference-only checkpoint evaluator without separating that path.

### Multi-step algebraic errors

The evaluator consumes six frames, `o_0, o_k, …, o_5k`, and evaluates horizons `n = 1, …, 5`. Define:

```text
d_n = encode(o_0, o_nk)                    # direct latent action
s_n = sum(encode(o_(j-1)k, o_jk), j=1..n)   # cumulative latent action
r_n = encode(o_nk, o_0)                    # reverse latent action

additivity error:   abs(d_n - s_n)
reversibility error: abs(d_n + r_n)
```

Errors are computed on continuous `phys_a_01` outputs, before re-quantizing the cumulative action for image decoding. They are **absolute errors, not MSE**, and are not normalized by latent magnitude or variance.

For each horizon, the script reports mean, standard deviation (`torch.std` with its default sample correction), maximum, and per-coordinate mean. The overall mean/std/max reduce over all batch and latent elements; the per-coordinate mean reduces over the batch. These describe one sampled evaluation batch, not a dataset-wide aggregate. The one-step additivity check is a consistency sanity check rather than a multi-step composition test.

### Reconstruction quality

Both routes decode from the same anchor image `o_0`:

| Route | Decoder input | Target |
| --- | --- | --- |
| Direct | Codes from `d_n` | `o_nk` |
| Cumulative (intended) | Codes obtained by re-quantizing `s_n` | `o_nk` |

**Source-audit caveat:** for `n >= 2`, the inspected trainer calls `decode_additivity_lam(accumulated, accumulated)`, while the corresponding model adds its two arguments. This actually quantizes `2 * s_n`, not `s_n`. The algebraic error calculation still uses `s_n` and is unaffected by that decoding mismatch. The one-step route uses the original single-step codes. This finding applies to the inspected development-tree snapshot; it does not establish which version produced any paper result.

The historical helper clamps images to `[0, 1]` and uses TorchMetrics PSNR and SSIM with `data_range=1.0`. Its LPIPS metric uses **AlexNet**, with images mapped to `[-1, 1]`. Higher PSNR/SSIM and lower LPIPS are better. This LPIPS value is not interchangeable with the VGG perceptual training loss above.

Results were appended to `latent_motion_metrics.log` in the evaluation's visualization directory, alongside `*-direct.png` and `*-accum.png` reconstructions. The log contains per-horizon metrics and summaries, including changes relative to the one-step reconstruction baseline.

### Comparison boundary

The inspected legacy configuration enables random-shift preprocessing (`train=True`), shuffled evaluation loading, and six-frame dataset loaders. Its configured base intervals are 5 for OXE and 10 for CALVIN. Exact comparisons require the same checkpoint, sampled frames, effective intervals, preprocessing, and metric implementation.

The released `evaluate_validation.sh` uses three-frame CALVIN samples and reports weighted training losses. It **does not reproduce the legacy five-horizon absolute-error or PSNR/SSIM/LPIPS tables**. The historical formulas above document the inspected source; no new multi-step results are claimed here.
