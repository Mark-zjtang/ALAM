# Latent-action evaluation

We assess latent actions through algebraic consistency and image reconstruction:

- **Additivity:** compare a directly encoded transition with the sum of consecutive latent actions.
- **Reversibility:** measure how closely forward and reverse latent actions cancel each other.
- **Reconstruction:** compare decoded frames with target frames using PSNR, SSIM, and LPIPS.

Lower algebraic errors and LPIPS, and higher PSNR and SSIM, indicate better performance. Comparisons use matched video sequences, frame intervals, and preprocessing. Evaluation data are not bundled with this repository.
