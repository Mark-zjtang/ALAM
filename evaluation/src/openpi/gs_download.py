
from shared import download

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/")
print("checkpoint_dir", checkpoint_dir)

# Create a trained policy.

