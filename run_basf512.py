import sys

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
from templates import *
from templates_latent import *
import torch

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    conf = basf512_autoenc()
    train(conf)
