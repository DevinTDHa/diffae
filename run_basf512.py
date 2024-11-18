import sys

sys.path.append("/home/tha/master-thesis-xai/thesis_utils")
from templates import *
from templates_latent import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run basf512 training")
    parser.add_argument(
        "--max_time", type=str, default="6:23:55:00", help="Maximum training time"
    )

    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    conf = basf512_autoenc()
    args = parser.parse_args()

    if args.max_time is not None:
        print("Running with max time", args.max_time)
    train(conf, max_time=args.max_time)
