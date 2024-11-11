from templates import *
from templates_latent import *

if __name__ == "__main__":
    # do run the run_basf512 before using the file to train the latent DPM

    # infer the latents for training the latent DPM
    # NOTE: not gpu heavy, but more gpus can be of use!
    gpus = [0]
    # gpus = [0, 1, 2, 3]
    conf = basf512_autoenc()
    conf.eval_programs = ["infer"]
    train(conf, gpus=gpus, mode="eval")  # DHA: Assume pretrained. The model is loaded in eval mode.

    # train the latent DPM
    # NOTE: only need a single gpu
    gpus = [0]
    conf = basf512_autoenc_latent()
    train(conf, gpus=gpus)
