import torch.optim as optim
from torch.nn import Module
from typing import List


def create_optimizers(generator: Module, discriminator: Module, lr: float = 0.0003, betas: List = None):
    """ This function returns the optimizers of the generator and the discriminator """

    if betas is None:
        betas = [0.5, 0.999]

    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    return g_optimizer, d_optimizer


# TODO: Create training function
def run_training():
    return


if __name__ == "__main__":
    run_training()
