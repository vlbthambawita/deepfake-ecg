import torch
import os
import numpy as np
from tqdm import tqdm
from . import Generator
from pathlib import Path
from typing import Union, Literal


def generate(num_of_sample: int, out_dir: Union[str, Path], start_id: int = 0, run_device: Literal["cpu", "cuda"] = "cpu") -> None:
    """Generate multiple 8-lead ECG waveforms and save them as ASCII files

    Args:
        num_of_sample (int): Number of ECG samples to generate
        out_dir (Union[str, Path]): Output directory path where files will be saved
        start_id (int): Starting ID for the generated samples
        run_device (Literal["cpu", "cuda"]): Device to run generation on ("cpu" or "cuda")

    Returns:
        None: Files are saved to the specified output directory with names {start_id}.asc to {start_id + num_of_sample - 1}.asc
             Each file contains ECG data in ASCII format with shape (5000, 8) for leads [I, II, V1, V2, V3, V4, V5, V6]
    """
    root_dir = Path(__file__).parent

    device = torch.device(run_device)

    netG = Generator()
    checkpoint = torch.load(
        os.path.join(root_dir, "checkpoints/g_stat.pt"),
        map_location=device,
        weights_only=True
    )
    netG.load_state_dict(checkpoint["stat_dict"])
    netG.to(device)
    netG.eval()

    for i in tqdm(range(start_id, start_id + num_of_sample)):
        noise = torch.Tensor(1, 8, 5000).uniform_(-1, 1)
        noise = noise.to(device)
        out = netG(noise)
        out_rescaled = out * 6000
        out_rescaled = out_rescaled.int()

        out_rescaled_t = torch.t(out_rescaled.squeeze())

        # asc_file = open("{}/{}.asc".format(asc_dir, i), 'ab')
        np.savetxt("{}/{}.asc".format(out_dir, str(i)), out_rescaled_t.detach().cpu().numpy(), fmt='%i')
        # asc_file.close()


def generate_as_numpy(run_device="cpu"):
    """Generate a single 8-lead ECG waveform using deepfakeecg model

    Args:
        run_device (str): Device to run generation on ("cpu" or "cuda")

    Returns:
        numpy.ndarray: Array of shape (5000, 8) containing the ECG data
                      for leads [I, II, V1, V2, V3, V4, V5, V6]
    """
    root_dir = Path(__file__).parent
    device = torch.device(run_device)

    netG = Generator()
    checkpoint = torch.load(
        os.path.join(root_dir, "checkpoints/g_stat.pt"),
        map_location=device,
        weights_only=True
    )
    netG.load_state_dict(checkpoint["stat_dict"])
    netG.to(device)
    netG.eval()

    noise = torch.Tensor(1, 8, 5000).uniform_(-1, 1)
    noise = noise.to(device)
    out = netG(noise)
    out_rescaled = out * 6000
    out_rescaled = out_rescaled.int()

    # Convert to numpy and transpose to get (5000, 8) shape
    data = out_rescaled.squeeze().t().detach().cpu().numpy()

    return data
