import torch
import os
import numpy as np
from tqdm import tqdm
from . import Generator
from pathlib import Path





def generate(num_of_sample, out_dir, start_id=0, run_device="cpu"):

    # Testing
    root_dir = Path(__file__).parent

    device = torch.device(run_device)

    netG = Generator()
    checkpoint= torch.load(os.path.join(root_dir, "checkpoints/g_stat.pt"), map_location=device)
    netG.load_state_dict(checkpoint["stat_dict"])
    netG.to(device)
    netG.eval()
   

    for i in tqdm(range(start_id, start_id  + num_of_sample)):
        noise = torch.Tensor(1, 8, 5000).uniform_(-1, 1)
        noise = noise.to(device)
        out = netG(noise)
        out_rescaled = out*6000
        out_rescaled =out_rescaled.int()
        
       
        out_rescaled_t = torch.t(out_rescaled.squeeze())

        #asc_file = open("{}/{}.asc".format(asc_dir, i), 'ab')
        np.savetxt("{}/{}.asc".format(out_dir,str(i)), out_rescaled_t.detach().cpu().numpy(), fmt='%i')
        #asc_file.close()


