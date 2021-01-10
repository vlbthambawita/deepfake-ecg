import torch
import numpy as np
from tqdm import tqdm


def generate(num_of_sample, out_dir, start_id=0, run_device="cuda"):

    device = torch.device(run_device)

    netG = torch.load("/home/vajira/DL/fake_ecg_data_RHTM/best_G_checkpoint_2500_from_006/g.pt", map_location="cpu")
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