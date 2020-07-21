# Usage - python generate_faces.py ((number of faces to generate)) ((filename to save image with))

from models import Generator , sampler_noise , show_imgs
from data_processing import denormalize
import torch
import sys , time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

NOISE_DIM = 96

def savegrid(ims, save_name , fill=True, showax=False):
    rows = int(np.ceil(np.sqrt(imgs.shape[0])))
    cols = int(np.ceil(np.sqrt(imgs.shape[0])))

    gridspec_kw = {'wspace': 0, 'hspace': 0} if fill else {}
    fig,axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw)

    if fill:
        bleed = 0
        fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax,im in zip(axarr.ravel(), ims):
        ax.imshow(im)
        if not showax:
            ax.set_axis_off()

    kwargs = {'pad_inches': .01} if fill else {}
    fig.savefig('{}.png'.format(save_name), **kwargs)

def imgs_to_array(imgs , name_):
    img_arr = []
    for i,img in enumerate(imgs):
        img_arr.append(denormalize(np.transpose(img.numpy() , (1,2,0))))
    return np.array(img_arr)

start = time.time()
print("Generating Images...")

Gt = Generator(NOISE_DIM , 64 , 3 , extra_layers=1)
Gt.load_state_dict(torch.load("finalG.pt"))
Gt.eval()

N_generate = int(sys.argv[1])
name = sys.argv[2]

noise = sampler_noise(N_generate , NOISE_DIM)
generated_imgs = Gt(noise).detach()

show_imgs(generated_imgs)
imgs = imgs_to_array(generated_imgs , name)
savegrid(imgs , name)

print("Finished all tasks in {:.4} seconds".format(time.time() - start))