import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

# torch.cuda.is_available()
device = torch.device("cpu")

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'

    model_dir = os.path.join(os.path.expanduser('~/data'), model_name, model_name + '.pth')

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)
    net.load_state_dict(torch.load(model_dir))

    net = net.to(device)

    net.eval()

    # --------- 4. inference for each image ---------
    input_paths = ["Girl_in_front_of_a_green_background.jpg"]

    for path in input_paths:
        image = Image.open(path).convert("RGB")

        width, height = image.size

        # resize to prevent out of memory error
        scale = 500.0 / max(width, height)

        width = int(scale * width)
        height = int(scale * height)

        image = image.resize((width, height), Image.BOX)

        image0 = np.array(image) / 255.0

        image = torch.from_numpy(image0.transpose(2, 0, 1)[np.newaxis, ...]).float()

        image = image.to(device)

        print("running u2net")

        d1,d2,d3,d4,d5,d6,d7= net(Variable(image))

        pred = d1[:,0,:,:]

        del d1,d2,d3,d4,d5,d6,d7

        print("converting")

        # normalization
        pred = normPRED(pred)

        pred = pred.detach().cpu().numpy()
        pred = pred[0, :, :]

        is_foreground = pred > 0.95
        is_background = pred < 0.05

        from scipy.ndimage.morphology import binary_erosion

        size = 11
        structure = np.ones((size, size), dtype=np.int)
        is_foreground = binary_erosion(is_foreground, structure=structure)
        is_background = binary_erosion(is_background, structure=structure, border_value=1)

        trimap = 0.5 * (np.ones_like(pred) + is_foreground - is_background)

        from pymatting import estimate_foreground_ml, stack_images, save_image, estimate_alpha_lkm

        print("alpha matting")

        alpha = estimate_alpha_lkm(image0, trimap, laplacian_kwargs=dict(radius=10))

        print("foreground estimation")

        foreground = estimate_foreground_ml(image0, alpha)

        print("saving")

        cutout = stack_images(foreground, alpha)

        save_image("cutout.png", cutout)
        save_image("foreground.png", foreground)
        save_image("timap.png", trimap)
        save_image("pred.png", pred)
        save_image("alpha.png", alpha)

if __name__ == "__main__":
    main()
