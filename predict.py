# Check torch version and CUDA status if GPU is enabled.
import torch
# Imports here

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
from PIL import Image
import numpy as np
import seaborn as sns
import json
import argparse

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path', type=str, default='flowers/test/37/image_03734.jpg',
                        help='path to image')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='link to checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top k most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='path to list categories')
    parser.add_argument('--gpu', action='store_true',default=True, help='using GPU')

    return parser.parse_args()



def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    if img.width < img.height:
        imgSize = 256, img.height
    else:
        imgSize = img.width, 256
    print(imgSize)
    img.thumbnail(imgSize)
    centerX = img.width/2
    centerY = img.height/2
    img = img.crop((centerX - 244/2, centerY - 244/2, centerX + 244/2, centerY + 244/2))

    np_image = np.array(img)/255.0
    np_image = (np_image - means)/stds

    np_image = np_image.transpose((2, 0, 1))

    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, category_names, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    img = process_image(image_path)
    cvt_img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).to(device)
    cvt_img = cvt_img.to(device)
    with torch.no_grad():
        ps = torch.exp(model.forward(cvt_img))
    topP, topC = ps.topk(top_k, dim=1)
    topP = topP.cpu().numpy().squeeze()
    topC = topC.cpu().numpy().squeeze()

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    topC = [idx_to_class[i] for i in topC]
    topFlower = [cat_to_name[l] for l in topC]
    return topP, topC, topFlower


def main():
    
    in_arg = get_input_args()
    checkpoint = torch.load(in_arg.checkpoint)
    print(checkpoint['architecture'])
    if checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    topP, topC, topFlower = predict(in_arg.image_path, model, in_arg.category_names, in_arg.top_k, in_arg.gpu) 
    plt.subplot(2,1,2)
    sns.barplot(x=topP, y=topC, color=sns.color_palette()[0]);
    plt.show()
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
