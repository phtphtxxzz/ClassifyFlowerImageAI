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

    parser.add_argument('data_directory', type=str, default='/flowers', help='path to data file')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='path to pth file')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN Model Architecture for classify image with default is vgg16')
    parser.add_argument('--learning_rate', type=float, default='0.003', help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='hidden units')
    parser.add_argument('--epochs', type=int, default=15, help='epochs')
    parser.add_argument('--gpu', action='store_true',default=True, help='using GPU')

    return parser.parse_args()

def train(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu)
    data_dir = data_directory
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(test_dir, transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 64)

    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('fc2', nn.Linear(hidden_units, 1024)),
                            ('relu2', nn.ReLU()),
                            ('fc3', nn.Linear(1024, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device);

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 50

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(valid_loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'architecture': arch,
        'classifier': model.classifier,  
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),  
        'epochs': epochs  
    }

    torch.save(checkpoint, save_dir)
        

def main():
    
    in_arg = get_input_args()
    train(in_arg.data_directory, in_arg.save_dir, in_arg.arch, in_arg.learning_rate, in_arg.hidden_units, in_arg.epochs, in_arg.gpu)
    
    

# Call to main function to run the program
if __name__ == "__main__":
    main()
