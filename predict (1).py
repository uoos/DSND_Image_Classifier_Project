import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
parser = argparse.ArgumentParser(
            prog='predict', 
            usage='prediction', 
            description='description', 
            epilog='end', 
            add_help=True, 
            )

parser.add_argument("input")
parser.add_argument("checkpoint")
parser.add_argument("--top_k", help='int', type=int)
parser.add_argument("--category_names")
parser.add_argument("--gpu", help='boolean', type=bool)

args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)
model = models.vgg16(pretrained = True)
model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['output_size_0']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checkpoint['output_size_0'], checkpoint['output_size_3']),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(checkpoint['output_size_3'], checkpoint['output_size_6']),
                                 nn.LogSoftmax(dim=1))
epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if args.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);

def process_image(image):
    im = Image.open(image)
    im = im.resize((224, 224))
    np_image = np.array(im) 
    np_image = np_image/255  
    np_image = (np_image - [0.485, 0.456, 0.406])/([0.229, 0.224, 0.225])
                                               
    np_image = np_image.transpose()                               
    return np_image


def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if args.top_k ==3:
        topk = 3
    else:
        topk = 5 
    model.eval()
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.float()
    if args.gpu == True:
        img = img.to(device)
    with torch.no_grad():
        output = model.forward(img)
        
    ps = torch.exp(output)
    probs, classes = ps.topk(topk, dim=1)
    probs = probs.tolist()
    probs = probs[0]
    classes = classes.tolist()
    classes = classes[0]
    classes= [str(n) for n in classes]
    
    return probs, classes

image_path = args.input
probs, classes = predict(image_path, model)

labels = []
for i in classes:
    label = cat_to_name[i]
    labels.append(label)

left = np.arange(len(probs))


if args.category_names == "cat_to_name.json":
    print(f"Flower name predictions: {labels}.. ")
    print(f"Probabilities: {probs}.. ")
else:
    print(f"Flower labels(#): {classes}.. ")
    print(f"Probabilities:: {probs}.. ")

#img = process_image(image_path)
#img = torch.from_numpy(img)
#img = img.float()
#image = img.numpy().transpose((1, 2, 0))
#mean = np.array([0.485, 0.456, 0.406])
#std = np.array([0.229, 0.224, 0.225])
#image = std * image + mean
#image = np.clip(image, 0, 1)

#if args.category_names == "cat_to_name.json":    
#    fig, ax = plt.subplots()
#    ax.imshow(image)
#    ax.set_title(labels[4])

#    plt.figure(figsize=(3,3))
#    plt.barh(left, probs)
#    plt.yticks(left, classes)
#    plt.show()
#else:
#    fig, ax = plt.subplots()
#    ax.imshow(image)
#    ax.set_title(labels[4])

#    plt.figure(figsize=(3,3))
#    plt.barh(left, probs)
#    plt.yticks(left, classes)
#    plt.show()
    
#python predict.py "flowers/train/102/image_08000.jpg" 'checkpoint.pth' --top_k 3 --category_names cat_to_name.json --gpu True