import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataset import Subset
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import argparse
 
parser = argparse.ArgumentParser(
            prog='argparseTest', 
            usage='Demonstration of argparser', 
            description='description', 
            epilog='end', 
            add_help=True, 
            )

parser.add_argument("data_directory")
parser.add_argument("--save_dir")
parser.add_argument("--arch")
parser.add_argument("--learning_rate", help='float', type=float)
parser.add_argument("--hidden_units",help='integer',
                    type=int)
parser.add_argument("--epochs",help='integer',
                    type=int)
parser.add_argument("--gpu", help='boolean', type=bool)

args = parser.parse_args()

data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]) 

valid_test_transforms = transforms.Compose([transforms.Resize(225),
                                            transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

#total_len = len(args.data_directory)
#train_len = int(len(args.data_directory)*0.7)
#train_dir = Subset(args.data_directory,  list(range(0,train_len)))
#valid_dir = Subset(args.data_directory,  list(range(train_len,total_len)))
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'

image_datasets=datasets.ImageFolder(train_dir,transform=data_transforms)
valid_dataset=datasets.ImageFolder(valid_dir, transform= valid_test_transforms)

#total_len = len(image_datasets)
#train_size = int(len(image_datasets)*0.7)
#val_size = total_len - train_size
#train_dataset = Subset(image_datasets,  list(range(0,train_size)))
#valid_dataset = Subset(valid_datasets,  list(range(train_size,total_len)))
#train_dataset, valid_dataset = torch.utils.data.random_split(image_datasets, [train_size, val_size])

dataloader = torch.utils.data.DataLoader(image_datasets,batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)

    
if args.arch == "vgg13":
    model = models.vgg13(pretrained = True)
else:
    model = models.vgg16(pretrained = True)

if args.gpu == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
for param in model.parameters():
    param.requires_grad = False

if args.hidden_units == 512:
    classifier = nn.Sequential(nn.Linear(25088, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))

else:
    classifier = nn.Sequential(nn.Linear(25088, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))

model.classifier = classifier
criterion = nn.NLLLoss()
    
if args.learning_rate == 0.001:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)
    
if args.gpu == True:
    model.to(device);

if args.epochs == 20:
    epochs = 20
else:
    epochs = 2
    
steps = 0
print_every = 10
running_loss = 0

for e in range(epochs):
        for images, labels in dataloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
        
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
if args.save_dir == None:
    pass
else:
    if args.hidden_units == 512:
        model.class_to_idx = image_datasets.class_to_idx
        checkpoint = {'input_size': 25088,
                  'output_size_0': 512,
                  'output_size_3': 512,
                  'output_size_6': 102,
                  'epoch':epochs,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    else:
        model.class_to_idx = image_datasets.class_to_idx
        checkpoint = {'input_size': 25088,
                  'output_size_0': 512,
                  'output_size_3': 256,
                  'output_size_6': 102,
                  'epoch':epochs,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, args.save_dir)

  #python train.py 'flowers' --save_dir 'checkpoint.pth' --learning_rate 0.01 --gpu True
