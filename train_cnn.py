## Set Home Directory (Modify as Needed)

home_directory = ''


# Import Libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import librosa
import librosa.display

import os
import time
import random

from PIL import Image

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastprogress import master_bar, progress_bar
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

USE_GPU = torch.cuda.is_available()

parent_directory = home_directory + 'UrbanSound8K/audio/'
numpy_save_directory = home_directory + 'UrbanSound8K/processed_np/'
model_save_directory = home_directory + 'models/'
    
sub_dirs = ['fold' + str(x) for x in np.arange(1,11)]


## Modeling

### Create Train and Test Dataset Classes

class TrainDataset(Dataset):
    def __init__(self, melspecs, labels, transforms):
        super().__init__()
        self.melspecs = melspecs
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.melspecs)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.melspecs[idx], mode='RGB')        
        image = self.transforms(image).div_(255)       
        label = self.labels[idx]
        
        return image, label


### Define Model Architecture and Parameters

transforms_dict = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512),
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x

model_params = {
    'num_epochs': 50, # increase after testing
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_clases': 10, 
    'eta_min': 1e-5,
    't_max': 10
}


# ### Set Random Seeds

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_GPU:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 73
set_seeds(SEED)


# ### Train Model

oof_labels = []
oof_preds = []

for sub_dir in sub_dirs:
    X_train = []
    train_labels = []
    X_valid = [] 
    valid_labels = [] 
    
    for fold in sub_dirs:
        file_list = os.listdir(numpy_save_directory + fold)
        if fold == sub_dir:
            for file_name in file_list:
                with np.load(numpy_save_directory + fold + '/' + file_name) as f:
                    X_valid.append(f['arr_0'])
                valid_labels.append(int(file_name.split('-')[1]))
                y_valid = np.zeros((len(valid_labels), 10)).astype(int)
                for i, j in enumerate(valid_labels):
                    y_valid[i, j] = 1
        else:
            for file_name in file_list:
                with np.load(numpy_save_directory + fold + '/' + file_name) as f:
                    X_train.append(f['arr_0'])
                train_labels.append(int(file_name.split('-')[1]))
                y_train = np.zeros((len(train_labels), 10)).astype(int)
                for i, j in enumerate(train_labels):
                    y_train[i, j] = 1

    train_dataset = TrainDataset(X_train, train_labels, transforms_dict['train'])
    valid_dataset = TrainDataset(X_valid, valid_labels, transforms_dict['train'])

    train_loader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=model_params['batch_size'], shuffle=False)
    
    if USE_GPU:
        model = Classifier().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        model = Classifier()
        criterion = nn.CrossEntropyLoss()
    
    optimizer = Adam(params=model.parameters(), lr=model_params['learning_rate'], amsgrad=False)
    scheduler = CosineAnnealingLR(optimizer, T_max=model_params['t_max'], eta_min=model_params['eta_min'])
    
    mb = master_bar(range(model_params['num_epochs']))
    
    for epoch in mb:
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for x_batch, y_batch in progress_bar(train_loader, parent=mb):
            if USE_GPU:
                preds = model(x_batch.cuda())
                loss = criterion(preds, y_batch.cuda())
            else:
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() / len(train_loader)

        model.eval()
        valid_preds = np.zeros((len(X_valid), 10))
        avg_val_loss = 0.

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            if USE_GPU:
                preds = model(x_batch.cuda()).detach()
                loss = criterion(preds, y_batch.cuda())
            else:
                preds = model(x_batch).detach()
                loss = criterion(preds, y_batch)

            preds = torch.sigmoid(preds)
            valid_preds[i * model_params['batch_size']: (i+1) * model_params['batch_size']] = preds.cpu().numpy()

            avg_val_loss += loss.item() / len(valid_loader)
            
        # valid_preds_tensor = torch.from_numpy(valid_preds)
        # y_valid_tensor = torch.from_numpy(y_valid).type_as(valid_preds_tensor)
        # epoch_val_loss = criterion(valid_preds_tensor, y_valid)
        accuracy = sum(1 for x,y in zip(valid_labels, valid_preds.argmax(axis=1).tolist()) if x == y) / len(valid_labels)    
            
        scheduler.step()
        
        elapsed = time.time() - start_time
        mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  accuracy: {accuracy:.4f}  time: {elapsed:.0f}s')
            
        if epoch == 0:
            best_accuracy = accuracy
            torch.save(model.state_dict(), home_directory + 'models/' + sub_dir + '_best_model.pt')
            best_preds = valid_preds.argmax(axis=1).tolist()
        else:
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), home_directory + 'models/' + sub_dir + '_best_model.pt')
                best_preds = valid_preds.argmax(axis=1).tolist()
    
    print(sub_dir + ' summary')
    print('------------------------')
    print(classification_report(np.argmax(valid_preds, axis=1), valid_labels))
    print('------------------------')
    print('best accuracy: ' + str(best_accuracy))
    print('------------------------')
    print('\n')

    oof_labels.append(valid_labels)
    oof_preds.append(best_preds)


oof_labels_flat = [item for sublist in oof_labels for item in sublist]
oof_preds_flat = [item for sublist in oof_preds for item in sublist]

oof_accuracy = sum(1 for x, y in zip(oof_labels_flat, oof_preds_flat) if x == y) / len(oof_labels_flat)

print('------------------------')
print('out-of-fold prediction accuracy: ' + str(oof_accuracy))


oof_accuracy = []
for i in range(len(oof_preds)):
    preds = oof_preds[i]
    labels = oof_labels[i]
    oof_accuracy.append(sum(1 for x, y in zip(labels, preds) if x == y) / len(labels))

print('Mean Out-of-Fold Prediction Accuracy: ' + str(np.mean(oof_accuracy)))
print('Out-of-Fold Prediction Accuracy Standard Deviation: ' + str(np.std(oof_accuracy)))


plt.figure(figsize=(18, 6))
sns.boxplot(oof_accuracy)
plt.title('UrbanSound8K Out-of-Fold Prediction Accuracy')
plt.savefig(home_directory + 'images/oof_pred_acc.png')
plt.show()