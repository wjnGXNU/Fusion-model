import PySide2
from matplotlib import pyplot as plt
import numpy as np
import random
import torch
import os
from sklearn.metrics import confusion_matrix
from EarlyStopping import EarlyStopping
from dataset_npy import BasicDataset
from fusion_trans import TransFuse_S
dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
from tabulate import tabulate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
enable_amp = True if "cuda" in device.type else False
learning_rate = 1e-4
training_epochs = 500
batch_size = 32
mitdb_train = BasicDataset(root_dir='G:\\Pycharm Community 2021.3\\PycharmProject\\ECG-classification-by-CNN-main\\data', train=True)
mitdb_test = BasicDataset(
    root_dir='G:\\Pycharm Community 2021.3\\PycharmProject\\ECG-classification-by-CNN-main\\data', train=False)
valid_size = 0.2
from torch.utils.data.sampler import SubsetRandomSampler
num_train = len(mitdb_train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(mitdb_train,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=0,
                                           drop_last=True)
valid_loader = torch.utils.data.DataLoader(mitdb_train,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=0,
                                           drop_last=True)
test_loader = torch.utils.data.DataLoader(mitdb_test,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          shuffle=True,
                                          drop_last=True)
model = TransFuse_S(pretrained=True).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
avg_loss = []
train_acc = []
test_acc = []
def train_model(model, batch_size, patience, n_epochs):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(1, n_epochs + 1):
        torch.cuda.empty_cache()
        if epoch % 5 == 0 and epoch != 0:
            for params in optimizer.param_groups:
                params['lr'] *= 0.9
        model.train()
        for batch, (data, target,RR) in enumerate(tqdm(train_loader), 1):
            data = np.expand_dims(data, axis=1)
            data = torch.tensor(data)
            optimizer.zero_grad()
            data = data.float()
            target = torch.topk(target, 1)[1].squeeze(1)
            data = data.to(device)
            target = target.to(device)
            RR = RR.to(device)
            output = model(data, target, RR)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        for data, target,RR in valid_loader:
            data = np.expand_dims(data, axis=1)
            data = torch.tensor(data)
            data = data.float()
            target = torch.topk(target, 1)[1].squeeze(1)
            data = data.to(device)
            target = target.to(device)
            RR = RR.to(device)
            output = model(data, target,RR)
            loss = criterion(output, target)
            valid_losses.append(loss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)
        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load('checkpoint.pth'))
    del loss,  output
    return model, avg_train_losses, avg_valid_losses
patience = 10
model, train_loss, valid_loss = train_model(model, batch_size, patience, training_epochs)
accuracy = 0
labels = []
guesses = []
y_true_list = []
y_pred_list = []
with torch.no_grad():
    for X_test, Y_test, RR in tqdm(test_loader):
        X_test = np.expand_dims(X_test, axis=1)
        X_test = torch.tensor(X_test)
        X_test = X_test.float().to(device)
        Y_test = Y_test.long().to(device)
        RR = RR.to(device)
        Y_test = torch.topk(Y_test, 1)[1].squeeze(1)
        model = torch.load('finish_model.pkl')
        model.eval()
        model = model.to(device)
        prediction = model(X_test,Y_test,RR)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        labels.extend(Y_test.tolist())
        guesses.extend(torch.argmax(prediction, 1).tolist())
        y_true_list.extend(Y_test)
        y_pred_list.extend(torch.argmax(prediction, 1).tolist())
        accuracy += correct_prediction.float().sum()
print('Accuracy:', accuracy.item() / len(test_loader.dataset))