import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import multiprocessing
import itertools
import argparse

from sklearn.model_selection import KFold, train_test_split
from scipy import stats


parser = argparse.ArgumentParser(description='LSTM Training')
parser.add_argument('--device', type=int, 
                    help='GPU Device Number')
parser.add_argument('--start', type=int, 
                    help='Start Index')
parser.add_argument('--end', type=int, 
                    help='End Index')
parser.add_argument('--process', type=int, 
                    help='Number of Processes')


args = parser.parse_args()

devicenum = args.device
start = args.start
end = args.end
process = args.process


# load
with open(f'../usedata/train_final_all.pickle', 'rb') as f:
    finaldata = pickle.load(f)

for k, v in finaldata.items():
    lstm_input = v.shape[1] - 4
    break

keys = list(finaldata.keys())
lipids = ['L3008_cholesterol', 'L3061_tg', 'L3062_hdl', 'L3068_ldl']

# print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
device = f"cuda:{str(devicenum)}" if torch.cuda.is_available() else "cpu"


class LipidDataset(Dataset):

    def __init__(self, mode='train', finaldata=finaldata, trainid=None,
                 mean=None, std=None, lipids=lipids, predict='L3008_cholesterol'):
        self.finaldata = finaldata
        self.trainid = trainid
        self.lipids = lipids
        self.predict = predict

        if mode == 'train':
            assert (mean is None) and (std is None), \
                "평균과 분산은 train 폴더의 있는 데이터로 구하기 때문에 None 으로 설정합니다."
            alltrain = np.vstack([np.array(v) for k, v in finaldata.items() if k in trainid])
            self.mean = np.array([np.mean(alltrain[:, i]) for i in range(alltrain.shape[1])])
            self.std = np.array([np.std(alltrain[:, i]) for i in range(alltrain.shape[1])])
        elif mode == 'test':
            assert (mean is not None) and (std is not None), \
                "평균과 분산은 `train_data`변수에 내장한 self.mean 과 self.std 를 사용합니다."
            self.mean = mean
            self.std = std
        else:
            print('Neither train nor test')

    def __len__(self):
        return len(self.trainid)

    def __getitem__(self, idx):
        self.data = self.finaldata[self.trainid[idx]]
        cols = self.data.columns
        self.data = np.array([(i - self.mean) / self.std for i in np.array(self.data)])
        self.data = pd.DataFrame(self.data, columns=cols)

        target = [np.array(self.data[self.predict])[-1]]
        sample = np.array(self.data[self.data.columns.difference(self.lipids)])
        #         sample = np.array([(i - self.mean) / self.std for i in sample])

        return torch.Tensor(sample), torch.Tensor(target)


def train(num_epochs, model, data_loader, val_loader, patience,
          criterion, optimizer, saved_dir, device):
    #     print('Start training..')
    best_loss = 9999999
    model.train()
    for epoch in range(num_epochs):
        if epoch == 0:
            early_stopping = EarlyStopping(patience=patience, path=saved_dir, verbose=False)
        else:
            early_stopping = EarlyStopping(patience=patience, best_score=best_score,
                                           counter=counter, path=saved_dir, verbose=False)

        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)

            outputs = model(sequence)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #             if (step + 1) % 10 == 0:
        #                 print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
        #                     epoch+1, num_epochs, step+1, len(data_loader), loss.item()))

        avrg_loss, rsq = validation(model, val_loader, criterion, device)
        best_score, counter, finish = early_stopping(avrg_loss, model)

        if finish:
            model.load_state_dict(torch.load(saved_dir))
            model.eval()
            avrg_loss, b = validation(model, val_loader, criterion, device)
            break

    return best_score, rsq


def validation(model, data_loader, criterion, device):
    b = []
    model.eval()
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        for step, (sequence, target) in enumerate(data_loader):
            sequence = sequence.type(torch.float32)
            target = target.type(torch.float32)
            sequence, target = sequence.to(device), target.to(device)
            outputs = model(sequence)
            loss = criterion(outputs, target)
            b.append((outputs, target))
            total_loss += loss
            cnt += 1
        avrg_loss = total_loss / cnt
    #         print('Validation Average Loss: {:.4f}'.format(avrg_loss))
    model.train()
    return avrg_loss, b


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, best_score=np.inf, counter=0, delta=0,
                 path=None, verbose=False):

        self.patience = patience
        self.verbose = verbose
        self.counter = counter
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
#                 print('Early Stopping Validated')
                self.early_stop = True

        else:
            self.save_checkpoint(val_loss, model)
            self.best_score = val_loss
            self.counter = 0

        return self.best_score, self.counter, self.early_stop

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if os.path.isfile(self.path):
            os.remove(self.path)
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


def rsquare(a, b):
    s, i, r, p, std = stats.linregress(a, b)
    return r ** 2


def trainsave(vars):
    batch_size, learning_rate, num_layers, hidden_size, predict = vars
    
    for pth in os.listdir(f'../results/vars/'):
        if f'../results/vars/ckpt_{predict}_batch_{batch_size}_lr_{learning_rate}_layers_{num_layers}_hidden_{hidden_size}' in pth:
            return False

    # Base Parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=777)
    patience = 50
    batch_size = batch_size
    num_epochs = 500
    learning_rate = learning_rate

    # Best Scores
    bs_box = []
    rsq_box = []
    
    for tid, vid in kf.split(keys):
        output_box = []
        target_box = []

        # Dataloader 구축; 5-fold validation
        trainid = [keys[i] for i in tid]
        valid = [keys[i] for i in vid]
        
        alltrain = np.vstack([np.array(v) for k, v in finaldata.items() if k in trainid])
        lipid_train = LipidDataset(mode='train', trainid=trainid, predict=predict)
        lipid_val = LipidDataset(mode='test', trainid=valid, predict=predict, mean=lipid_train.mean, std=lipid_train.std)
        train_loader = DataLoader(lipid_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(lipid_val, batch_size=batch_size, shuffle=False)

        #         print(f'\nnum_layers: {num_layers}\nhidden_size: {hidden_size}\n')

        # Training
        torch.manual_seed(7777)
        model = LSTM(num_layers=num_layers, hidden_size=hidden_size)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        saved_dir = f'../results/saved/ckpt_{predict}_batch_{batch_size}_lr_{learning_rate}_layers_{num_layers}_hidden_{hidden_size}.pt'

        _, rsq = train(num_epochs, model, train_loader, val_loader, patience,
                                criterion, optimizer, saved_dir, device)

        for r in rsq:
            output_box = torch.cat([torch.Tensor(output_box), torch.Tensor.cpu(r[0])])
            target_box = torch.cat([torch.Tensor(target_box), torch.Tensor.cpu(r[1])])
        
        bs_box.append(criterion(output_box, target_box).numpy())
        rsq_box.append(rsquare(output_box.numpy().reshape(-1), target_box.numpy().reshape(-1)))

        break

    MSE = np.mean(bs_box)
    RSQ = np.mean(rsq_box)
    print(f'ckpt path: {saved_dir}\nBest MSE: {round(MSE, 3)}\nBest RSQ: {round(RSQ, 3)}')
    f = open(f'../results/vars/ckpt_{predict}_batch_{batch_size}_lr_{learning_rate}_layers_{num_layers}_hidden_{hidden_size} MSE {round(MSE, 3)} RSQ {round(RSQ, 3)}.txt', 'w')
    
    for i in rsq_box:
        data = str(i) + "\n"
        f.write(data)
    f.close()


b = [1] # batch_size
l = [0.001, 0.0003] # learning_rate
n = [2, 3, 4] # num_layers
h = [(i)*100 for i in range(4, 7)] # hidden_size

vars = list(itertools.product(b, l, n, h, lipids))


if __name__ == '__main__':
    
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=process)
    pool.map(trainsave, vars[start:end])
    pool.close()
    pool.join()






