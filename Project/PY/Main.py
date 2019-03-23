import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from timerForParsing import timeSince

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

trainData = np.load('npy_data/NPYData.npy')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.3)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size)).to(device)

def randomTrainingExample():
    i = random.randint(0, len(trainData) - 1)
    LEN = min(len(trainData[i]), 1600)
    r = random.randint(0, len(trainData[i]) - LEN - 2)
    if (LEN < 1600):
        r = 0
    line = trainData[i][r:r+LEN]
    inputs = Variable(torch.from_numpy(np.asarray(trainData[i][r:r+LEN])).long())
    targets = Variable(torch.from_numpy(np.asarray(trainData[i][r+1:r+LEN+1])).long())
    return inputs.to(device), targets.to(device)

def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.init_hidden()
    rnn.zero_grad()    
    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    loss.backward()
    optimizer.step()
    return output, loss.item() / input_line_tensor.size(0)

rnn = RNN(416, 1500, 416, 3).to(device)

all_losses = []

n_iters = 1200
print_every = 50
plot_every = 5
total_loss = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.001)

start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        percent = iter / n_iters * 100
        print(timeSince(start, percent / 100), ' — ', percent, '%', ' — ', loss)

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

plt.figure(figsize=(12,12))
plt.plot(all_losses)
plt.show()

torch.save(rnn.state_dict(), 'model.pt')