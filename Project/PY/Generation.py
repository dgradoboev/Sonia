import numpy as np
import random
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import mido

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(device)

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
    
rnn = RNN(416, 1500, 416, 3).to(device)
rnn.load_state_dict(torch.load("CHOPIN_6HOURS_model.pt", map_location='cpu'))
rnn.eval()

def evaluate(beginning=[15], max_len=1000, temperature=0.8):
    hidden = rnn.init_hidden()
    start_input = Variable(torch.from_numpy(np.asarray(beginning)).long()).to(device)
    predicted = beginning

    for p in range(len(beginning) - 1):
        _, hidden = rnn(start_input[p], hidden)
    input = start_input[-1]
    
    for p in range(max_len):
        output, hidden = rnn(input, hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = int(torch.multinomial(output_dist, 1)[0])

        predicted.append(top_i)
        input = Variable(torch.from_numpy(np.asarray([top_i])).long()).to(device)

    return predicted

evDict = {}
evDictBack = {}
c = 1;
for i in range(0, 33):
    evDict['set_velocity ' + str(i)] = c
    evDictBack[c] = 'set_velocity ' + str(i)
    c += 1
for i in range(1, 129):
    evDict['note_on ' + str(i)] = c
    evDictBack[c] = 'note_on ' + str(i)
    c += 1
for i in range(1, 129):
    evDict['note_off ' + str(i)] = c
    evDictBack[c] = 'note_off ' + str(i)
    c += 1
for i in range(0, 126):
    evDict['time_shift ' + str(i)] = c
    evDictBack[c] = 'time_shift ' + str(i)
    c += 1

print("Seed: ")
seed = int(input())
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)

print("filename: ")
name = input()
PATH = name + ".mid"
song = evaluate([random.randint(1, 400)], 1000, 0.6)
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
vel = 0
note = 0
m = ""
for t in song:
    tmp = evDictBack[t]
    mode, c = tmp.split()
    if mode == "note_on" or mode == "note_off":
        note = int(c)
        m = mode       
    elif mode == "set_velocity":
        vel = int(c)*4
        if vel == 128:
            vel = 127
    elif mode == "time_shift" and m == "note_on" or m == "note_off":
        tm = int(mido.second2tick(int(c)*8*4/1000, 384, 512820))
        track.append(mido.Message(m, note=note, velocity=vel, time=tm))
mid.save(PATH)

from midi2audio import FluidSynth
OUTPUT_WAV = "./" + name + ".wav"
fs = FluidSynth('FluidR3Mono_GM.sf3')
fs.midi_to_audio("./" + PATH, OUTPUT_WAV)