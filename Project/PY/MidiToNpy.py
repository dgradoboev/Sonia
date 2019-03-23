import math
import mido
import numpy as np
import os
import shutil
import time
from timerForParsing import timeSince

datadir = 'correct_data/'
npydir = 'npy_data/'

os.mkdir(npydir)

evDict = {}
c = 1;
for i in range(0, 33):
    evDict['set_velocity ' + str(i)] = c
    c += 1
for i in range(1, 129):
    evDict['note_on ' + str(i)] = c
    c += 1
for i in range(1, 129):
    evDict['note_off ' + str(i)] = c
    c += 1
for i in range(0, 126):
    evDict['time_shift ' + str(i)] = c
    c += 1
    
data = []
count = 0

start = time.time()

for subdir, dirs, files in os.walk(datadir):
    for file in files:
        count += 1
        if count % 100 == 0:
            percent = count*100 // len(files)
            print(timeSince(start, percent / 100), ' — ', percent, '%')
        seq = []
        MIDI = mido.MidiFile(datadir + file)
        for s in MIDI:
            if (s.type == 'note_on' or s.type == 'note_off'):
                seq.append(evDict['set_velocity ' + str(round(s.velocity / 4))])
                seq.append(evDict[s.type +' ' + str(s.note)])
                tmpTime = math.ceil(s.time * 125)
                while (tmpTime > 125):
                    seq.append(evDict['time_shift ' + str(125)])
                    tmpTime -= 125
                if (tmpTime != 0):
                    seq.append(evDict['time_shift ' + str(tmpTime)])
        data.append(seq)
print("FINISHED", end='\n\n')
np.save(npydir + 'NPYData.npy', np.asarray(data))
print("SAVED")