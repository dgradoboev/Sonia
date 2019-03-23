import mido
import math
import os
import shutil
import time
from timerForParsing import timeSince

datadir = 'data/'
correctdir = 'correct_data/'

os.mkdir(correctdir)

wrong = 0
allF = 0

start = time.time()

for subdir, dirs, files in os.walk(datadir):
    for file in files:
        allF += 1
        if allF % 100 == 0:
            percent = allF*100 // len(files)
            print(timeSince(start, percent / 100), ' — ', percent, '%')
        tmpMIDI = mido.MidiFile(datadir + file)
        count = 0
        for x in tmpMIDI:
            if x.type == 'note_off':
                count = 1
                break
        if count == 0:
            wrong += 1
        else:
            shutil.copy(datadir + file, correctdir + file)
print("FINISHED", end='\n\n')
print('Files:', allF)
print('Wrong files:', wrong)