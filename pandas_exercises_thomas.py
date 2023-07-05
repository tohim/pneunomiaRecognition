# Homework
# Thomas Himmelstoß - 5005208

import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# always take the same random seed for reproducibility of code
np.random.seed(3)
# dataframe variables
columnAmount = 5
indexAmount = 10000

# Task 1/2/3
# create matrix with random numbers between 0-10 with a size of 5x10000 with according
# timestamp vector (linspace) for each index entry and column names
randNmbrs = np.random.randint(0, 10, size=(indexAmount, columnAmount))
columnNames = 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5'
linMat = np.linspace(1, indexAmount, num=indexAmount)
randMat = pd.DataFrame(randNmbrs, columns=columnNames, index=linMat)

# Task 4
# create vector with 7 trigger events
# creating the trigger points between 0 and 1000 - with the other 5 triggers distributed in between
triggerNmbr = 7
triggerPointsUnif = list(np.random.uniform(1000, 9000, triggerNmbr - 2))
triggerPoints = np.round(triggerPointsUnif, 0)
# creating the jitter with +/- 2 seconds (2 seconds*1000/60 to fit the sampling points per minute)
jitter = np.random.randint(low=-2 * (1000 / 60), high=+2 * (1000 / 60), size=triggerNmbr)
# then creating the trigger vector - for i and point using enumerate to get an indexed list for the triggers
# starting point and ending point at +/- 50 to avoid the jitter to be below or above possible indices
# and thus producing zero values which later distort the means
triggers = [50] + [(point + jitter[i]) for i, point in enumerate(triggerPoints)] + [9950]
triggers.sort()  # sorting the triggers by value from min to max

# Task 5
# creating/ initiating the new column "Events" with NaN values - then adding the triggers to randMat
randMat['Events'] = np.nan
randMat.loc[triggers, 'Events'] = 1
# side note: loc gets rows/columns with particular labels / iloc  gets same but at integer location
print('This is the DataFrame randMat:', '\n', randMat)
print()
# getting minutes of timestamps
timestampMinutes = [val / 1000 for val in triggers]
print('These are all the Trigger Points in Timestamps:', triggers)
print('These are all the Trigger Points in Minutes:', timestampMinutes)
print()
print('This is the Duration of each Trigger Point in Timestamps:', jitter)
print('This is the Duration of each Trigger Point in Seconds:', jitter / 1000 * 60)
print()

# Task 6
# plot the dataframe
randMat[['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5']].plot(subplots=True, layout=(5, 1))
# plt.title('Dataframe Plot')
plt.xlabel('Timestamps')
plt.ylabel('Channels')
plt.show()

# # channel 1
# randMat['Channel 1'].plot()
# plt.title('Channel 1')
# plt.xlabel('Timestamps')
# plt.ylabel('Value')
# plt.show()
#
# # just for my interest: print only trigger points of channel 1
# new_trigger = [trigger-1 for trigger in triggers]
# randMat.iloc[new_trigger].plot(y='Channel 1')
# plt.title('Channel 1 Trigger Points')
# plt.xlabel('Timestamps')
# plt.ylabel('Value')
# plt.show()
#
# # channel 2
# randMat['Channel 2'].plot()
# plt.title('Channel 2')
# plt.xlabel('Timestamps')
# plt.ylabel('Value')
# plt.show()
#
# # channel 3
# randMat['Channel 3'].plot()
# plt.title('Channel 3')
# plt.xlabel('Timestamps')
# plt.ylabel('Value')
# plt.show()
#
# # channel 4
# randMat['Channel 4'].plot()
# plt.title('Channel 4')
# plt.xlabel('Timestamps')
# plt.ylabel('Value')
# plt.show()
#
# # channel 5
# randMat['Channel 5'].plot()
# plt.title('Channel 5')
# plt.xlabel('Timestamps')
# plt.ylabel('Value')
# plt.show()

# Task 7
# write a function that returns the timestamp at which the trigger event occurs
# simpler way would be to just use the variable "triggerPoint" from before - but this function is able to
# detect any of the triggers that could appear in the example dataframe
print('All Timestamps for Trigger Events:')

def eventTime():
    # for loop takes the x from 1 to length of randMat and with 'if' iterates through the column 'Events' and whenever
    # it is 1 it saves the index value at current position x in variable t
    timeStamps = []  # creating an empty list to store the timeStamps
    for x in range(len(randMat)):
        if randMat['Events'].iloc[x] == 1:
            t = randMat.index[x]
            timeStamps.append(t)  # add current time stamp t at [x] to the list timeStamps
            tMinute = t / 1000
            tSecs = np.round(tMinute * 60, 0)
            print('Timestamp', t, ' is equal to:', tMinute, 'minutes/', tSecs, 'seconds')
    return timeStamps

# call the function
# eventTime()


# Task 8
# creating the epochs with duration 2 sec before and 3 sec after trigger event by using the eventTime() output
# for the respective time of a trigger
triggerEpochs = nk.epochs_create(randMat, events=eventTime(), sampling_rate=1, epochs_start=-2 * (1000 / 60),
                                 epochs_end=3 * (1000 / 60))
print('\nThese are the Trigger Epochs:', '\n', triggerEpochs)

# plotting the Epochs for every Trigger per Channel
nk.epochs_plot(triggerEpochs, legend=False)
plt.ylabel('Channels')
plt.xlabel('Epoch Time Sequence', loc='center')
plt.show()
# I wanted to use the epochs_plot, but sadly I am not sure how to adjust the title/ legends/ labels here - looks
# a bit messy. Because of that the y-axis and the last column 'Events' looks weird

print('\nPrinting the respective Mean Values over each Channel for each Epoch:')

# extract only the channel values
meanEpochsDfs = []  # list to store all dfs

for epochID, epochInfo in triggerEpochs.items():
    print('\nEpoch ID:', epochID)
    meanEpoch = pd.DataFrame()
    # safe column mean (axis=1 refers to the mean over rows, instead of columns)
    meanEpoch['mean'] = epochInfo[['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5']].mean(axis=1)
    meanEpochsDfs.append(meanEpoch)
    print(meanEpoch)

# plotting each Epoch Mean
# Epoch 1 Mean Plot
# meanEpochsDfs[1].plot()
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Mean Value')
# plt.title('Epoch 1 Mean')
# plt.legend(loc='lower right')
# plt.show()
#
# # Epoch 2 Mean Plot
# meanEpochsDfs[2].plot()
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Mean Value')
# plt.title('Epoch 2 Mean')
# plt.legend(loc='lower right')
# plt.show()
#
# # Epoch 3 Mean Plot
# meanEpochsDfs[3].plot()
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Mean Value')
# plt.title('Epoch 3 Mean')
# plt.legend(loc='lower right')
# plt.show()
#
# # Epoch 4 Mean Plot
# meanEpochsDfs[4].plot()
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Mean Value')
# plt.title('Epoch 4 Mean')
# plt.legend(loc='lower right')
# plt.show()
#
# # Epoch 5 Mean Plot
# meanEpochsDfs[5].plot()
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Mean Value')
# plt.title('Epoch 5 Mean')
# plt.legend(loc='lower right')
# plt.show()

# plotting all Epoch Means in 1 figure
fig, axs = plt.subplots(7)
fig.suptitle('All Epoch Means')
fig.text(0.03, 0.5, 'Epochs', verticalalignment='center', rotation='vertical')
fig.text(0.5, 0.03, 'Trigger Epoch Sequence', horizontalalignment='center', rotation='horizontal')
axs[0].plot(meanEpochsDfs[0])
axs[1].plot(meanEpochsDfs[1])
axs[2].plot(meanEpochsDfs[2])
axs[3].plot(meanEpochsDfs[3])
axs[4].plot(meanEpochsDfs[4])
axs[5].plot(meanEpochsDfs[5])
axs[6].plot(meanEpochsDfs[6])
fig.show()


# Extra Task - just out of interest:
# getting the mean of each channel by averaging
# Channel 1 Mean
channelMean1 = nk.epochs_average(triggerEpochs, which='Channel 1')
# (channelMean1['Channel 1_Mean']).plot()
# plt.title('Channel 1 Mean')
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Values')
# plt.legend(loc='lower right')
# plt.show()
#
# # Channel 2 Mean
channelMean2 = nk.epochs_average(triggerEpochs, which='Channel 2')
# (channelMean2['Channel 2_Mean']).plot()
# plt.title('Channel 2 Mean')
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Values')
# plt.legend(loc='lower right')
# plt.show()
#
# # Channel 3 Mean
channelMean3 = nk.epochs_average(triggerEpochs, which='Channel 3')
# (channelMean3['Channel 3_Mean']).plot()
# plt.title('Channel 3 Mean')
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Values')
# plt.legend(loc='lower right')
# plt.show()
#
# # Channel 4 Mean
channelMean4 = nk.epochs_average(triggerEpochs, which='Channel 4')
# (channelMean4['Channel 4_Mean']).plot()
# plt.title('Channel 4 Mean')
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Values')
# plt.legend(loc='lower right')
# plt.show()
#
# # Channel 5 Mean
channelMean5 = nk.epochs_average(triggerEpochs, which='Channel 5')
# (channelMean5['Channel 5_Mean']).plot()
# plt.title('Channel 5 Mean')
# plt.xlabel('Trigger Epoch Sequence')
# plt.ylabel('Values')
# plt.legend(loc='lower right')
# plt.show()

# plotting all Channel Means in 1 figure
fig, axs = plt.subplots(5)
fig.suptitle('All Channel Means')
fig.text(0.03, 0.5, 'Channels', verticalalignment='center', rotation='vertical')
fig.text(0.5, 0.03, 'Trigger Epoch Sequence', horizontalalignment='center', rotation='horizontal')
axs[0].plot((channelMean1['Channel 1_Mean']))
axs[1].plot((channelMean2['Channel 2_Mean']))
axs[2].plot((channelMean3['Channel 3_Mean']))
axs[3].plot((channelMean4['Channel 4_Mean']))
axs[4].plot((channelMean5['Channel 5_Mean']))
fig.show()
