import random
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

##############################################################################
min_val = 1
max_val = 10

random_matrix = np.random.randint(min_val,max_val,(5,10001))

df = pd.DataFrame(random_matrix, index=["channel1", "channel2", "channel3", "channel4", "channel5"])

rng = pd.date_range(pd.to_datetime('2023-01-14 14:50:00'), pd.to_datetime('2023-01-14 15:00:00'), freq="60ms")

df = pd.concat([df.T, pd.Series(rng)], axis=1, ignore_index=False).rename(columns={0: "Time"})

##############################################################################

num_triggers = 7        # Number of triggers
interval = 60           # Time interval between triggers
jitter_deviation = 2    # Jitter deviation

jitter = np.random.uniform(-jitter_deviation, jitter_deviation, size=num_triggers)   # Generate random jitter values for each trigger

trigger_times = np.cumsum(np.full(num_triggers, interval)) + jitter                  # Create vector with trigger + jitter

#timestamps = trigger_times                                                          #Variable saved to be used later to know when the triggers happen in sec
trigger_times = trigger_times * 1000/60                                              #There is 10,000 points in 10 min -> 600 s -> 1,000 data points in 60 seconds
trigger_times = pd.Series(trigger_times)                                             #Saved as a series (better handling properties with pandas)
trigger_times = trigger_times.astype(int)                                            #Save triggers as integers since they will be used as index points and we only have integer indexes

vector = np.zeros(10001)                                                             #Create a vector of zeros

vector[trigger_times] = 1

##############################################################################

df = pd.concat([df, pd.Series(vector)], axis=1, ignore_index=False).rename(columns={0: "events"})
print(df)

df.plot(subplots=True, layout=(7, 1))
plt.show()

##############################################################################
epochs = []
for i in trigger_times:
    epoch_start = int(i - (2 * (1000/60)))           # 2 seconds before
    epoch_stop = int(i + (3 * (1000/60)))            # 3 seconds after
    epoch = df.iloc[epoch_start:epoch_stop+1]        # Extract the epoch from the dataframe -> iloc extracts
    epochs.append(epoch)

mean_epochs = pd.DataFrame()

for i, epoch in enumerate(epochs):
        mean_epoch = epoch.mean(numeric_only=True)
        mean_epochs['Epoch {}'.format(i+1)] = mean_epoch

##############################################################################

# Plot the mean epochs for each channel
mean_epochs.plot()
plt.xlabel('Time')
plt.ylabel('Mean Value')
plt.title('Mean Channels')
plt.legend(loc='lower right')
plt.show()
timestamps = np.round(df.index[df['events'] == 1], 2)                       #trigger points
timestamps

print(timestamps)

