#################################################################
#       Solution for the Pandas and Neurokit2 Task Sheet        #
#               made for the course NE2104.NCS                  #
#       ------------------------------------------------        #
#       Author: Cornelius Breitenbach, Matr.Nr.: 5004007        #
#################################################################


import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

matrix_size = (10000, 4)    # 4 Channels + Timestamps
fs = 100/6                  # 10000 samples = 10 min = 600 s : fs = 10000/600 Hz

times = pd.timedelta_range(start='0m', end='10m', periods=matrix_size[0], name='Timestamp')
col_names = ['CH' + str(x) for x in range(1, matrix_size[1]+1)] # names for the channels
rnd_gen = np.random.default_rng()   # init random generator

def create_random_df(rnd, ppg=True):
    if ppg != True:
        ## make df out of random matrix
        dd_df = rnd.integers(11, size=matrix_size)
        # create df from matrix, set column names (1-indexed)
        dd_df = pd.DataFrame(dd_df, columns=col_names)
        dd_df.reindex(index=times)
        dd_df.index.names = [times.name]
        dd_df.columns = col_names
    else:
        ## make df out of dummy signals - in this case PPG
        dd_df = []
        num_channels = matrix_size[1]
        num_rnd_sig = nk.misc.spawn_random_state(rnd, n_children=num_channels)
        print(f'Generating PPG-Signal')
        for i in range(len(num_rnd_sig)):
            print('\tfor Channel {}'.format(i + 1))
            dd_df.append(nk.ppg_simulate(duration=matrix_size[0]/1000,      # add more variation in form of
                                         motion_amplitude=i * 0.7,     # motion artefacts
                                         burst_number=i % 2,           # spikes
                                         random_state=num_rnd_sig[i])) # dependant on channel
            # obviously, channels of the same signal should not be based on random data,
            # since it's simulating on single physiological measurement. but it's for showing prettier graphs ...
        print('Building the DataFrame might take a bit.')
        dd_df = pd.DataFrame(data=dd_df).T         # transpose to fit to example shape (10000 rows, 4 cols)

        dd_df.reindex(index=times)                 # set Timestamps as index
        dd_df.index.names = [times.name]           # name index and columns
        dd_df.columns = col_names
    return dd_df


use_biological = True   # False for using random generated data, True for ppg signals
df = create_random_df(rnd_gen, ppg=use_biological)

# create trigger / events
# doing with index is fine, but Time(delta) is easier to read
events = pd.Series(data=[0]*matrix_size[0], index=times, name='events')

num_trigger = 7
jitter = rnd_gen.uniform(low=-2, high=2, size=num_trigger)
trigger_t = np.cumsum(np.full(num_trigger, 60)).astype('float') # 7 triggers, one every 60 s
trigger_t = np.floor((trigger_t + jitter)).tolist()             # add jitter, save as rounded s in list
events[events.index.seconds.isin(trigger_t)] = 1                # set trigger on every occurrence in trigger

df['events'] = events.values                                    # append values to df
df.reindex(index=times)
df['seconds'] = times / np.timedelta64(1,'s')                   # the prettier version
df.plot(x='seconds', subplots=True)
plt.show()

# get a list of times when an event happened
def get_times(x, t):
    '''get_times returns a list of the input DataFrame x
    containing all Timestamps when a Trigger event took place.'''
    y = x[x.events != 0].iloc[-1].values.tolist()       # returning indices
    ret_list = []
    for val in y:   # change to timestamps and return
        ret_list.append(pd.Timedelta(val, 's'))
    return ret_list

trigger_timestamps = get_times(df, times)

# find epochs by using nk
events = nk.events_find(df.events)

# separate epochs
start_onset = 2     # s
end_offset = 3      # s
epochs = nk.epochs_create(df, events,
                          sampling_rate=int(fs),
                          epochs_start=-start_onset,
                          epochs_end=end_offset)

# calculate means
# calculate mean of each channel across epochs
mean_ch = []
mean_ep = []
for epoch in epochs.values():
    mean_ch.append(epoch[col_names].mean())     # mean for each channel and epoch
    mean_ep.append(epoch)


mean_ch = pd.DataFrame(mean_ch)     # df, because I wanted to include more things
mean_ch.plot()                      # didn't happen, tho
plt.xlabel('epoch')
plt.ylabel('mean amplitude')
plt.show()

# calculate mean across all channels
mean_ep = pd.concat(mean_ep, ignore_index=True)
mean_ep['avg'] = mean_ep[col_names].T.mean()    # transp. for easier handling
mean_ep[['avg', 'events']].plot(label=['mean across channels','events'])  # plot only two cols
plt.ylabel('amplitude')
plt.xlabel('seconds')
plt.legend()
plt.show()