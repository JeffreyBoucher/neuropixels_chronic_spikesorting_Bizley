#### packages that are external but "more official"
import pandas as pd
import numpy as np

def apply_behavior_filter(df, filter,window = [-1,2],dp = None,evtype = None,metaFileData=None,pulseDictThisSession=None):
    '''

    :param df: dataframe with synchronized neural and behavioral data
    :param filter: type of trials that we want
    :param window:
    :param dp: data path, only needed for at least one filter (exlude_saturations) and so is often omitted as an argument.
    :return: data frame with only the filtered trials
    '''

    if filter == 'Target trials':
        return df[df['catchTrial'] != 1]
    if filter == 'Catch trials':
        return df[df['catchTrial'] == 1]
    if filter == 'Level cue':
        return df[df['currAtten'] > 0]
    if filter == 'No Level Cue':
        return df[df['currAtten'] == 0]
    if filter == 'Non Correction Trials':
        return df[df['correctionTrial'] == 0]
    if filter == 'Correction Trials':
        return df[df['correctionTrial'] == 1]
    if filter == 'Sound Right':
        return df[df['side'] == 1]
    if filter == 'Sound Left':
        return df[df['side'] == 0]
    if filter == 'Correct':
        return df[df['correct'] == 1]
    if filter == 'Noiseless':
        return df[df['currNoiseAtten'] == 120]
    if (filter == 'Noiseful')|(filter == 'Noiseful_stricter'):
        return df[df['currNoiseAtten'] < 0] # a bit stricter... I have sessions where apparently ferrets will have 5 and 10. That could mean anything.
        #return df[df['currNoiseAtten'] < 100]
    if filter == 'TrialDoesNotEnd':
        # only include trials which end after the window is over. For "trial start" windows.
        return df[df['centreRelease'] > window[1]]
    if filter == 'TargetNotInWindow':
        # only includes trials for which the target does not appear in the window selected. For "trial start" windows.
        return df[df['targTimes'] > window[1]]
    if filter == 'TargetAtLeast2SecondsIn':
        # only includes trials for which the target appears after 2 seconds. For "target onset" windows.
        return df[df['targTimes'] > 2]
    if filter == 'FA trials':
        return df[df['response'] == 5]
    if filter == 'Hit trials':
        return df[(df['response'] == 1) | (df['response'] == 0)]
    if filter == 'Miss trials':
        return df[df['response'] == 7]
    if filter == 'Talker 1':
        return df[df['talker'] == 1]
    if filter == 'Talker 2':
        return df[df['talker'] == 2]
    if (filter == 'Miss & FA')|(filter == 'Miss and FA'):
        return df[(df['response'] == 5) | (df['response'] == 7)]
    if filter == 'Hit and FA':
        return df[(df['response'] == 1) | (df['response'] == 0) | (df['response'] == 5)]
    if filter == 'Hit and Miss and Wrong Direction': ### used to be Hit and Miss but does not exclude Wrong Direction
        return df[(df['response'] == 1) | (df['response'] == 0) | (df['response'] == 7)]
    if filter[0:7] == 'ILD_dB_':
        ILD_requested = int(filter[7:10])
        return df[df['currentILD'] == ILD_requested]
    if filter == 'Exclude_Saturations':
        ev_times = df[evtype].to_numpy()
        lower_bound = ev_times + window[0] ### lower bound of time to exclude as recorded in behavior file
        upper_bound = ev_times + window[1] ### upper bound of time to exclude as recorded in behavior file
        sat_ranges_inSamples = np.genfromtxt(dp, delimiter=' ')
        sat_ranges_original = sat_ranges_inSamples/metaFileData['imSampRate'] ### sat ranges in time, before alignment so in terms of imec time. ### TODO make sure that this is the correct sampling rate later in life.
        print('double check this sampling rate is right after you do the calibration')
        sat_ranges = np.atleast_2d(np.zeros((sat_ranges_original.shape)))
        if sat_ranges_original.any(): ### This is a more direct copy of Jules's version... Gives results that are different by 0.005 for some reason? ### I should turn this into a function and use it for both saturation and event alignment.
            buffer_before_each_pulse = -2 ### don't know what this is or how to understand it at the moment.
            pulse_times = pulseDictThisSession['imec_pulse_time'] ### gets imec pulse times. Should already have equal trial number to tdt.
            # get trial times
            tdt_pulse_times = pulseDictThisSession['tdt_pulse_time'] ### gets tdt pulse times.

            # align spike time to behaviour time and keep only spikes around pulses
            sat_times_around_pulses = []
            sat_times_around_pulses_off = []
            for i, (pulse_time, tdt_pulse_time) in enumerate(zip(pulse_times, tdt_pulse_times)): ### for every pair of imec and tdt pulse...
                # TODO check if there is 0.5s delay between tdt and imec #### I asked Jules about this and he said he no longer is worried about this basically
                sat_ranges = np.atleast_2d(sat_ranges_original - pulse_time + tdt_pulse_time) ### dubtract the pulse time and add the tdt time in order to put the sat ranges in terms of tdt time. Accounts for offset.
                ### so now we have a set of all sat ranges in the same units as tdt. Now, we can go through every trial and see if there is a match. If there is a match, we exclude it.

                # avoid overlapping windows between trials
                if (i < len(pulse_times) - 1): ### this is done for all but the last trial.
                    curr_window = [buffer_before_each_pulse, (pulse_times[i + 1] - pulse_time) + buffer_before_each_pulse] ### so basically there is an explicit window from one pulse to the next, with the -2 there so that the baseline uses the same pulse I guess.

                # Deal with first and last trial
                if i == 0:
                    #mask = (sat_ranges[:,0] >= 0) & (sat_ranges[:,0] <= tdt_pulse_time + curr_window[1]) ### collect all sat_ranges which end up in this first trial. Unlike spikes, sat_ranges are ranges. I should maybe handle pulses seperately for the beginning and end, actually...
                    mask = (sat_ranges[:, 0] <= tdt_pulse_time + curr_window[1]) ### I am fine with including pre-behavior for this purpose (of sats)
                elif i == len(pulse_times) - 1:
                    mask = (sat_ranges[:,0] >= tdt_pulse_time + buffer_before_each_pulse)
                else:
                    mask = (sat_ranges[:,0] >= tdt_pulse_time + curr_window[0]) & (sat_ranges[:,0] <= tdt_pulse_time + curr_window[1])

                sat_times_around_pulses.extend(sat_ranges[mask,0]) ### so the thing I think is happening here is that every saturation range is being stated in terms of the nearest pulse. Or rather, the pulse that applies to it (not nearest because it is assymetric, focused at the beginning of each trial).
                sat_times_around_pulses_off.extend(sat_ranges[mask,1])

            # Go back to old format
            sat_ranges = np.atleast_2d(np.zeros((sat_ranges_original.shape)))
            sat_ranges[:,0] = sat_times_around_pulses
            sat_ranges[:,1] = sat_times_around_pulses_off

        notSafeTrialsMatrix = np.full((len(ev_times),sat_ranges.shape[0]),False)
        if sat_ranges.any():
            for i,currentTestRange in enumerate(np.atleast_2d(sat_ranges)): ### go through every sat range, which has now been stated in terms of nearby pulse times.
                notSafeTrialsMatrix[:,i] = ((lower_bound > currentTestRange[0]) & (lower_bound < currentTestRange[1]) | (upper_bound > currentTestRange[0]) & (upper_bound < currentTestRange[1])) ### determine if each trial is unsafe in terms of that saturation range. Following the full loop, every trial will have been gone through.


        return df[~notSafeTrialsMatrix.any(axis=1)] ### strong expectation that, in the dataset for which I expect most issue, this will be lateralization dependent.
    else:
        return f'Filter "{filter}" not found'

