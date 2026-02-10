from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import spikeinterface.extractors as se
import os
import numpy as np
import spikeinterface.full as si
import spikeinterface.sortingcomponents.motion as sm


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names, getSessionsWithinMap
from neuropixels_chronic_spikesorting_Bizley.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing
from neuropixels_chronic_spikesorting_Bizley.helpers.npyx_metadata_fct import load_meta_file

import neuropixels_chronic_spikesorting_Bizley.outerLoopConfigs as outerLoopConfigs
import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.

def main():

    ## below two maybe temporary, may be moved to a config file soon.
    runSessionLoop = True # Also controls saveMultirecInfoFile and doSpikesorting
    saveMultirecInfoFile = True
    doSpikeSorting = False
    doPostprocessing = False

    #get sessions within map

    sessionsWithinMap = getSessionsWithinMap(outerLoopConfigs.SessionsInOrder,all_VE_config.session_path,all_VE_config.stream_id,all_VE_config.channel_map_to_use) ### possibly this just returns SessionsInOrder in a way resulting from a vestige. It used to get everything, but now... Anyway this can be marked for potential future deletion

    # session_name = '021122_trifle_pm3_g0'
    #for session in session_path.glob('*'):
    for i, session in enumerate(sessionsWithinMap):

        multirec_info = {'name': [],
                         'start_time': [],
                         # 'stop_time': [],
                         'duration': [],
                         'fs': [],
                         'n_samples': [],
                         'multirec_start_sample': [],
                         'multirec_stop_sample': [],
                         'fullpath_as_string': []}
        dict_of_recordings = {}
        session_name = session.name

        print(f'Processing {session_name}')
        working_dir = all_VE_config.output_folder / 'tempDir' / all_VE_config.ferret / session_name
        dp = all_VE_config.session_path / session_name

        probeFolder = list(dp.glob('*' + all_VE_config.stream_id[:-3]))
        probeFolder = probeFolder[0]  # name of probe folder
        if runSessionLoop:
            local_probeFolder = all_VE_config.saturatedZonesLocations / Path(*probeFolder.parts[1:]) #might currently only store saturation functions. if so, should rename
            recording = si.read_spikeglx(probeFolder, stream_id=all_VE_config.stream_id)  # for uncompressed
            recording = spikeglx_preprocessing(recording, doRemoveBadChannels=outerLoopConfigs.doRemoveBadChannels,
                                               skipStuffThatKSGUIDoes=outerLoopConfigs.skipStuffThatKSGUIDoes,
                                               local_probeFolder=local_probeFolder, badChannelList=all_VE_config.badChannelList,
                                               bin_s_sessionCat=outerLoopConfigs.bin_s_sessionCat,silenceOrNoiseReplace=outerLoopConfigs.silenceOrNoiseReplace_sessionwise)
            ### do things related to the construction of a file which stores the recording information.
            multirec_info['name'].append(session_name)
            multirec_info['fs'].append(recording.get_sampling_frequency())
            multirec_info['n_samples'].append(recording.get_num_samples())
            multirec_info['duration'].append(recording.get_total_duration())

            meta = load_meta_file(probeFolder / (session_name + '_t0.' + all_VE_config.stream_id + '.meta'))
            multirec_info['start_time'].append(meta['fileCreateTime'])

            #if i == 0:
            multirec_info['multirec_start_sample'].append(0)
            # else:
            #     # multirec_info['multirec_start_sample'].append(int(
            #     #     multirec_info['multirec_start_sample'][i-1] + (multirec_info['duration'][i-1].total_seconds() * multirec_info['fs'][i-1])+1))
            #
            #     multirec_info['multirec_start_sample'].append(
            #         multirec_info['multirec_start_sample'][i - 1] + (multirec_info['n_samples'][i - 1]) + 1)

            # multirec_info['multirec_stop_sample'].append(int(multirec_info['multirec_start_sample'][i] + (multirec_info['duration'][i].total_seconds() * multirec_info['fs'][i])))
            multirec_info['multirec_stop_sample'].append(
                multirec_info['multirec_start_sample'][0] + (multirec_info['n_samples'][0]))
            multirec_info['fullpath_as_string'].append(list(probeFolder.glob('*.bin'))[0].__str__())
            chan_dict = get_channelmap_names(dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
            chan_map_name = chan_dict[session_name + "_" + all_VE_config.stream_id[:-3]]

            if chan_map_name == all_VE_config.channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
                if chan_map_name in dict_of_recordings:
                    dict_of_recordings[chan_map_name].append(recording)
                else:
                    dict_of_recordings[chan_map_name] = [recording]
            #This is only relevant when considering other probe maps --> not for PFC_shank0
            # elif chan_map_name == channel_map_to_use_other_ref:
            #     print(session_name + ' is ' + channel_map_to_use_other_ref)
            #     if channel_map_to_use in dict_of_recordings:
            #         dict_of_recordings[channel_map_to_use].append(recording)
            #     else:
            #         dict_of_recordings[channel_map_to_use] = [recording]

            multirecordings = dict_of_recordings
            multirecordingInput = multirecordings[all_VE_config.channel_map_to_use][0]


        # recording = si.concatenate_recordings(dict_of_recordings)

        ## output_folder_temp = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name # oldstyle
        ## output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name
        output_folder_temp = all_VE_config.output_folder / 'tempDir' / all_VE_config.ferret / all_VE_config.sessionSetLabel / session_name
        output_folder_sorted = all_VE_config.output_folder / 'spikesorted' / all_VE_config.ferret / all_VE_config.sessionSetLabel / session_name
        phy_folder = output_folder_sorted / 'KiloSortSortingExtractor' / 'phy_folder'  # probably.
        phy_folder.mkdir(parents=True, exist_ok=True)
        if False: #skipSessionsAlreadyDone: # should skip sessions already done but doesn't work. If the sorting stops partway through this folder will exist, but you would still want it to be rerun. Need to make a new thing later.
            if (output_folder_temp / Path("tempDir")).is_dir():
                continue
        if saveMultirecInfoFile & runSessionLoop: # would like to seperate this from doing the preprocessing I think...
            df_rec = pd.DataFrame(multirec_info)
            df_rec.to_csv(phy_folder / 'multirec_info.csv',
                          index=False)  # this is the earliest phy folder around and may be a problem... This saves the multirec info

        #try:
        if doSpikeSorting & runSessionLoop:
            sorting = spikesorting_pipeline(
                usedRecording,
                output_folder=output_folder_temp,
                sorter='kilosort4',
                concatenated=True
            )
        elif doPostprocessing == True:
            sorting = si.read_sorter_folder(output_folder_temp / 'tempDir' / 'kilosort4_output')

        if doPostprocessing:  ### turn off in order to do postprocessing in a seperate job later. Frees up gpu, but fundamentally depends on size of job. ### also currently we don't do it.
            sorting = spikesorting_postprocessing(sorting, output_folder=output_folder_sorted)


if __name__ == '__main__':
    main()