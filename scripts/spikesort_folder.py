from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import spikeinterface.extractors as se
import os
import numpy as np
import shutil ### to remove a directory, not needed long term


import spikeinterface.full as si
import spikeinterface.sortingcomponents.motion as sm


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names, getSessionsWithinMap
from neuropixels_chronic_spikesorting_Bizley.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing, makeAndProcessRaw
from neuropixels_chronic_spikesorting_Bizley.helpers.npyx_metadata_fct import load_meta_file

import neuropixels_chronic_spikesorting_Bizley.outerLoopConfigs as outerLoopConfigs
import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.

def main():

    ## these are a bunch of parameters you should set before running spikesorting which are hopefully named in a self-explanatory way.
    runSessionLoop = True # Also controls saveMultirecInfoFile and doSpikesorting
    saveMultirecInfoFile = True
    overwriteRaw = False # determines if you recreate a raw that is already detected. Set to true if you want to remake it, false otherwise.
    createCompressed = False # determines whether you create compressed raw. Good to do if you need the space, but all processing requires unpacking it.
    doSaturationReplace = False ### in the new version of si this no longer works for confusing reasons...
    doSpikeSorting = True
    doPostprocessing = False ### I currently do not use this at all. I left the infrastructure in case you want to do any kind of postprocessing.
    floatIntoKilosort = True ### I feel like kilosort should work better if you input int16 (because that is the spikeglx format), but historically I have better luck with float... Anyway you can decide whether it is float or int here.

    #get sessions within map
    if not all_VE_config.SurveyOverride: ### if not a survey...
        NAS_SessionsWithinMap,Local_SessionsWithinMap = getSessionsWithinMap(outerLoopConfigs.NAS_SessionsInOrder,all_VE_config.NAS_session_path,all_VE_config.stream_id,all_VE_config.channel_map_to_use,outerLoopConfigs.Local_SessionsInOrder) ### possibly this just returns SessionsInOrder in a way resulting from a vestige. It used to get everything, but now... Anyway this can be marked for potential future deletion
    else: ### if survey, the order doesn't actually matter, and they will probably be "in order" anyway.
        NAS_SessionsWithinMap = outerLoopConfigs.NAS_SessionsInOrder
        Local_SessionsWithinMap = outerLoopConfigs.Local_SessionsInOrder
    # session_name = '021122_trifle_pm3_g0'
    #for session in session_path.glob('*'):
    for i, session in enumerate(NAS_SessionsWithinMap):

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
        NAS_currentDataPath = all_VE_config.NAS_session_path / session_name

        probeFolder = list(NAS_currentDataPath.glob('*' + all_VE_config.stream_id[:-3])) ### instead of depending on streamid I should read the meta files in both folders and decide based on probemap.
        probeFolder = probeFolder[0]  # name of probe folder
        if runSessionLoop:
            local_probeFolder = all_VE_config.local_session_path / Path(*probeFolder.parts[-2:]) ### this is a local version of the top-evel folder of the spikeglx session
            local_probeFolder.mkdir(parents=True, exist_ok=True)
            recording = si.read_spikeglx(probeFolder, stream_id=all_VE_config.stream_id)  # for reading uncompressed from NAS
            rawFolderName = local_probeFolder / Path('catgt_CorrectedRaw')
            if (not any((list(rawFolderName.glob('*.bin'))))) | overwriteRaw: # if you don't have a raw or you say you want to overwrite it, we make a new raw.
                makeAndProcessRaw(NAS_session_path = all_VE_config.NAS_session_path,session_name=session_name,stream_id=all_VE_config.stream_id,local_session_path=all_VE_config.local_session_path,badChannelList=all_VE_config.badChannelList,catgt_location=outerLoopConfigs.catgt_location,rawFolderName=rawFolderName,overwriteRaw = overwriteRaw,createCompressed = createCompressed)
            recording = si.read_spikeglx(rawFolderName, stream_id=all_VE_config.stream_id)
            recording = spikeglx_preprocessing(recording,local_probeFolder=local_probeFolder,doSaturationReplace=doSaturationReplace,
                                               silenceOrNoiseReplace=outerLoopConfigs.silenceOrNoiseReplace_sessionwise,floatDataTypeForDriftCorrection=floatIntoKilosort)
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
            chan_dict = get_channelmap_names(NAS_currentDataPath)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
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
            if not all_VE_config.SurveyOverride: ### If we aren't doing SurveyOverride, then we want to keep track of the channel_map we intend to use so we can see if there is a mismatch later.
                multirecordingInput = multirecordings[all_VE_config.channel_map_to_use][0]


        # recording = si.concatenate_recordings(dict_of_recordings)

        ## output_folder_temp = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name # oldstyle
        ## output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name
        output_folder_temp = all_VE_config.output_folder / 'tempDir' / all_VE_config.ferret / all_VE_config.sessionSetLabel / session_name
        output_folder_sorted = all_VE_config.output_folder / 'spikesorted' / all_VE_config.ferret / all_VE_config.sessionSetLabel / session_name
        phy_folder = output_folder_sorted / 'KiloSortSortingExtractor' / 'phy_folder'  # probably.
        phy_folder.mkdir(parents=True, exist_ok=True)
        if True: #skipSessionsAlreadyDone: # should skip sessions already done but doesn't work. If the sorting stops partway through this folder will exist, but you would still want it to be rerun. Need to make a new thing later.
            if (output_folder_temp / Path("tempDir")).is_dir():
                continue
        if saveMultirecInfoFile & runSessionLoop: # would like to seperate this from doing the preprocessing I think...
            df_rec = pd.DataFrame(multirec_info)
            df_rec.to_csv(phy_folder / 'multirec_info.csv',
                          index=False)  # this is the earliest phy folder around and may be a problem... This saves the multirec info... Also this stuff might be better saved in spikesortingInput...

        usedRecording = recording ### there may be no advantage to this at all, it is a vestige of when I was concatenating.
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