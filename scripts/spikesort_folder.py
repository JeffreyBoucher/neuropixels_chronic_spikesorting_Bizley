from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt
import torch
import pandas as pd

import spikeinterface.extractors as se
import os
import numpy as np
import shutil ### to remove a directory, not needed long term


import spikeinterface.full as si
import spikeinterface.sortingcomponents.motion as sm

import kilosort as ks


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names, getSessionsWithinMap
from neuropixels_chronic_spikesorting_Bizley.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing, makeAndProcessRaw
from neuropixels_chronic_spikesorting_Bizley.helpers.npyx_metadata_fct import load_meta_file

import neuropixels_chronic_spikesorting_Bizley.outerLoopConfigs as outerLoopConfigs
import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.

def main():

    ## these are a bunch of parameters you should set before running spikesorting which are hopefully named in a self-explanatory way.
    saveMultirecInfoFile = True ### whether to save metadata. You'll only want to turn it off if you are testing a thing and don't want to overwrite an old file or something...
    overwriteRaw = False # determines if you recreate a raw that is already detected. Set to true if you want to remake it, false otherwise.
    overwriteChanMap = True # determines whether you overwrite the channel map. In my opinion this should always be on.
    overwriteSpikesorting = False # determines whether you redo spikesortings that are already done. Overwrite when you try new things, don't sh
    createCompressed = False # determines whether you create compressed raw. Good to do if you need the space, but all processing requires unpacking it.
    doSaturationReplace = False ### Need to make a catgt version of this...
    doSpikeSorting = True
    JeffManuallySkipsAThingTemporarily = True

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

        NASprobeFolder = list(NAS_currentDataPath.glob('*' + all_VE_config.stream_id[:-3])) ### instead of depending on streamid I should read the meta files in both folders and decide based on probemap.
        NASprobeFolder = NASprobeFolder[0]  # name of probe folder
        local_probeFolder = all_VE_config.local_session_path / Path(*NASprobeFolder.parts[-2:]) ### this is a local version of the top-evel folder of the spikeglx session
        local_probeFolder.mkdir(parents=True, exist_ok=True)
        rawFolderName = local_probeFolder / Path('catgt_CorrectedRaw')
        if (not any((list(rawFolderName.glob('*.bin'))))) | overwriteRaw: # if you don't have a raw or you say you want to overwrite it, we make a new raw.
            ### currently, I do not have a saturation removal function because my old one depended on SI. To be dealt with later.

            makeAndProcessRaw(NAS_session_path = all_VE_config.NAS_session_path,session_name=session_name,stream_id=all_VE_config.stream_id,local_session_path=all_VE_config.local_session_path,badChannelList=all_VE_config.badChannelList,catgt_location=outerLoopConfigs.catgt_location,rawFolderName=rawFolderName,overwriteRaw = overwriteRaw,createCompressed = createCompressed)
        recording = si.read_spikeglx(rawFolderName, stream_id=all_VE_config.stream_id) ### use si just to read the meta file.


        ### preparing some things specifically for ks spikesorting

        settings = {'n_chan_bin': len(recording.channel_ids)+1} ### set settings dictionary, including required nchan_bin which should be correct as written.

        ### code up to chanMap.json being saved copied from spikeinterface.
        groups = recording.get_channel_groups()
        positions = np.array(recording.get_channel_locations())
        if positions.shape[1] != 2:
            raise RuntimeError("3D 'location' are not supported. Set 2D locations instead.")

        n_chan = recording.get_num_channels()
        chanMap = np.arange(n_chan)
        xc = positions[:, 0]
        yc = positions[:, 1]
        unique_groups = set(groups)
        group_map = {group: idx for idx, group in enumerate(unique_groups)}
        kcoords = np.array([group_map[group] for group in groups], dtype=int)

        probe = {
            "chanMap": chanMap,
            "xc": xc,
            "yc": yc,
            "kcoords": kcoords,
            "n_chan": n_chan,
        }
        sorter_output_folder = all_VE_config.sessionSetSortedFolder / session_name
        probe_name = sorter_output_folder / Path("chanMap.json")  ### I still need to figure out how to automatically generate this file. When I do, I can save it into "inputs" somewhere.
        if (not probe_name.is_file())|(overwriteChanMap):
            ks.io.save_probe(probe, str(sorter_output_folder / "chanMap.json"))

        ### do things related to the construction of a file which stores the recording information.
        multirec_info['name'].append(session_name)
        multirec_info['fs'].append(recording.get_sampling_frequency())
        multirec_info['n_samples'].append(recording.get_num_samples())
        multirec_info['duration'].append(recording.get_total_duration())

        meta = load_meta_file(NASprobeFolder / (session_name + '_t0.' + all_VE_config.stream_id + '.meta')) ### this reads the NAS meta file. If you don't want to do that for some reason, note that "fileCreateTime", the only reason we load this, is stored as "fileCreateTime_original" in the new meta file.
        multirec_info['start_time'].append(meta['fileCreateTime'])

        multirec_info['multirec_start_sample'].append(0) ### this use to be integrated into a concatenation thing. That's why we bother with this.

        # multirec_info['multirec_stop_sample'].append(int(multirec_info['multirec_start_sample'][i] + (multirec_info['duration'][i].total_seconds() * multirec_info['fs'][i])))
        multirec_info['multirec_stop_sample'].append(
            multirec_info['multirec_start_sample'][0] + (multirec_info['n_samples'][0]))
        multirec_info['fullpath_as_string'].append(list(NASprobeFolder.glob('*.bin'))[0].__str__())
        chan_dict = get_channelmap_names(NAS_currentDataPath)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
        chan_map_name = chan_dict[session_name + "_" + all_VE_config.stream_id[:-3]]

        if chan_map_name == all_VE_config.channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
            if chan_map_name in dict_of_recordings:
                dict_of_recordings[chan_map_name].append(recording)
            else:
                dict_of_recordings[chan_map_name] = [recording]

        multirecordings = dict_of_recordings
        if not all_VE_config.SurveyOverride: ### If we aren't doing SurveyOverride, then we want to keep track of the channel_map we intend to use so we can see if there is a mismatch later.
            multirecordingInput = multirecordings[all_VE_config.channel_map_to_use][0]

        if saveMultirecInfoFile:  # would like to seperate this from doing the preprocessing I think...
            df_rec = pd.DataFrame(multirec_info)
            df_rec.to_csv(sorter_output_folder / 'multirec_info.csv',
                          index=False)  # this is the earliest phy folder around and may be a problem... This saves the multirec info... Also this stuff might be better saved in spikesortingInput...


        if doSpikeSorting&((overwriteSpikesorting)|(not any(list(sorter_output_folder.glob('params.py'))))):
            if JeffManuallySkipsAThingTemporarily:
                if i == 1: ###
                    continue
            ks.run_kilosort(settings, probe=None, probe_name=probe_name, filename=None,
                 data_dir=rawFolderName, file_object=None, results_dir=sorter_output_folder,
                 data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                 progress_bar=None, save_extra_vars=True, clear_cache=False,
                 save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                 verbose_console=True, verbose_log=True, torch_thread_lim=None)



if __name__ == '__main__':
    main()