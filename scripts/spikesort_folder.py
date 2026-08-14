from pathlib import Path
import pandas as pd
from itertools import product as combination_product

import numpy as np
import shutil

import spikeinterface.full as si

import kilosort as ks


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import get_channelmap_names, getSessionsWithinMap
from neuropixels_chronic_spikesorting_Bizley.spikesorting import applyOverStrike, nullify_saturations, makeAndProcessRaw, getLFP
from neuropixels_chronic_spikesorting_Bizley.sessionSetManagement.getSessionsWithinMapForSessionSet import getSessionsWithinMapForSessionSet
from generally_useful_functions_Bizley.getCorrectRecordingForRegion import getCorrectRecordingForRegion

import neuropixels_chronic_spikesorting_Bizley.outerLoopConfigs as outerLoopConfigs
import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.
from neuropixels_chronic_spikesorting_Bizley.sessionSetManagement.gatherSessionSetListForFerret import gatherSessionSetListForFerret
from neuropixels_chronic_spikesorting_Bizley.sessionSetManagement.SpecifySessionSetParameters import SpecifySessionSetParameters

def main():

    ## these are a bunch of parameters you should set before running spikesorting which are hopefully named in a self-explanatory way.
    saveMultirecInfoFile = True ### whether to save metadata. You'll only want to turn it off if you are testing a thing and don't want to overwrite an old file or something...
    overwriteRaw = False # determines if you recreate a raw that is already detected. Set to true if you want to remake it, false otherwise.
    overwriteChanMap = True # determines whether you overwrite the channel map. In my opinion this should always be on.
    overwriteSpikesorting = False # determines whether you redo spikesortings that are already done. Overwrite when you try new things, don't sh
    createCompressed = False # determines whether you create compressed raw. Good to do if you need the space, but all processing requires unpacking it.
    doSpikeSorting = True
    copyAuxData = False
    overwriteAuxData = False
    saveLFP = True ### saves an additional raw file which is focused on the LFP. Will be downsampled to 1000 Hz
    overwriteLFP = True

    #get sessions within map
    if all_VE_config.runOneSessionSetOldStyle:
        if not all_VE_config.SurveyOverride: ### if not a survey...
            NAS_SessionsWithinMap,Local_SessionsWithinMap = getSessionsWithinMap(outerLoopConfigs.NAS_SessionsInOrder,all_VE_config.NAS_session_path,all_VE_config.stream_id,all_VE_config.channel_map_to_use,outerLoopConfigs.Local_SessionsInOrder) ### possibly this just returns SessionsInOrder in a way resulting from a vestige. It used to get everything, but now... Anyway this can be marked for potential future deletion
        else: ### if survey, the order doesn't actually matter, and they will probably be "in order" anyway.
            NAS_SessionsWithinMap = outerLoopConfigs.NAS_SessionsInOrder
            Local_SessionsWithinMap = outerLoopConfigs.Local_SessionsInOrder
        toIterate = '' ### dummy variable
        sessionSetParams['badChannelList'] = all_VE_config.badChannelList
        sessionSetParams['channel_map_to_use'] = all_VE_config.channel_map_to_use
        breakitonpurpose ### in order to make this backward compatable, all relevant all_VE_config variables need to be set as dict variables within sessionSetParams
    else:
        recordingZones = gatherSessionSetListForFerret(all_VE_config.ferret) ### still doing one ferret at a time. Though there is potential to iterate them... Don't really see a good reason to do so.

        #sessionSetParams = SpecifySessionSetParameters(all_VE_config.ferret,SessionSetName,all_VE_config.regionLabels)
        toIterate = list(combination_product([all_VE_config.ferret],recordingZones,all_VE_config.regionLabels)) ### this function seems nondeterministic?
        toIterate = sorted(toIterate,key=lambda tup: tup[2]) ### first sort by region
        toIterate = sorted(toIterate,key=lambda tup: tup[1]) ### then sort by session set.
        pass
    for ixixi,sessionSet in enumerate(toIterate):
        if not all_VE_config.runOneSessionSetOldStyle:
            sessionSetParams = SpecifySessionSetParameters(sessionSet[0],sessionSet[1],sessionSet[2])
            NAS_SessionsWithinMap,Local_SessionsWithinMap = getSessionsWithinMapForSessionSet(sessionSetParams)
            (all_VE_config.output_folder / sessionSet[0] / sessionSet[1] / sessionSet[2]).mkdir(parents=True, exist_ok=True)
            if saveLFP:
                LFP_sessionSet_folder =  all_VE_config.LFP_folder / sessionSet[0] / sessionSet[1] / sessionSet[2]
                LFP_sessionSet_folder.mkdir(parents=True, exist_ok=True)

            errorFileName =  all_VE_config.output_folder / sessionSet[0] / sessionSet[1] / sessionSet[2] / "errorTracking.txt"
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
            imecFolderList = list(session.glob('*' + 'imec' + '*'))
            metaFileData,usedImro,NASprobeFolder = getCorrectRecordingForRegion(imecFolderList,sessionSetParams['channel_map_to_use'],localData=False)
            if metaFileData == None:
                with open(errorFileName, "a") as f:
                    print('no recording at ' + session_name + ' for ' + sessionSetParams['channel_map_to_use'], file=f) ### might want to print things at additional locations...
                continue # nothing to do here... ### TODO print to file. Probably today. Need to figure out everythin I want to print though.
            local_probeFolder = all_VE_config.local_session_path / Path(*NASprobeFolder.parts[-2:]) ### this is a local version of the top-evel folder of the spikeglx session
            local_probeFolder.mkdir(parents=True, exist_ok=True)
            if copyAuxData:
                if (not any(list(local_probeFolder.parent.glob('*.bin')))) | overwriteAuxData:
                    probable_ni_file = list(NASprobeFolder.parent.glob('*.bin'))[0] ### onebox might have to do this differently
                    probable_ni_meta_file = list(NASprobeFolder.parent.glob('*.meta'))[0] ### onebox might have to do this differently
                    target_ni_file = local_probeFolder.parent / probable_ni_file.name ### the place where we are sending it
                    target_ni_meta_file = local_probeFolder.parent / probable_ni_meta_file.name
                    shutil.copy(probable_ni_file,target_ni_file)
                    shutil.copy(probable_ni_meta_file,target_ni_meta_file)
                pass
            rawFolderName = local_probeFolder / Path('catgt_CorrectedRaw')
            if outerLoopConfigs.doSaturationReplace:
                NASrecording = si.read_spikeglx(NASprobeFolder, stream_id=usedImro)
                try:
                    windowsToSilenceArray = nullify_saturations(NASrecording, local_probeFolder=local_probeFolder) ### a numpy array with sample ranges that we want to exclude. Will be input into OverStrike later.
                except:
                    with open(errorFileName, "a") as f:
                        print('saturation detection failed for ' + session_name + ' for ' + sessionSetParams['channel_map_to_use'], file=f)
                        continue
            if (not any((list(rawFolderName.glob('*.bin'))))) | overwriteRaw: # if you don't have a raw or you say you want to overwrite it, we make a new raw.
                try:
                    makeAndProcessRaw(NAS_session_path = all_VE_config.NAS_session_path,session_name=session_name,stream_id=usedImro,local_session_path=all_VE_config.local_session_path,badChannelList=sessionSetParams['badChannelList'],catgt_location=outerLoopConfigs.catgt_location,rawFolderName=rawFolderName,overwriteRaw = overwriteRaw,createCompressed = createCompressed)
                    if outerLoopConfigs.doSaturationReplace: # requires saying you will overwrite. I hope that is natural.
                        applyOverStrike(list(rawFolderName.glob('*.bin'))[0],outerLoopConfigs.OverStrike_location,windowsToSilenceArray)
                except:
                    with open(errorFileName, "a") as f:
                        print('preprocessed raw creation failed for ' + session_name + ' for ' + sessionSetParams['channel_map_to_use'], file=f)
                        continue
            if saveLFP:
                LFP_session_folder = LFP_sessionSet_folder / session_name
                LFP_session_folder.mkdir(parents=True, exist_ok=True)
                try:
                    if (not any((list(LFP_session_folder.glob('*.bin'))))) | overwriteLFP:
                        getLFP(NAS_session_path = all_VE_config.NAS_session_path,session_name=session_name,stream_id=usedImro,LFP_session_folder=LFP_session_folder,badChannelList=sessionSetParams['badChannelList'],catgt_location=outerLoopConfigs.catgt_location,overwriteLFP = overwriteLFP,doSaturationReplace = outerLoopConfigs.doSaturationReplace,windowsToSilenceArray=windowsToSilenceArray)

                except:
                    with open(errorFileName, "a") as f:
                        print('LFP creation failed for ' + session_name + ' for ' + sessionSetParams['channel_map_to_use'], file=f)

                pass
            recording = si.read_spikeglx(rawFolderName, stream_id=usedImro) ### use si just to read the meta file.


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
            sorter_output_folder =  all_VE_config.output_folder / sessionSet[0] / sessionSet[1] / sessionSet[2] / session_name
            probe_name = sorter_output_folder / Path("chanMap.json")  ### I still need to figure out how to automatically generate this file. When I do, I can save it into "inputs" somewhere.
            if (not probe_name.is_file())|(overwriteChanMap):
                ks.io.save_probe(probe, str(sorter_output_folder / "chanMap.json"))

            ### do things related to the construction of a file which stores the recording information.
            multirec_info['name'].append(session_name)
            multirec_info['fs'].append(recording.get_sampling_frequency())
            multirec_info['n_samples'].append(recording.get_num_samples())
            multirec_info['duration'].append(recording.get_total_duration())

            multirec_info['start_time'].append(metaFileData['fileCreateTime']) ### it is very important that this metaFileData comes from the NAS! otherwise the fileCreateTime will be the time you catGTed it.

            multirec_info['multirec_start_sample'].append(0) ### this use to be integrated into a concatenation thing. That's why we bother with this.

            # multirec_info['multirec_stop_sample'].append(int(multirec_info['multirec_start_sample'][i] + (multirec_info['duration'][i].total_seconds() * multirec_info['fs'][i])))
            multirec_info['multirec_stop_sample'].append(
                multirec_info['multirec_start_sample'][0] + (multirec_info['n_samples'][0]))
            multirec_info['fullpath_as_string'].append(list(NASprobeFolder.glob('*.bin'))[0].__str__())
            chan_dict = get_channelmap_names(session)  ### I should figure out how to get rid of this... Works fine for now.
            chan_map_name = chan_dict[session_name + "_" + usedImro[:-3]]

            if chan_map_name == sessionSetParams['channel_map_to_use']:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
                if chan_map_name in dict_of_recordings:
                    dict_of_recordings[chan_map_name].append(recording)
                else:
                    dict_of_recordings[chan_map_name] = [recording]

            multirecordings = dict_of_recordings
            if not all_VE_config.SurveyOverride: ### If we aren't doing SurveyOverride, then we want to keep track of the channel_map we intend to use so we can see if there is a mismatch later.
                multirecordingInput = multirecordings[sessionSetParams['channel_map_to_use']][0]

            if saveMultirecInfoFile:  # would like to seperate this from doing the preprocessing I think...
                df_rec = pd.DataFrame(multirec_info)
                df_rec.to_csv(sorter_output_folder / 'multirec_info.csv',
                              index=False)  # this is the earliest phy folder around and may be a problem... This saves the multirec info... Also this stuff might be better saved in spikesortingInput...

            ### note I am currently focusing cpu us
            if doSpikeSorting&((overwriteSpikesorting)|(not any(list(sorter_output_folder.glob('params.py'))))): ### overwrite will currently not remove bombcell results and such from the folder. It should.
                try:
                    ks.run_kilosort(settings, probe=None, probe_name=probe_name, filename=None,
                         data_dir=rawFolderName, file_object=None, results_dir=sorter_output_folder,
                         data_dtype=None, do_CAR=True, invert_sign=False, device=None,
                         progress_bar=None, save_extra_vars=True, clear_cache=False,
                         save_preprocessed_copy=False, bad_channels=None, shank_idx=None,
                         verbose_console=True, verbose_log=True, torch_thread_lim=None)
                except:
                    with open(errorFileName, "a") as f:
                        print('spikesorting failed for ' + session_name + ' for ' + sessionSetParams['channel_map_to_use'], file=f)


if __name__ == '__main__':
    main()