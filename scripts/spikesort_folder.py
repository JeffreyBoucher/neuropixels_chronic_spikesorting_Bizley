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


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names
from neuropixels_chronic_spikesorting_Bizley.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing
from neuropixels_chronic_spikesorting_Bizley.helpers.npyx_metadata_fct import load_meta_file

def main():

    if True: # contains folder and file arguments, including higher-level ones like ferret name. Likely to be specific to my project


        #recordingZone = 'PFC_Boule_Borders_Top_GroundrefIThink'
        #recordingZone = 'PFC_Boule_Borders_Top'
        # recordingZone = 'PFC_shank0_Challah'
        #recordingZone = 'PFC_shank3_Challah'
        #recordingZone = 'PFC_shank3_Challah_bottom'
        #recordingZone = "ACx_Challah"
        recordingZone = "ACx_Challah_top_groundref"
        output_folder = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output')
        sessionsToDo = 'all' # vestige from get_drift_per_session

    if True: # contains si arguments
        desired_n_jobs = 16
        si.set_global_job_kwargs(n_jobs=desired_n_jobs)
        doRemoveBadChannels = 1  # set as 1 if you are sorting, set as 0 if you want to save a thing for the ks4 gui or something
        skipStuffThatKSGUIDoes = 0 # even still we don't want to do this, I think, because of the noise estimation.

    if True: # arguments handling how much code to run
        skipStuffThatKSGUIDoes = 0  # KS GUI does CAR and bandpass filter and it is a bit opaque how to turn off the latter.

        whitenManually = 0
        skipSessionsAlreadyDone = 1
        doSpikeSorting = 1 # covers both preprocessing and spikesorting in this case.
        doPostprocessing = 0 # shouldn't need to by current plans

    if not doSpikeSorting:
        print('warning: not doing any spikesorting. Doing postprocessing instead')

    if True: # contains probemap specific stuff. Highly specific to my project. I certainly will functionalize this soon so I can use it in both this and the drift correction script.
        if recordingZone == 'ACx_Challah_top_groundref':
            stream_id = 'imec0.ap'
            sessionSetLabel = 'All_ACx_Top_groundref_MonthOfMay2024'
            channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro'
            #channel_map_to_use_other_ref = 'Challah_top_b1_horizontal_band_joint_tip.imro' ### this wouldn't actually work...
            badChannelList = [66,105,149,170,175,209,210,239,354,369]
            ferret = 'F2302_Challah'
        elif recordingZone == 'PFC_shank0_Challah':
            stream_id = 'imec1.ap'
            #sessionSetLabel = 'PFC_shank0'
            sessionSetLabel = 'PFC_shank0'
            channel_map_to_use = 'Challah_top_PFC_shank0.imro'
            badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
            # something else also. Need to read metadata ### I don't know what I meant by this and it doesn't seem to be true
            ferret = 'F2302_Challah'
        elif recordingZone == 'PFC_shank3_Challah':
            stream_id = 'imec1.ap'
            sessionSetLabel = 'PFC_shank3'
            channel_map_to_use = 'Challah_top_PFC_shank3.imro'
            badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
            # something else also. Need to read metadata
            ferret = 'F2302_Challah'
        elif recordingZone == 'PFC_shank3_Challah_bottom':
            stream_id = 'imec1.ap'
            sessionSetLabel = 'PFC_shank3_bottom'
            channel_map_to_use = 'Challah_bottom_PFC_shank3_tip_ref.imro'
            badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339] ### I double checked, this is still exactly right a year later.
            # something else also. Need to read metadata
            ferret = 'F2302_Challah'
        elif recordingZone == 'ACx_Boule':
            stream_id = 'imec1.ap'
            sessionSetLabel = 'All_ACx_Top'
            channel_map_to_use = 'Boule_top_ACx_tipref.imro'
            # channel_map_to_use_other_ref = ''
            ferret = 'F2301_Boule'
        elif recordingZone == 'PFC_Boule_Center_Top':
            stream_id = 'imec0.ap'
            sessionSetLabel = 'PFC_Shanks_1_2'
            channel_map_to_use = 'Boule_PFC_Shanks_1_2_tipref.imro'
            ferret = 'F2301_Boule'
        elif recordingZone == "PFC_Boule_Borders_Top":
            stream_id = 'imec0.ap'
            sessionSetLabel = 'PFC_Shanks_0_3'
            channel_map_to_use = 'Boule_PFC_Shanks_0_3_tipRef.imro'
            badChannelList = [19,128,161,291,315]
            ferret = 'F2301_Boule'
        elif recordingZone == "PFC_Boule_Borders_Top_GroundrefIThink":
            stream_id = 'imec0.ap'
            sessionSetLabel = 'PFC_Shanks_0_3'
            channel_map_to_use = 'Boule_PFC_Shanks_0_3.imro'
            badChannelList = [19, 128, 161, 291, 315]
            # channel_map_to_use_other_ref = ''
            ferret = 'F2301_Boule'

    if True: # manages the highest-level selection of sessions via regex. A bit outdated now that session-sets are implemented.
        if sessionSetLabel == 'All_ACx_Top':
            sessionString = '[0-9][0-9]*' ### this actually selects more than just the top
        elif sessionSetLabel == 'Tens_Of_June':
            sessionString = '1[0-9]06*'
        elif sessionSetLabel == 'TheFirstDay':
            sessionString = '1305*'
        elif sessionSetLabel == 'TheFirstSession':
            sessionString = '1305*AM*'
        elif 'MonthOfMay2024' in sessionSetLabel:
            sessionString = '[0-9][0-9]052024*'
        else:
            sessionString = '[0-9][0-9]*'
    session_path = Path('Z:/Data/Neuropixels/' + ferret)
    SessionsInOrder = sort_np_sessions(list(session_path.glob(sessionString)))
    sessionSetName = 'everythingAllAtOnce'

    sessionsWithinMap = []
    for i,session in enumerate(SessionsInOrder):
        session_name = session.name
        dp = session_path / session_name
        chan_dict = get_channelmap_names(dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
        if (session_name + "_" + stream_id[:-3]) in chan_dict:
            if any(v == channel_map_to_use for v in chan_dict.values()):
                sessionsWithinMap.append(session)
        else:
            print('a bug you should solve')
            pass



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
        working_dir = output_folder / 'tempDir' / ferret / session_name
        dp = session_path / session_name

        probeFolder = list(dp.glob('*' + stream_id[:-3]))
        probeFolder = probeFolder[0]  # name of probe folder
        if doSpikeSorting:
            # recording = si.read_cbin_ibl(probeFolder)  # for compressed
            # if windowsToSilenceArray == 1: #get csv saturation files
            #     windowsToSilenceArray = np.loadtxt(Path(f'D:/CSV_Saturation/F2302_Challah/{session_name}/{session_name}_{stream_id[:-3]}/saturatedZones.csv'))
            # else:
            #     windowsToSilenceArray = 1
            local_probeFolder = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs') / Path(*probeFolder.parts[1:]) #useful if you run the find saturation function
            #local_probeFolder = Path(f'D:/CSV_Saturation/F2302_Challah/{session_name}/{session_name}_{stream_id[:-3]}/')
            recording = si.read_spikeglx(probeFolder, stream_id=stream_id)  # for uncompressed
            recording = spikeglx_preprocessing(recording, doRemoveBadChannels=doRemoveBadChannels,
                                               skipStuffThatKSGUIDoes=skipStuffThatKSGUIDoes,
                                               local_probeFolder=local_probeFolder, badChannelList=badChannelList)
            ### do things related to the construction of a file which stores the recording information.
            multirec_info['name'].append(session_name)
            multirec_info['fs'].append(recording.get_sampling_frequency())
            multirec_info['n_samples'].append(recording.get_num_samples())
            multirec_info['duration'].append(recording.get_total_duration())

            meta = load_meta_file(probeFolder / (session_name + '_t0.' + stream_id + '.meta'))
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
            chan_dict = get_channelmap_names(
                dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
            chan_map_name = chan_dict[session_name + "_" + stream_id[:-3]]

            if chan_map_name == channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
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
            multirecordingInput = multirecordings[channel_map_to_use][0]
        # recording = si.concatenate_recordings(dict_of_recordings)

        output_folder_temp = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name
        output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone / sessionSetLabel / sessionSetName / session_name
        phy_folder = output_folder_sorted / 'KiloSortSortingExtractor' / 'phy_folder'  # probably.
        phy_folder.mkdir(parents=True, exist_ok=True)
        if whitenManually:
            whitened_recording = si.whiten(recording=multirecordingInput,dtype="float32") # I stopped trying to force this to be intscaled to 200 because it never friggin worked. Something I can consider is doing a few of the preprocessing steps within kilosort so that I don't run into an issue here...
            usedRecording = whitened_recording
        else:
            usedRecording = multirecordingInput
        if skipSessionsAlreadyDone:
            if (output_folder_temp / Path("tempDir")).is_dir():
                continue
        if doSpikeSorting:
            df_rec = pd.DataFrame(multirec_info)
            df_rec.to_csv(phy_folder / 'multirec_info.csv',
                          index=False)  # this is the earliest phy folder around and may be a problem... This saves the multirec info

        #try:
        if doSpikeSorting:
            sorting = spikesorting_pipeline(
                usedRecording,
                output_folder=output_folder_temp,
                sorter='kilosort4',
                concatenated=True
            )
        else:
            sorting = si.read_sorter_folder(output_folder_temp / 'tempDir' / 'kilosort4_output')
        output_dir = output_folder_sorted
        if doPostprocessing:  ### turn off in order to do postprocessing in a seperate job later. Frees up gpu, but fundamentally depends on size of job.
            sorting = spikesorting_postprocessing(sorting, output_folder=output_dir)

        # except Exception as e:
        #     print(f'Error processing {recordingZone}: {e}')



if __name__ == '__main__':
    main()