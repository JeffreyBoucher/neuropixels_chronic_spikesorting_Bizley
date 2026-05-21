from pathlib import Path
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import torch
import spikeinterface.extractors as se
import os
import numpy as np
import spikeinterface.full as si
import spikeinterface.sortingcomponents.motion as sm
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion

import neuropixels_chronic_spikesorting_Bizley.outerLoopConfigs as outerLoopConfigs
import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names
from neuropixels_chronic_spikesorting_Bizley.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing
from neuropixels_chronic_spikesorting_Bizley.helpers.npyx_metadata_fct import load_meta_file

def main():
    setsOfSessionsPerGrouping = []
    for sessionSetCount, currentSetOfSessions in enumerate(outerLoopConfigs.NAS_SetsOfConcatenatedSessions):  # first, determine the sessions which require further analysis. ### this tells me that yes I do want to create dummy files locally even if I am NAS based
            sessionsWithinMap = []
            for i, session in enumerate(currentSetOfSessions):
                session_name = session.name
                if (all_VE_config.frequencyOfConcatenation == 'weekly_heuristic') & (not i):
                    sessionSetName = 'weekOf' + session_name[4:8] + session_name[2:4] + session_name[0:2]  # name after first day of week. Also, swap to year month day so that things are alphabetical
                elif (not (all_VE_config.frequencyOfConcatenation == 'weekly_heuristic')) & (not i):
                    sessionSetName = session_name
                print(f'Processing {sessionSetName}')
                dp = all_VE_config.NAS_session_path / session_name
                chan_dict = get_channelmap_names(dp)
                if (session_name + "_" + all_VE_config.stream_id[:-3]) in chan_dict:
                    if any(v == all_VE_config.channel_map_to_use for v in chan_dict.values()):
                        sessionsWithinMap.append(session)
                else:
                    print('a bug you should solve')
                    pass

            if any(sessionsWithinMap):
                setsOfSessionsPerGrouping.append(sessionsWithinMap)
            elif all_VE_config.SurveyOverride:
                setsOfSessionsPerGrouping.append(
                    currentSetOfSessions)  ### if Survey, map will constantly change and everything should be well-specified, so we don't need to consider map in the same way...


    for sessionSetCount,currentSetOfSessions in enumerate(setsOfSessionsPerGrouping): #with weekly heuristic, each current set of session is a list of path objects
        if not all_VE_config.sessionsToDo == 'all':
            if not (sessionSetCount in all_VE_config.sessionsToDo):
                continue
        ### Make a file that keeps track of the recording info
        # I think I have already done this in outerLoopConfigs, and will comment this out until I know I don't really need it.
        # multirec_info = {'name': [],
        #                  'start_time': [],
        #                  # 'stop_time': [],
        #                  'duration': [],
        #                  'fs': [],
        #                  'n_samples': [],
        #                  'multirec_start_sample': [],
        #                  'multirec_stop_sample': [],
        #                  'fullpath_as_string': []}
        dict_of_recordings = {}
        sessionLoopBreakFlag = False
        for i,session in enumerate(currentSetOfSessions):
            session_name = session.name
            if (all_VE_config.frequencyOfConcatenation == 'weekly_heuristic') & (not i):
                sessionSetName = 'weekOf' + session_name[4:8] + session_name[2:4] + session_name[0:2]  # name after first day of week. Also, swap to year month day so that things are alphabetical
            elif (not (all_VE_config.frequencyOfConcatenation == 'weekly_heuristic')) & (not i):
                sessionSetName = session_name
            print(f'Processing {sessionSetName}')
            working_dir = all_VE_config.output_folder / 'tempDir' / all_VE_config.ferret / session_name
            NAS_currentDataPath = all_VE_config.NAS_session_path / session_name
            chan_dict = get_channelmap_names(NAS_currentDataPath)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
            chan_map_name = chan_dict[session_name + "_" + all_VE_config.stream_id[:-3]]
            if (not (chan_map_name == all_VE_config.channel_map_to_use)):
                print('But not really, because of the probe map')
                print('this currently handles weeks where I switch maps very badly!')
                sessionLoopBreakFlag = True
                break
            probeFolder = list(NAS_currentDataPath.glob('*' + all_VE_config.stream_id[:-3]))
            probeFolder = probeFolder[0]
            if outerLoopConfigs.doPreprocessing:
                # recording = si.read_cbin_ibl(probeFolder)  # for compressed
                local_probeFolder = all_VE_config.local_session_path / Path(*probeFolder.parts[-2:])
                recording = si.read_spikeglx(probeFolder, stream_id=all_VE_config.stream_id) # for uncompressed
                recording = spikeglx_preprocessing(recording, doRemoveBadChannels=outerLoopConfigs.doRemoveBadChannels,skipStuffThatKSGUIDoes=outerLoopConfigs.skipStuffThatKSGUIDoes,local_probeFolder=local_probeFolder,badChannelList=all_VE_config.badChannelList,bin_s_sessionCat=outerLoopConfigs.bin_s_sessionCat,doSaturationReplace=outerLoopConfigs.doSaturationReplace,silenceOrNoiseReplace=outerLoopConfigs.silenceOrNoiseReplace_sessionwise,floatDataTypeForDriftCorrection=True)
                ### do things related to the construction of a file which stores the recording information.
                outerLoopConfigs.multirec_info['name'].append(session_name)
                outerLoopConfigs.multirec_info['fs'].append(recording.get_sampling_frequency())
                outerLoopConfigs.multirec_info['n_samples'].append(recording.get_num_samples())
                outerLoopConfigs.multirec_info['duration'].append(recording.get_total_duration())

                meta = load_meta_file(probeFolder / (session_name + '_t0.' + all_VE_config.stream_id + '.meta'))
                outerLoopConfigs.multirec_info['start_time'].append(meta['fileCreateTime'])

                if i == 0:
                    outerLoopConfigs.multirec_info['multirec_start_sample'].append(0)
                else:
                    # multirec_info['multirec_start_sample'].append(int(
                    #     multirec_info['multirec_start_sample'][i-1] + (multirec_info['duration'][i-1].total_seconds() * multirec_info['fs'][i-1])+1))

                    outerLoopConfigs.multirec_info['multirec_start_sample'].append(
                        outerLoopConfigs.multirec_info['multirec_start_sample'][i - 1] + (outerLoopConfigs.multirec_info['n_samples'][i - 1]) + 1)

                # multirec_info['multirec_stop_sample'].append(int(multirec_info['multirec_start_sample'][i] + (multirec_info['duration'][i].total_seconds() * multirec_info['fs'][i])))
                outerLoopConfigs.multirec_info['multirec_stop_sample'].append(
                    outerLoopConfigs.multirec_info['multirec_start_sample'][i] + (outerLoopConfigs.multirec_info['n_samples'][i]))
                outerLoopConfigs.multirec_info['fullpath_as_string'].append(list(probeFolder.glob('*.bin'))[0].__str__())

                if chan_map_name == all_VE_config.channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
                    if chan_map_name in dict_of_recordings:
                        dict_of_recordings[chan_map_name].append(recording)
                    else:
                        dict_of_recordings[chan_map_name] = [recording]

                else:
                    print('this line should never be reached')
            # week_session_correspondance.append([week, i+last_session_previous_week]) ###TESTIFINEED ### I do need but hopefully not forever
            # if i==len(currentSetOfSessions)-1: ###TESTIFINEED
            #     last_session_previous_week+=i+1 ###TESTIFINEED
        if sessionLoopBreakFlag:
            sessionLoopBreakFlag = False
            continue
        if outerLoopConfigs.doPreprocessing:
            if len(currentSetOfSessions) > 1:
                multirecordings = {channel_map: si.concatenate_recordings(dict_of_recordings[channel_map]) for channel_map in dict_of_recordings}
                multirecordings = {channel_map: multirecordings[channel_map].set_probe(dict_of_recordings[channel_map][0].get_probe())  for channel_map in multirecordings}
                multirecordingInput = multirecordings[all_VE_config.channel_map_to_use]
            else:
                multirecordings = dict_of_recordings
                multirecordingInput = multirecordings[all_VE_config.channel_map_to_use][0]
        # recording = si.concatenate_recordings(dict_of_recordings)

        # output_folder_temp = all_VE_config.output_folder / 'tempDir' / all_VE_config.ferret / all_VE_config.recordingZone / all_VE_config.sessionSetLabel / sessionSetName
        # output_folder_sorted = all_VE_config.output_folder / 'spikesorted' / all_VE_config.ferret / all_VE_config.recordingZone / all_VE_config.sessionSetLabel / sessionSetName
        output_folder_temp = all_VE_config.output_folder / 'tempDir' / all_VE_config.ferret / all_VE_config.sessionSetLabel
        output_folder_sorted = all_VE_config.output_folder / 'spikesorted' / all_VE_config.ferret / all_VE_config.sessionSetLabel
        motionMapFolder_used = all_VE_config.motionMapFolder / 'spikesorted' / all_VE_config.ferret / all_VE_config.sessionSetLabel

        phy_folder = output_folder_sorted / 'KiloSortSortingExtractor' / 'phy_folder'  # probably.
        ### save multirec_info
        phy_folder.mkdir(parents=True, exist_ok=True)
        #multirecordingInput = multirecordingInput.remove_channels(badChannelList)

        if all_VE_config.doMultipleShanks: ### Currently, should always be true.
            if outerLoopConfigs.doPreprocessing: # loading and processing are pretty well split this time, the result of the load will be in a different format. Plotting will be load-relegated, drift saving will be here (maybe later, both)
            ### first split shanks.
                for ixi,currentRecording in enumerate(multirecordingInput.recording_list): ### malheurusement I need to do my shank splitting on each session pre-concatenation... I can either do that here or earlier... But it actually is not so big a deal to just do it here, let's do it here.
                    if not ixi: # initialize on first loop.
                        multirecordingSplit = currentRecording.split_by("group") # this will be appended to
                    else:
                        recordingSplit = currentRecording.split_by("group")
                        for shankCount, thisShank in enumerate(recordingSplit):
                            multirecordingSplit[shankCount] = si.concatenate_recordings([multirecordingSplit[shankCount],recordingSplit[shankCount]])
                ## save shank label for all used channels
                shank_ids_all_channels = currentRecording._properties["group"]
                motionMapFolder_used.mkdir(parents=True, exist_ok=True)
                np.save(motionMapFolder_used / "shank_ids_all_channels", shank_ids_all_channels)
                ## save the "sessionEnds" times. Because for an si specific reason, the individual sessions don't play nice when you concatenate them...
                sessionEnds = np.asarray(multirecordingInput._recording_segments[0].all_length)/30000
                sessionEnds = np.cumsum(sessionEnds)
                np.save(motionMapFolder_used / "sessionEnds", sessionEnds)



                job_kwargs = dict(chunk_duration='1s',n_jobs=outerLoopConfigs.desired_n_jobs,progress_bar=True) #,
                ### motion correction
                for shankCount, enumerateIsNotWorkingSoIgnoreThis in enumerate(multirecordingSplit):

                    multirecordingThisShank = multirecordingSplit[shankCount]

                    peaks = detect_peaks(recording=multirecordingThisShank, method="locally_exclusive", peak_sign="neg",
                                             detect_threshold=5.0, exclude_sweep_ms=0.1, radius_um=50,
                                             **job_kwargs)  # seems like there isn't a unique peak detection algorithm for ks? I don't actually think that's true, I believe they use templates... But with SI, maybe not. I should maybe just look into the high level function and see what is done
                    peak_locations = localize_peaks(recording=multirecordingThisShank, peaks=peaks, method="grid_convolution",
                                                    weight_method={"mode": "gaussian_2d",
                                                                   "sigma_list_um": np.linspace(5, 25, 5)}, **job_kwargs)
                    motion = estimate_motion(recording=multirecordingThisShank, peaks=peaks,
                                             peak_locations=peak_locations, method="iterative_template", bin_s=outerLoopConfigs.bin_s_sessionCat,
                                             rigid=False, win_step_um=50.0, win_scale_um=100.0, hist_margin_um=0,
                                             win_shape="gaussian",
                                             num_amp_bins=5)  # this works on my test case. ks defaults, but with win_scale_um=100.0,  win_step_um=50.0, win_shape="gaussian", num_amp_bins=5, bin_s=6.0
                    rec_corrected = interpolate_motion(recording=multirecordingThisShank, motion=motion,
                                                               border_mode="force_extrapolate",
                                                               spatial_interpolation_method="kriging", sigma_um=20., p=2)


                    peak_locations_after = sm.correct_motion_on_peaks(peaks, peak_locations, motion, rec_corrected)

                    if outerLoopConfigs.savePreprocessing: ### need to make changes to this to either save each shank seperately or to save them all together... Should also have a list of channel-group matchups.
                        shankfolderName = Path("Shank_" + str(shankCount)) ### I did not check if group number and absolute shank number align. Probably they do.
                        picklePath = output_folder_sorted / shankfolderName
                        picklePath.mkdir(parents=True, exist_ok=True)
                        pickleName = 'preprocess_results.pkl'
                        if (not (picklePath / pickleName).is_file())|outerLoopConfigs.overwritePreprocessing:
                            pickleJar = dict(peaks=peaks, peak_locations=peak_locations, multirecordingThisShank=multirecordingThisShank,rec_corrected=rec_corrected,motion=motion,peak_locations_after=peak_locations_after)
                            with open(picklePath / pickleName, 'wb') as file:
                                pickle.dump(pickleJar, file)


                    if outerLoopConfigs.calculateSessionMotionDisplacement:
                        # sessionEnds = np.asarray(multirecordingThisShank._recording_segments[0].all_length)/30000 ### calculated this earlier because you can't do this per-shank for some reason.
                        # sessionEnds = np.cumsum(sessionEnds)
                        # Recording average motion over a week
                        motion_space = motion.displacement
                        motion_space = motion_space[0]
                        bins_per_session=[]
                        bins_per_session_numpyStyle = np.zeros((len(sessionEnds),motion_space.shape[1]))
                        for count,i in enumerate(sessionEnds):
                            end_session_bin= np.where(motion.temporal_bin_edges_s[0] < i)
                            endSessionBin = end_session_bin[0][-1]-1
                            if i==sessionEnds[0]:
                                bins_per_session.append(np.mean(motion_space[0:endSessionBin], axis=0))
                                bins_per_session_numpyStyle[count,:] = np.mean(motion_space[0:endSessionBin], axis=0)
                            else:
                                bins_per_session.append(np.mean(motion_space[prev_end_session_bin:endSessionBin], axis=0))
                                bins_per_session_numpyStyle[count,:] = np.mean(motion_space[prev_end_session_bin:endSessionBin], axis=0)
                            prev_end_session_bin = endSessionBin
                        motion_average = np.mean(motion_space, axis=0)
                        record_motion_this_shank = [] # used to be record_motion_weeks, but we no longer actually want that format anyway and it was causing bugs.
                        record_space_this_shank = []
                        record_motion_sessions = []
                        record_space_bins_sessions = []
                        if not sessionSetCount:
                            record_motion_this_shank.append(motion_average)
                            record_space_this_shank.append(motion.spatial_bins_um)
                            for i in bins_per_session:
                                record_motion_sessions.append(i)
                                record_space_bins_sessions.append(motion.spatial_bins_um)
                        else:
                            record_motion_this_shank.append(motion_average)
                            record_space_this_shank.append(motion.spatial_bins_um + record_motion_this_shank[sessionSetCount-1])
                            for i in bins_per_session:
                                record_motion_sessions.append(i)
                                record_space_bins_sessions.append(motion.spatial_bins_um + record_motion_this_shank[sessionSetCount-1])
                            #record_motion.append(record_motion[sessionSetCount-1]+motion_average)
                        #record_space_bins.append(motion.spatial_bins_um)
                        breakPointSpot = "here"


                    ### the following will have been designed to have looped multiple times maybe, but we don't need that functionality.
                        # week+=1 # just in case, but I don't see this variable... ###TESTIFINEED

                        record_motion_this_shank = np.array(record_motion_this_shank)
                        record_space_this_shank = np.array(record_space_this_shank)
                        N, M = record_motion_this_shank.shape
                        rows = []
                        for i in range(N):
                            for j in range(M):
                                rows.append((all_VE_config.sessionsToDo[i], record_motion_this_shank[i, j], record_space_this_shank[i, j])) ### WARNING use of sessionsToDo may be wrong here

                        df_weeks = pd.DataFrame(rows, columns=["session_week", "motion", "center_space_bin"])

                        record_motion_sessions = np.array(record_motion_sessions)
                        record_space_bins_sessions = np.array(record_space_bins_sessions)
                        N, M = record_motion_sessions.shape
                        rows = []
                        for i in range(N):
                            for j in range(M):
                                rows.append((i, record_motion_sessions[i, j], record_space_bins_sessions[i, j]))

                        df_sessions = pd.DataFrame(rows, columns=["session", "motion", "center_space_bin"])
                        if outerLoopConfigs.resaveMotionIfLoadingPreprocessing:
                            motionSaveFolder = motionMapFolder_used / shankfolderName
                            motionSaveFolder.mkdir(parents=True, exist_ok=True)


                            # Save to CSV
                            # df_weeks.to_csv(motionSaveFolder / "motion_weeks.csv", index=False)
                            df_sessions.to_csv(motionSaveFolder / "motion_sessions.csv", index=False) ### This should be the only one to care about but currently the others are needed for things to work.
                            # week_session_correspondance = np.array(week_session_correspondance)
                            # np.save(motionSaveFolder / "session_to_week_id", week_session_correspondance)



                            ## below is new thing to help track which shank is on which channel.
                            channelIDsThisShank = multirecordingThisShank.channel_ids[:]
                            numericID = np.zeros(len(channelIDsThisShank),dtype=int)
                            for i,stringID in enumerate(channelIDsThisShank):
                                numericID[i] = int(stringID[(stringID.find('AP')+2):]) # will not work if there aren't zeros in front.
                            # Save to CSV
                            np.save(motionSaveFolder / "Absolute_IDs_this_slice", numericID)

                        pass
            else:
                for folderCount,shankfolderName in enumerate(list(motionMapFolder_used.glob('Shank_[0-9]'))):

                    picklePath = output_folder_sorted / shankfolderName
                    pickleName = 'preprocess_results.pkl'
                    if (not (picklePath / pickleName).is_file())|outerLoopConfigs.overwritePreprocessing:
                        with open(picklePath / pickleName, 'rb') as file:
                            pickleJar = pickle.load(file)
                    peaks = pickleJar["peaks"]
                    peak_locations = pickleJar["peak_locations"]
                    multirecordingThisShank = pickleJar["multirecordingThisShank"]
                    rec_corrected = pickleJar["rec_corrected"]
                    motion = pickleJar["motion"]
                    peak_locations_after = pickleJar["peak_locations_after"]

                    if outerLoopConfigs.checkMotionPlotsOnline:
                        matplotlib.use(outerLoopConfigs.matplotlibGUItype)
                        scatterArray = np.zeros((len(peak_locations), 3))
                        xLocationToCheckShank = np.zeros((len(peak_locations), 1))
                        for iiii in range(0, len(peak_locations)):
                            scatterArray[iiii, 0] = peaks[iiii][0]

                            scatterArray[iiii, 1] = peak_locations[iiii][1]
                            scatterArray[iiii, 2] = peak_locations_after[iiii][1]
                            xLocationToCheckShank[iiii] = peak_locations[iiii][0]
                        plt.figure()
                        plt.scatter([1, 2, 3], [2, 4, 3])  # for some reason this helps the other figures load...
                        plt.show()
                        plt.figure()
                        plt.scatter(scatterArray[:, 0] / 30000, scatterArray[:, 1], s=0.0005, c="black")
                        plt.gca().invert_yaxis()
                        plt.show()
                        plt.figure()
                        plt.scatter(scatterArray[:, 0] / 30000, scatterArray[:, 2], s=0.0005, c="black")
                        plt.gca().invert_yaxis()
                        plt.show()
                        plt.figure()
                        plt.hist(xLocationToCheckShank,100) # for some reason this helps the other figures load...
                        plt.show()

                        plt.figure()
                        plt.scatter([1, 2, 3], [2, 4, 3])  # for some reason this helps the other figures load...
                        plt.show()


                        breakPointSpot = "here"

                    if outerLoopConfigs.calculateSessionMotionDisplacement:
                        # sessionEnds = np.asarray(multirecordingThisShank._recording_segments[0].all_length)/30000
                        # sessionEnds = np.cumsum(sessionEnds)
                        # Recording average motion over a week
                        sessionEnds = np.load((motionMapFolder_used / "sessionEnds.npy")) ### must be loaded
                        motion_space = motion.displacement
                        motion_space = motion_space[0]
                        bins_per_session=[]
                        bins_per_session_numpyStyle = np.zeros((len(sessionEnds),motion_space.shape[1]))
                        for count,i in enumerate(sessionEnds):
                            end_session_bin= np.where(motion.temporal_bin_edges_s[0] < i)
                            endSessionBin = end_session_bin[0][-1]-1
                            if i==sessionEnds[0]:
                                bins_per_session.append(np.mean(motion_space[0:endSessionBin], axis=0))
                                bins_per_session_numpyStyle[count,:] = np.mean(motion_space[0:endSessionBin], axis=0)
                            else:
                                bins_per_session.append(np.mean(motion_space[prev_end_session_bin:endSessionBin], axis=0))
                                bins_per_session_numpyStyle[count,:] = np.mean(motion_space[prev_end_session_bin:endSessionBin], axis=0)
                            prev_end_session_bin = endSessionBin
                        motion_average = np.mean(motion_space, axis=0)
                        record_motion_this_shank = [] # used to be record_motion_weeks, but we no longer actually want that format anyway and it was causing bugs.
                        record_space_this_shank = []
                        record_motion_sessions = []
                        record_space_bins_sessions = []
                        if not sessionSetCount:
                            record_motion_this_shank.append(motion_average)
                            record_space_this_shank.append(motion.spatial_bins_um)
                            for i in bins_per_session:
                                record_motion_sessions.append(i)
                                record_space_bins_sessions.append(motion.spatial_bins_um)
                        else:
                            record_motion_this_shank.append(motion_average)
                            record_space_this_shank.append(motion.spatial_bins_um + record_motion_this_shank[sessionSetCount-1])
                            for i in bins_per_session:
                                record_motion_sessions.append(i)
                                record_space_bins_sessions.append(motion.spatial_bins_um + record_motion_this_shank[sessionSetCount-1])
                            #record_motion.append(record_motion[sessionSetCount-1]+motion_average)
                        #record_space_bins.append(motion.spatial_bins_um)
                        breakPointSpot = "here"


                    ### the following will have been designed to have looped multiple times maybe, but we don't need that functionality.
                        # week+=1 # just in case, but I don't see this variable... ###TESTIFINEED

                        record_motion_this_shank = np.array(record_motion_this_shank)
                        record_space_this_shank = np.array(record_space_this_shank)
                        N, M = record_motion_this_shank.shape
                        rows = []
                        for i in range(N):
                            for j in range(M):
                                rows.append((all_VE_config.sessionsToDo[i], record_motion_this_shank[i, j], record_space_this_shank[i, j])) ### WARNING use of sessionsToDo may be wrong here

                        df_weeks = pd.DataFrame(rows, columns=["session_week", "motion", "center_space_bin"])

                        record_motion_sessions = np.array(record_motion_sessions)
                        record_space_bins_sessions = np.array(record_space_bins_sessions)
                        N, M = record_motion_sessions.shape
                        rows = []
                        for i in range(N):
                            for j in range(M):
                                rows.append((i, record_motion_sessions[i, j], record_space_bins_sessions[i, j]))

                        df_sessions = pd.DataFrame(rows, columns=["session", "motion", "center_space_bin"])
                        if outerLoopConfigs.resaveMotionIfLoadingPreprocessing:
                            motionSaveFolder = motionMapFolder_used / shankfolderName
                            motionSaveFolder.mkdir(parents=True, exist_ok=True)

                            # df_weeks.to_csv(motionSaveFolder / "motion_weeks.csv", index=False)
                            df_sessions.to_csv(motionSaveFolder / "motion_sessions.csv", index=False) ### This should be the only one to care about but currently the others are needed for things to work.
                            # week_session_correspondance = np.array(week_session_correspondance)
                            # np.save(motionSaveFolder / "session_to_week_id", week_session_correspondance)

                            ## below is new thing to help track which shank is on which channel.
                            channelIDsThisShank = multirecordingThisShank.channel_ids[:]
                            numericID = np.zeros(len(channelIDsThisShank),dtype=int)
                            for i,stringID in enumerate(channelIDsThisShank):
                                numericID[i] = int(stringID[(stringID.find('AP')+2):]) # will not work if there aren't zeros in front.
                            # Save to CSV
                            np.save(motionSaveFolder / "Absolute_IDs_this_slice", numericID)

                        pass



if __name__ == '__main__':
    main()


