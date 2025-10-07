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


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names
from neuropixels_chronic_spikesorting_Bizley.spikesorting import spikesorting_pipeline, spikesorting_postprocessing, spikeglx_preprocessing
from neuropixels_chronic_spikesorting_Bizley.helpers.npyx_metadata_fct import load_meta_file

def main():
    matplotlib.use('TkAgg')
    if True: # contains folder and file arguments, including higher-level ones like ferret name. Likely to be specific to my project


        #recordingZone = 'PFC_Boule_Borders_Top_GroundrefIThink'
        #recordingZone = 'PFC_Boule_Borders_Top'
        recordingZone = 'PFC_shank0_Challah'
        #recordingZone = 'PFC_shank3_Challah'
        #recordingZone = "ACx_Challah"
        frequencyOfConcatenation = 'weekly_heuristic'#'do_everything' #'weekly_heuristic' or 'do_everything'. do_everything is contained per probemap.
        output_folder = Path('D:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output')
        sessionsToDo = 'all'
    if True: # contains si arguments. Likely to be nonspecific.
        desired_n_jobs = 16
        bin_s_sessionCat = 60.0
        si.set_global_job_kwargs(n_jobs=desired_n_jobs)
        doRemoveBadChannels = 1  # currently uses the manual list...
        skipStuffThatKSGUIDoes = 0  # KS GUI does CAR and bandpass filter and it is a bit opaque how to turn off the latter.

    if True: # arguments handling how much code to run
        doPreprocessing = 1 # if you want to load your drift maps without recalculating them, turn this off.
        savePreprocessing = 0
        overwritePreprocessing = 0
        checkMotionPlotsOnline = 1 # turn this off if you don't want to view the plots.
        calculateSessionMotionDisplacement = 1 # will probably never be turned off, since this is the whole point of the code.
        if not doPreprocessing:
            print('warning: not doing any preprocessing.')


    if True: # contains probemap specific stuff. Highly specific to my project.
        if recordingZone == 'ACx_Challah':
            stream_id = 'imec0.ap'
            sessionSetLabel = 'All_ACx_Top'
            channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro'
            #channel_map_to_use_other_ref = 'Challah_top_b1_horizontal_band_joint_tip.imro' ### this wouldn't actually work...
            badChannelList = [66,105,149,170,175,209,210,239,354,369]
            ferret = 'F2302_Challah'
        elif recordingZone == 'PFC_shank0_Challah':
            stream_id = 'imec1.ap'
            sessionSetLabel = 'PFC_shank0'
            channel_map_to_use = 'Challah_top_PFC_shank0.imro'
            badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
            # something else also. Need to read metadata
            ferret = 'F2302_Challah'
        elif recordingZone == 'PFC_shank3_Challah':
            stream_id = 'imec1.ap'
            sessionSetLabel = 'PFC_shank3'
            channel_map_to_use = 'Challah_top_PFC_shank3.imro'
            badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
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
        else:
            sessionString = '[0-9][0-9]*'
    session_path = Path('Z:/Data/Neuropixels/' + ferret)  # path to where all relevant sessions are stored
    SessionsInOrder = sort_np_sessions(list(session_path.glob(sessionString)))
    SetsOfConcatenatedSessions = []

    if frequencyOfConcatenation == 'weekly_heuristic':
        ## aggregate sessions as long as they are less than two days apart. It will fail only to catch if I skep two weekdays. I still need to deal with month though.
        tempPerWeek = []
        countOfConcatenatedSessions = 1  # counting from 1
        week_session_correspondance = []
        week = 0
        for i, session in enumerate(SessionsInOrder):
            session_name = session.name
            if not i:
                priorDay = int(session_name[0:2])
                priorMonth = int(session_name[2:4])
                priorYear = int(session_name[4:8])
                tempPerWeek.append(session)
            else:
                currentDay = int(session_name[0:2])
                if (
                        currentDay - priorDay) < 0:  ## then the month rolled, and things get complicated because of February
                    if (priorMonth == 1) | (priorMonth == 3) | (priorMonth == 5) | (priorMonth == 7) | (
                            priorMonth == 8) | (priorMonth == 10) | (priorMonth == 12):
                        priorDaysInMonth = 31
                    elif (priorMonth == 4) | (priorMonth == 6) | (priorMonth == 9) | (priorMonth == 11):
                        priorDaysInMonth = 30
                    elif (priorMonth == 2) & (not (priorYear % 4)):
                        priorDaysInMonth = 29  # untested but should work
                    else:
                        priorDaysInMonth = 28
                    if ((currentDay + priorDaysInMonth) - priorDay) > 2:
                        SetsOfConcatenatedSessions.append(tempPerWeek)
                        tempPerWeek = []
                        tempPerWeek.append(session)
                        countOfConcatenatedSessions += 1
                    else:
                        tempPerWeek.append(session)
                elif (currentDay - priorDay) > 2:  ## in this case, week is over
                    SetsOfConcatenatedSessions.append(tempPerWeek)
                    tempPerWeek = []
                    tempPerWeek.append(session)
                    countOfConcatenatedSessions += 1
                else:
                    tempPerWeek.append(session)
                priorDay = currentDay
                priorMonth = int(session_name[2:4])
                priorYear = int(session_name[4:8])
        SetsOfConcatenatedSessions.append(tempPerWeek)  # this means that the final week will be appended, I think.
    elif frequencyOfConcatenation == 'do_everything':
        SetsOfConcatenatedSessions =[SessionsInOrder]
        sessionSetName = 'everythingAllAtOnce'

    record_motion_sessions = []
    record_motion_weeks = []
    record_space_bins_sessions = []
    record_space_bins_weeks = []#recording drift/motion for UnitMatch later

    week_session_correspondance = []
    week = 0
    last_session_previous_week = 0
    setsOfSessionsPerGrouping = []
    for sessionSetCount,currentSetOfSessions in enumerate(SetsOfConcatenatedSessions): # first, determine the sessions which require further analysis.

        ### Make a file that keeps track of the recording info
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
        sessionLoopBreakFlag = False
        sessionsWithinMap = []
        for i,session in enumerate(currentSetOfSessions):
            session_name = session.name
            if (frequencyOfConcatenation == 'weekly_heuristic') & (not i):
                sessionSetName = 'weekOf' + session_name[4:8] + session_name[2:4] + session_name[0:2]  # name after first day of week. Also, swap to year month day so that things are alphabetical
            elif (not (frequencyOfConcatenation == 'weekly_heuristic')) & (not i):
                sessionSetName = session_name
            print(f'Processing {sessionSetName}')
            dp = session_path / session_name
            chan_dict = get_channelmap_names(dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
            if (session_name + "_" + stream_id[:-3]) in chan_dict:
                if any(v == channel_map_to_use for v in chan_dict.values()):
                    sessionsWithinMap.append(session)
            else:
                print('a bug you should solve')
                pass
        if any(sessionsWithinMap):
            setsOfSessionsPerGrouping.append(sessionsWithinMap)
    for sessionSetCount,currentSetOfSessions in enumerate(setsOfSessionsPerGrouping): #with weekly heuristic, each current set of session is a list of path objects
        if not sessionsToDo == 'all':
            if not (sessionSetCount in sessionsToDo):
                continue
        ### Make a file that keeps track of the recording info
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
        sessionLoopBreakFlag = False
        for i,session in enumerate(currentSetOfSessions):
            session_name = session.name
            if (frequencyOfConcatenation == 'weekly_heuristic') & (not i):
                sessionSetName = 'weekOf' + session_name[4:8] + session_name[2:4] + session_name[
                                                                                    0:2]  # name after first day of week. Also, swap to year month day so that things are alphabetical
            elif (not (frequencyOfConcatenation == 'weekly_heuristic')) & (not i):
                sessionSetName = session_name
            print(f'Processing {sessionSetName}')
            working_dir = output_folder / 'tempDir' / ferret / session_name
            dp = session_path / session_name
            chan_dict = get_channelmap_names(dp)  # almost works but something about the format is different. no "imRoFile" perameter. There is something called an "imRoTable" which is probably also what I want. But let's deal with this later, when we know we need it. Because, honestly, we want something more sophisticated than this eventually.
            chan_map_name = chan_dict[session_name + "_" + stream_id[:-3]]
            if (not (chan_map_name == channel_map_to_use)):
                print('But not really, because of the probe map')
                print('this currently handles weeks where I switch maps very badly!')
                sessionLoopBreakFlag = True
                break
            probeFolder = list(dp.glob('*' + stream_id[:-3]))
            probeFolder = probeFolder[0]
            if doPreprocessing:
                # recording = si.read_cbin_ibl(probeFolder)  # for compressed
                local_probeFolder = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs') / Path(*probeFolder.parts[1:])
                recording = si.read_spikeglx(probeFolder, stream_id=stream_id) # for uncompressed
                recording = spikeglx_preprocessing(recording, doRemoveBadChannels=doRemoveBadChannels,skipStuffThatKSGUIDoes=skipStuffThatKSGUIDoes,local_probeFolder=local_probeFolder,badChannelList=badChannelList, addNoiseForMotionCorrection=1,bin_s_sessionCat=bin_s_sessionCat)
                ### do things related to the construction of a file which stores the recording information.
                multirec_info['name'].append(session_name)
                multirec_info['fs'].append(recording.get_sampling_frequency())
                multirec_info['n_samples'].append(recording.get_num_samples())
                multirec_info['duration'].append(recording.get_total_duration())

                meta = load_meta_file(probeFolder / (session_name + '_t0.' + stream_id + '.meta'))
                multirec_info['start_time'].append(meta['fileCreateTime'])

                if i == 0:
                    multirec_info['multirec_start_sample'].append(0)
                else:
                    # multirec_info['multirec_start_sample'].append(int(
                    #     multirec_info['multirec_start_sample'][i-1] + (multirec_info['duration'][i-1].total_seconds() * multirec_info['fs'][i-1])+1))

                    multirec_info['multirec_start_sample'].append(
                        multirec_info['multirec_start_sample'][i - 1] + (multirec_info['n_samples'][i - 1]) + 1)

                # multirec_info['multirec_stop_sample'].append(int(multirec_info['multirec_start_sample'][i] + (multirec_info['duration'][i].total_seconds() * multirec_info['fs'][i])))
                multirec_info['multirec_stop_sample'].append(
                    multirec_info['multirec_start_sample'][i] + (multirec_info['n_samples'][i]))
                multirec_info['fullpath_as_string'].append(list(probeFolder.glob('*.bin'))[0].__str__())

                if chan_map_name == channel_map_to_use:  # I did this for basically no reason. The reason is because it fits the expected format of Jules and because someday I may want to organize things by the probe map.
                    if chan_map_name in dict_of_recordings:
                        dict_of_recordings[chan_map_name].append(recording)
                    else:
                        dict_of_recordings[chan_map_name] = [recording]

                else:
                    print('this line should never be reached')
            week_session_correspondance.append([week, i+last_session_previous_week])
            if i==len(currentSetOfSessions)-1:
                last_session_previous_week+=i+1
        if sessionLoopBreakFlag:
            sessionLoopBreakFlag = False
            continue
        if doPreprocessing:
            if len(currentSetOfSessions) > 1:
                multirecordings = {channel_map: si.concatenate_recordings(dict_of_recordings[channel_map]) for channel_map in dict_of_recordings}
                multirecordings = {channel_map: multirecordings[channel_map].set_probe(dict_of_recordings[channel_map][0].get_probe())  for channel_map in multirecordings}
                multirecordingInput = multirecordings[channel_map_to_use]
            else:
                multirecordings = dict_of_recordings
                multirecordingInput = multirecordings[channel_map_to_use][0]
        # recording = si.concatenate_recordings(dict_of_recordings)

        output_folder_temp = output_folder / 'tempDir' / ferret / recordingZone / sessionSetLabel / sessionSetName
        output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone / sessionSetLabel / sessionSetName
        phy_folder = output_folder_sorted / 'KiloSortSortingExtractor' / 'phy_folder'  # probably.
        ### save multirec_info
        phy_folder.mkdir(parents=True, exist_ok=True)
        #multirecordingInput = multirecordingInput.remove_channels(badChannelList)
        if doPreprocessing:
            ### recent stuff to hide with ifs later
            ### motion correction

            job_kwargs = dict(chunk_duration='1s',n_jobs=desired_n_jobs,progress_bar=True) #,
            ### KS defaults, which I am using
            peaks = detect_peaks(recording=multirecordingInput, method="locally_exclusive", peak_sign="neg",
                                 detect_threshold=8.0, exclude_sweep_ms=0.1, radius_um=50,
                                 **job_kwargs)  # seems like there isn't a unique peak detection algorithm for ks? I don't actually think that's true, I believe they use templates... But with SI, maybe not. I should maybe just look into the high level function and see what is done
            peak_locations = localize_peaks(recording=multirecordingInput, peaks=peaks, method="grid_convolution",
                                            weight_method={"mode": "gaussian_2d",
                                                           "sigma_list_um": np.linspace(5, 25, 5)}, **job_kwargs)
            if False: # contains defaults for other stuff
                #### commented out: DREDGE defaults
                peaks = detect_peaks(recording=multirecordingInput, method="locally_exclusive",peak_sign="neg",detect_threshold=8.0,exclude_sweep_ms=0.8,radius_um=80.0, **job_kwargs) #
                peak_locations = localize_peaks(recording=multirecordingInput, peaks=peaks, method="monopolar_triangulation", **job_kwargs)
                ### DREDGE MOTION
                motion = estimate_motion(recording=pickleJar["multirecordingInput"], peaks=peaks,
                                         peak_locations=peak_locations, method="dredge_ap", direction="y",
                                         rigid=False, win_step_um=400.0, win_scale_um=400.0, win_margin_um=None, bin_s=30,
                                         time_horizon_s=100) ### this is all mostly default DREDGE, but the time horizon is new. On my test case it works passably on my first two sessions, then completely fails to capture the third day for unknown reasons.
                rec_corrected = interpolate_motion(recording=whitened_recording,motion=motion,border_mode="remove_channels",spatial_interpolation_method="kriging",sigma_um=30.)

                ### KS MOTION
                ## default (works decently)
                motion = estimate_motion(recording=pickleJar["multirecordingInput"], peaks=peaks,
                                          peak_locations=peak_locations, method="iterative_template", bin_s=2.0,
                                          rigid=False, win_step_um=200.0, win_scale_um=400.0,hist_margin_um=0 ,
                                          win_shape="rect")
            motion = estimate_motion(recording=multirecordingInput, peaks=peaks,
                                     peak_locations=peak_locations, method="iterative_template", bin_s=bin_s_sessionCat,
                                     rigid=False, win_step_um=50.0, win_scale_um=100.0, hist_margin_um=0,
                                     win_shape="gaussian",
                                     num_amp_bins=5)  # this works on my test case. ks defaults, but with win_scale_um=100.0,  win_step_um=50.0, win_shape="gaussian", num_amp_bins=5, bin_s=6.0

            # whitened_recording = si.whiten(recording=multirecordingInput,dtype=float,int_scale=200)

            # rec_corrected = interpolate_motion(recording=whitened_recording, motion=motion, border_mode="force_extrapolate",
            #                                    spatial_interpolation_method="kriging", sigma_um=20.,p=2) # ks defaults. force extrapolate maybe weird.
            rec_corrected_nonWhitened = interpolate_motion(recording=multirecordingInput, motion=motion,
                                                           border_mode="force_extrapolate",
                                                           spatial_interpolation_method="kriging", sigma_um=20., p=2)
            rec_corrected = rec_corrected_nonWhitened.astype('int16')
            rec_corrected = si.whiten(recording=rec_corrected, int_scale=200)
            peak_locations_after = sm.correct_motion_on_peaks(peaks, peak_locations, motion, rec_corrected)
            multirecordingInput = multirecordingInput.astype('int16')
            whitened_recording = si.whiten(recording=multirecordingInput, int_scale=200)

            if False: # plotting functions. Will probably extract them otherwise.
            ### see below the length I need to go to to plot the peak locations as a function of space and time...
                figure_folder = output_folder_sorted / Path("figures_Jeffstyle")
                figure_folder.mkdir(parents=True, exist_ok=True)
                scatterArray = np.zeros((len(peak_locations), 3))
                for iiii in range(0, len(peak_locations)):
                    scatterArray[iiii, 0] = peaks[iiii][0]
                    scatterArray[iiii, 1] = peak_locations[iiii][1]
                    scatterArray[iiii, 2] = peak_locations_after[iiii][1]
                plt.figure()
                plt.scatter(scatterArray[:,0]/30000, scatterArray[:,1],s=0.01,c="black")
                plt.gca().invert_yaxis()
                plt.savefig(figure_folder / Path("MotionScatterBeforeCorrection.pdf"))
                plt.figure()
                plt.scatter(scatterArray[:, 0] / 30000, scatterArray[:, 2], s=0.01, c="black")
                plt.gca().invert_yaxis()
                plt.savefig(figure_folder / Path("MotionScatterAfterCorrection.pdf"))

            if savePreprocessing:
                picklePath = output_folder_sorted
                pickleName = 'preprocess_results.pkl'
                if (not (picklePath / pickleName).is_file())|overwritePreprocessing:
                    pickleJar = dict(peaks=peaks, peak_locations=peak_locations, multirecordingInput=multirecordingInput,rec_corrected=rec_corrected,whitened_recording=whitened_recording,rec_corrected_nonWhitened=rec_corrected_nonWhitened,motion=motion,peak_locations_after=peak_locations_after)
                    with open(picklePath / pickleName, 'wb') as file:
                        pickle.dump(pickleJar, file)

        # multirecordingInput = multirecordingInput.save(folder="C:\Jeffrey\Projects\SpeechAndNoise\Spikesorting_Inputs\Preprocessed", n_jobs=10, chunk_duration='1s') # this function removes the extra channel. So 384 instead of 385.
        if doPreprocessing:
            df_rec = pd.DataFrame(multirec_info)
            df_rec.to_csv(phy_folder / 'multirec_info.csv',
                          index=False)  # this is the earliest phy folder around and may be a problem...
        else:
            # then need to load the recording
            # I originally saved this after entering spikesorting_pipeline so may need to change names later.
            picklePath = output_folder_sorted
            pickleName = 'preprocess_results.pkl'
            with open(picklePath / pickleName, 'rb') as file:
                pickleJar = pickle.load(file)
            peaks = pickleJar["peaks"]
            peak_locations = pickleJar["peak_locations"]
            multirecordingInput = pickleJar["multirecordingInput"]
            rec_corrected = pickleJar["rec_corrected"]
            if not("motion" in pickleJar):
                motion = estimate_motion(recording=multirecordingInput, peaks=peaks,
                                         peak_locations=peak_locations, method="iterative_template", bin_s=bin_s_sessionCat,
                                         rigid=False, win_step_um=50.0, win_scale_um=100.0, hist_margin_um=0,
                                         win_shape="gaussian", num_amp_bins=5)
                peak_locations_after = sm.correct_motion_on_peaks(peaks, peak_locations, motion, multirecordingInput)
                whitened_recording = pickleJar["whitened_recording"] # load also the other stuff before replacing
                rec_corrected_nonWhitened = pickleJar["rec_corrected_nonWhitened"]

                pickleJar = dict(peaks=peaks, peak_locations=peak_locations, multirecordingInput=multirecordingInput,
                                 rec_corrected=rec_corrected, whitened_recording=whitened_recording,
                                 rec_corrected_nonWhitened=rec_corrected_nonWhitened, motion=motion,
                                 peak_locations_after=peak_locations_after)
                with open(picklePath / pickleName, 'wb') as file:
                    pickle.dump(pickleJar, file)
            else:
                motion = pickleJar["motion"]
                peak_locations_after = pickleJar["peak_locations_after"]
        if checkMotionPlotsOnline:
            matplotlib.use('TkAgg')
            scatterArray = np.zeros((len(peak_locations), 3))
            for iiii in range(0, len(peak_locations)):
                scatterArray[iiii, 0] = peaks[iiii][0]
                scatterArray[iiii, 1] = peak_locations[iiii][1]
                scatterArray[iiii, 2] = peak_locations_after[iiii][1]
            plt.figure()
            plt.scatter([1, 2, 3], [2, 4, 3])  # for some reason this helps the other figures load...
            plt.show()
            plt.figure()
            plt.scatter(scatterArray[:, 0] / 30000, scatterArray[:, 1], s=0.01, c="black")
            plt.gca().invert_yaxis()
            plt.show()
            plt.figure()
            plt.scatter(scatterArray[:, 0] / 30000, scatterArray[:, 2], s=0.01, c="black")
            plt.gca().invert_yaxis()
            plt.show()
            plt.figure()
            plt.scatter([1,2,3],[2,4,3]) # for some reason this helps the other figures load...
            plt.show()
            breakPointSpot = "here"
        if calculateSessionMotionDisplacement:
            sessionEnds = np.asarray(multirecordingInput._recording_segments[0].all_length)/30000
            sessionEnds = np.cumsum(sessionEnds)
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
            if not sessionSetCount:
                record_motion_weeks.append(motion_average)
                record_space_bins_weeks.append(motion.spatial_bins_um)
                for i in bins_per_session:
                    record_motion_sessions.append(i)
                    record_space_bins_sessions.append(motion.spatial_bins_um)
            else:
                record_motion_weeks.append(motion_average)
                record_space_bins_weeks.append(motion.spatial_bins_um + record_motion_weeks[sessionSetCount-1])
                for i in bins_per_session:
                    record_motion_sessions.append(i)
                    record_space_bins_sessions.append(motion.spatial_bins_um + record_motion_weeks[sessionSetCount-1])
                #record_motion.append(record_motion[sessionSetCount-1]+motion_average)
            #record_space_bins.append(motion.spatial_bins_um)
            breakPointSpot = "here"


            if False: # within here, I will stre the means to make the version of the plot I used for ICAC 2025, which is a mostly specialized plot for PFC first week Challah
                figure_folder = output_folder_sorted / Path("figures_Jeffstyle")
                figure_folder.mkdir(parents=True, exist_ok=True)
                yrangeForPlots = np.array([3330,4000])
                xRangeForPlots = np.array([237,2575])
                sessionEndsFirstFour = np.asarray(multirecordingInput._recording_segments[0].all_length)[0:4]/30000
                sessionEndsFirstFour = np.cumsum(sessionEndsFirstFour)
                if True: # first, before the correction
                    figureAddress = figure_folder / Path('/BeforeCorrection.pdf')
                    Before_Correction_pdf = PdfPages(figureAddress)

                    BeforeCorrectionPlot, BeforeCorrectionAxis = plt.subplots()


                    BeforeCorrectionAxis.scatter((scatterArray[:, 0] / 30000)-xRangeForPlots[0], scatterArray[:, 1], s=0.01, c="black")
                    for sessionEnds in sessionEndsFirstFour:
                        BeforeCorrectionAxis.axvline(x=sessionEnds-xRangeForPlots[0], color='r', linestyle='--')


                    BeforeCorrectionAxis.set_xlim([xRangeForPlots[0]-xRangeForPlots[0], xRangeForPlots[1]-xRangeForPlots[0]])
                    BeforeCorrectionAxis.set_ylim([yrangeForPlots[0], yrangeForPlots[1]])
                    BeforeCorrectionAxis.invert_yaxis()

                    Before_Correction_pdf.savefig(BeforeCorrectionPlot)
                    Before_Correction_pdf.close()
                    #BeforeCorrectionPlot.show()

                if True: # then, after the correction
                    figureAddress = figure_folder / Path('/AfterCorrection.pdf')
                    After_Correction_pdf = PdfPages(figureAddress)

                    AfterCorrectionPlot, AfterCorrectionAxis = plt.subplots()


                    AfterCorrectionAxis.scatter((scatterArray[:, 0] / 30000)-xRangeForPlots[0], scatterArray[:, 2], s=0.01, c="black")
                    for sessionEnds in sessionEndsFirstFour:
                        AfterCorrectionAxis.axvline(x=sessionEnds-xRangeForPlots[0], color='r', linestyle='--')


                    AfterCorrectionAxis.set_xlim([xRangeForPlots[0]-xRangeForPlots[0], xRangeForPlots[1]-xRangeForPlots[0]])
                    AfterCorrectionAxis.set_ylim([yrangeForPlots[0], yrangeForPlots[1]])
                    AfterCorrectionAxis.invert_yaxis()
                    After_Correction_pdf.savefig(AfterCorrectionPlot)
                    After_Correction_pdf.close()
                    #AfterCorrectionPlot.show()
            if False: # Another week, to show off the session-length-dependent cell.
                figure_folder = output_folder_sorted / Path("figures_Jeffstyle")
                figure_folder.mkdir(parents=True, exist_ok=True)
                yrangeForPlots = np.array([4333,4863])
                xRangeForPlots = np.array([0,5000])
                sessionEndsFirstFew = np.asarray(multirecordingInput._parent._recording_segments[0].all_length)[0:7]/30000
                sessionEndsFirstFew = np.cumsum(sessionEndsFirstFew)
                if True: # first, before the correction
                    figureAddress = figure_folder / Path('/Satiation.pdf')
                    Satiation_pdf = PdfPages(figureAddress)

                    SatiationPlot, SatiationAxis = plt.subplots()


                    SatiationAxis.scatter((scatterArray[:, 0] / 30000)-xRangeForPlots[0], scatterArray[:, 1], s=0.01, c="black")
                    for sessionEnds in sessionEndsFirstFew:
                        SatiationAxis.axvline(x=sessionEnds-xRangeForPlots[0], color='r', linestyle='--')


                    SatiationAxis.set_xlim([xRangeForPlots[0]-xRangeForPlots[0], xRangeForPlots[1]-xRangeForPlots[0]])
                    SatiationAxis.set_ylim([yrangeForPlots[0], yrangeForPlots[1]])
                    SatiationAxis.invert_yaxis()

                    Satiation_pdf.savefig(SatiationPlot)
                    Satiation_pdf.close()
                    SatiationPlot.show()
        week+=1

    record_motion_weeks = np.array(record_motion_weeks)
    record_space_bins_weeks = np.array(record_space_bins_weeks)
    N, M = record_motion_weeks.shape
    rows = []
    for i in range(N):
        for j in range(M):
            rows.append((sessionsToDo[i], record_motion_weeks[i, j], record_space_bins_weeks[i, j])) ### WARNING use of sessionsToDo may be wrong here

    df_weeks = pd.DataFrame(rows, columns=["session_week", "motion", "center_space_bin"])

    record_motion_sessions = np.array(record_motion_sessions)
    record_space_bins_sessions = np.array(record_space_bins_sessions)
    N, M = record_motion_sessions.shape
    rows = []
    for i in range(N):
        for j in range(M):
            rows.append((i, record_motion_sessions[i, j], record_space_bins_sessions[i, j]))

    df_sessions = pd.DataFrame(rows, columns=["session", "motion", "center_space_bin"])

    # Save to CSV
    df_weeks.to_csv(phy_folder / "motion_weeks.csv", index=False)
    df_sessions.to_csv(phy_folder / "motion_sessions.csv", index=False)
    week_session_correspondance = np.array(week_session_correspondance)
    np.save(phy_folder / "session_to_week_id", week_session_correspondance)


if __name__ == '__main__':
    main()


