from pathlib import Path
import numpy as np
import spikeinterface.full as si
import matplotlib.pyplot as plt
import spikeinterface.widgets as sw

import neuropixels_chronic_spikesorting_Bizley.externalBugFixes.spike_interface.v_103_0.silence_periods as silence_periods_file
import neuropixels_chronic_spikesorting_Bizley.config as global_configs
import torch
import pickle


def spikeglx_visualize(recording):
    #At the moment, a function designed to be break-pointed into.
    #Basically I just want a venue to look at data in ways I want.

    ### First: plot traces. Without any processing, they will be raw; you can do CMR or bandpass or phase shift or bad channel removal to improve things.
            # by the way, if you ever want to load traces into RAM, it's "get_traces". A method on the recording object.

    badChannelList = [66, 105, 149, 170, 175, 209, 210, 239, 354, 369]

    recording = si.phase_shift(recording)
    new_channel_ids = recording.channel_ids[~np.in1d(recording.channel_ids, recording.channel_ids[badChannelList])]
    recording = si.channelslice.ChannelSliceRecording(recording, new_channel_ids)
    satRange = [191, 192] # range which includes a small period of saturation
    satRangeSamples = [int(satRange[0] * 30000), int(satRange[1] * 30000)]
    spikeViewRegion = [317.14,317.15]#[191.4,191.41]
    spikeViewRegionSamples = [int(spikeViewRegion[0] * 30000), int(spikeViewRegion[1] * 30000)]
    testData = recording.get_traces(0,spikeViewRegionSamples[0],spikeViewRegionSamples[1])
    plt.imshow(testData.T, aspect='auto', clim=[-2000, 2000])
    #recording = si.blank_staturation(recording, abs_threshold=1900, direction="both")
    #w_ts = sw.plot_timeseries(recording, time_range=satRange,clim=(-1955,1955))
    plt.show()
    recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000)

    recording = si.common_reference(recording, reference='global', operator='median')

    w_ts2 = sw.plot_timeseries(recording, time_range=spikeViewRegion, clim=(-20, 20))
    plt.show()
    fakey = 1 + 1
def spikeglx_preprocessing(recording,doRemoveBadChannels =1,skipStuffThatKSGUIDoes = 0,local_probeFolder=None,badChannelList=[], addNoiseForMotionCorrection=1,bin_s_sessionCat=6.0):
    recording = si.phase_shift(recording) #mandatory for NP recordings because the channels are not sampled at the same time.
    windowsToSilenceArray = nullify_saturations(recording, local_probeFolder=local_probeFolder) # find saturations before any further processing. Only finds, doesn't correct.


    if doRemoveBadChannels: # next remove bad channels
        #bad_channel_ids, channel_labels = si.detect_bad_channels(recording) # this seems not as good as I need it to be... Totally misses all my bad channels. Maybe just need to change parameters...
        #recording = recording.remove_channels(bad_channel_ids) # I don't want to remove them here because I am concatenating. Remove later
        new_channel_ids = recording.channel_ids[~np.in1d(recording.channel_ids, recording.channel_ids[badChannelList])]
        recording = si.channelslice.ChannelSliceRecording(recording, new_channel_ids)
    if not skipStuffThatKSGUIDoes: # I actually might move this to after the concatenation?
        recording = si.bandpass_filter(recording, freq_min=300, freq_max=6000) # filter each
        if False: # here I do peak detection and localization early because it is quicker with one session... I do this to test a processing method.
            ### make a subselection of the recording so you can do this faster
            low_bound_temp = int(round(960*30000))
            high_bound_temp = int(round(975*30000))
            recording_preCMR = recording.frame_slice(low_bound_temp,high_bound_temp)
            recording_CMR = si.common_reference(recording_preCMR, reference='global', operator='median')
            si.highpass_spatial_filter(recording)
            #recording_localRef = si.common_reference(recording_preCMR, reference='local', operator='median',local_radius=(60,200)) ### I don't know if local radius is in units of channels or microns.
            recording_localRef = si.highpass_spatial_filter(recording_preCMR)
            desired_n_jobs = 16
            from spikeinterface.sortingcomponents.peak_detection import detect_peaks
            from spikeinterface.sortingcomponents.peak_selection import select_peaks
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks
            from spikeinterface.sortingcomponents.motion import estimate_motion, interpolate_motion

            job_kwargs = dict(chunk_duration='1s', n_jobs=desired_n_jobs, progress_bar=True)  # ,
            ### KS defaults, which I am using
            peaks_CMR = detect_peaks(recording=recording_CMR, method="locally_exclusive", peak_sign="neg",
                                 detect_threshold=8.0, exclude_sweep_ms=0.1, radius_um=50,
                                 **job_kwargs)  # seems like there isn't a unique peak detection algorithm for ks? I don't actually think that's true, I believe they use templates... But with SI, maybe not. I should maybe just look into the high level function and see what is done
            peak_locations_CMR = localize_peaks(recording=recording_CMR, peaks=peaks_CMR, method="grid_convolution",
                                            weight_method={"mode": "gaussian_2d",
                                                           "sigma_list_um": np.linspace(5, 25, 5)}, **job_kwargs)
            peaks_localRef = detect_peaks(recording=recording_localRef, method="locally_exclusive", peak_sign="neg",
                                 detect_threshold=8.0, exclude_sweep_ms=0.1, radius_um=50,
                                 **job_kwargs)  # seems like there isn't a unique peak detection algorithm for ks? I don't actually think that's true, I believe they use templates... But with SI, maybe not. I should maybe just look into the high level function and see what is done
            peak_locations_localRef = localize_peaks(recording=recording_localRef, peaks=peaks_localRef, method="grid_convolution",
                                            weight_method={"mode": "gaussian_2d",
                                                           "sigma_list_um": np.linspace(5, 25, 5)}, **job_kwargs)
            if False: # here I do plotting

                import matplotlib
                import matplotlib.pyplot as plt
                matplotlib.use('TkAgg')
                scatterArray_CMR = np.zeros((len(peak_locations_CMR), 2))
                for iiii in range(0, len(peak_locations_CMR)):
                    scatterArray_CMR[iiii, 0] = peaks_CMR[iiii][0]
                    scatterArray_CMR[iiii, 1] = peak_locations_CMR[iiii][1]
                scatterArray_localRef = np.zeros((len(peak_locations_localRef), 2))
                for iiii in range(0, len(peak_locations_localRef)):
                    scatterArray_localRef[iiii, 0] = peaks_localRef[iiii][0]
                    scatterArray_localRef[iiii, 1] = peak_locations_localRef[iiii][1]
                plt.figure()
                plt.scatter([1, 2, 3], [2, 4, 3])  # for some reason this helps the other figures load...
                plt.show()
                plt.figure()
                plt.scatter(scatterArray_CMR[:, 0] / 30000, scatterArray_CMR[:, 1], s=0.01, c="black")
                plt.gca().invert_yaxis()
                plt.show()
                plt.figure()
                plt.scatter(scatterArray_localRef[:, 0] / 30000, scatterArray_localRef[:, 1], s=0.01, c="black")
                plt.gca().invert_yaxis()
                plt.show()
                breakPointSpot = "here"
        recording = si.common_reference(recording, reference='global',operator='median')  # common reference, I think it's much better to do this after removing bad channels. IBL does a spatial highpass with very low cutoff, butI will keep median for now.
    recording = recording.astype(dtype="float32")
    if windowsToSilenceArray.any():  # code breaks when no saturation
        if global_configs.useBugFixedSilencePeriods:
            recording = silence_periods_file.silence_periods(recording, windowsToSilenceArray, mode="noise")
        else:
            recording = si.silence_periods(recording, windowsToSilenceArray, mode="noise") # Doing this before whitening might cause issues... I need to understand how whitening works better.
        # recording = recording.astype('int16') ### reportedly converting back in this way is dangerous, and I should look into it. I need to do it for harddrive space, but otherwise I shouldn't bother.
    if addNoiseForMotionCorrection:
        pad_size, pad_bounds = size_pad_to_add(recording,bin_s_sessionCat)
        padRecording = recording.frame_slice(0,pad_size)
        recording = si.concatenate_recordings([recording, padRecording])
        if global_configs.useBugFixedSilencePeriods:
            recording = silence_periods_file.silence_periods(recording, [[pad_bounds[0],recording.get_num_frames()]], mode="noise")
        else:
            recording = si.silence_periods(recording, [[pad_bounds[0],recording.get_num_frames()]], mode="noise")
    # spikeglx_visualize(recording)
    return recording

def nullify_saturations(recording,surrondingToAlsoNullify=100,local_probeFolder=None):
    # The idea behind this function is to:
    #     - Detect saturated periods like si.blank_saturation
    #           (in fact, this program does this channel by channel I think. I will actually want to apply my threshold on the
    #           average of all channels, which will greatly help with avoiding false positives)
    #     -  mark these times for all later analyses to ignore
    #     -  save and load those times so that silence_periods can later be used to target them
    #
    # The idea is that it because it is impossible to recover spikes in saturated periods, we simply want to
    # prevent them from annoying us in our processing steps. I may write additional "nullify" functions
    # in the future should other excludable periods turn up.

    # it's important to run this functionon the raw data, but I prefer to apply the silence_periods function at the end'
    #    'of the processing chain.

    # additionally: this function is useful for nullifying *any* sort of problem. I therefore also include functionality
    # to notice particular sessions and silence particular ranges for them.

    # Arguments:
    #
    #     -recording (the si recording)
    #     -surroundingToAlsoNullify (in ms. So input 10 here and 10 ms on each end of each null period will also be null.
    #                The intention here is to deal with the up and down ramps involved in saturation conservatively.
    loadSatsFromFile = 1
    alsoCutOffBeginningAndEnd = 1

    local_probeFolder.mkdir(parents=True, exist_ok=True)
    fileNameSat = local_probeFolder / Path("saturatedZones.csv")
    if (fileNameSat.is_file())&loadSatsFromFile:
        windowsToSilenceArray = np.loadtxt(fileNameSat)
        if windowsToSilenceArray.size == 0:
            windowsToSilenceArray = np.zeros((0, 2))
        if (not (windowsToSilenceArray.ndim == 2))&(windowsToSilenceArray.shape[0]==2):
            windowsToSilenceArray = windowsToSilenceArray.reshape(1, 2)
    else:
        nullSat_init_args = (recording,surrondingToAlsoNullify)# needs to be a tuple.
        n_jobs = 10 # 1 for testing, -1 to use all cores
        nullSat = si.ChunkRecordingExecutor(recording,func=nullify_saturations_loopImplementation,init_func = nullify_saturations_init_func,init_args=nullSat_init_args,n_jobs=n_jobs,chunk_size = 60000,handle_returns=True,progress_bar=True) # key function to let me do this here. It will parallelize everything for me.
        windowsToSilence = nullSat.run()
        windowsToSilenceArray = np.zeros((0,2))
        for chunkSet in windowsToSilence: # yes it's silly to do this afterward instead of during, but it's really not a big deal. Default function outputs list.
            windowsToSilenceArray = np.concatenate((windowsToSilenceArray,chunkSet))
        np.savetxt(fileNameSat, windowsToSilenceArray)
    if alsoCutOffBeginningAndEnd:
        secondsToCut = 2
        windowsToSilenceArray = np.concatenate(([[0,int(recording.sampling_frequency*secondsToCut)]],windowsToSilenceArray))
        windowsToSilenceArray = np.concatenate((windowsToSilenceArray,  [[(recording.get_num_frames()-int(recording.sampling_frequency*secondsToCut)),recording.get_num_frames()]]))
    if True: # section for dealing with specific sessions.
        topFolderString = local_probeFolder.parts[-1]
        if topFolderString[0:19] == "23052024_AM_Challah":
            # ferret held in hand 45 seconds from end of recording. Should affect both probes
            if alsoCutOffBeginningAndEnd:
               windowsToSilenceArray = np.concatenate((windowsToSilenceArray[0:-1,:], [[(recording.get_num_frames() - int(recording.sampling_frequency * 45)),recording.get_num_frames()]]))

            else:
                windowsToSilenceArray = np.concatenate((windowsToSilenceArray, [[(recording.get_num_frames() - int(recording.sampling_frequency * 45)),recording.get_num_frames()]]))
        if topFolderString[0:19] == "05062024_PM_Challah":
            # ferret held in hand 60 seconds in the middle of the recording... should affect both probes. Might affect behavior. I'll assume at first there is no overlap with detected saturations because why would there be...
                beginTime = 296.4
                endTime = 358.2
                windowsToSilenceArray = np.concatenate((windowsToSilenceArray, [
                    [int(30000*beginTime),
                     int(30000*endTime)],]))

    return windowsToSilenceArray

def nullify_saturations_init_func(recording,surrondingToAlsoNullify):
    # It seems like the only place you can put any information about the recording or w/e is here.
    worker_ctx = {}
    worker_ctx["recording"] = recording
    worker_ctx["surrondingToAlsoNullify_samples"] = round(((recording.sampling_frequency))*(surrondingToAlsoNullify/1000))
    worker_ctx["kernelForSurroundingConvolution"] = np.ones((worker_ctx["surrondingToAlsoNullify_samples"]*2 + 1))
    worker_ctx["abs_sat_min"] = -1880 # in my experience it floats around 1900. Also actual max is supposed to be 1955.
    worker_ctx["abs_sat_max"] = 1880 # splitting them because it really depends. I don't think I actually have problems with max sat currently...
    worker_ctx["fractionChannelsSatToEnforceExclusion"] = 0.5 # in some ways feels low, but it is a lot. And doing plots, this seems like it serves.
    return worker_ctx
def nullify_saturations_loopImplementation(segment_index, frame_start, frame_stop, worker_ctx):
    # function to pass to ChunkRecordingExecutor.
    working_chunk = worker_ctx["recording"].get_traces(segment_index, frame_start, frame_stop)
    overMax = working_chunk > worker_ctx["abs_sat_max"]
    underMin = working_chunk < worker_ctx["abs_sat_min"]
    fractionChannelsOverMaxPerTime = np.sum(overMax.astype(int),axis=1)/working_chunk.shape[1]
    fractionChannelsUnderMinPerTime = np.sum(underMin.astype(int),axis=1)/working_chunk.shape[1]
    if any(fractionChannelsUnderMinPerTime > worker_ctx["fractionChannelsSatToEnforceExclusion"]) | any(fractionChannelsOverMaxPerTime > worker_ctx["fractionChannelsSatToEnforceExclusion"]):

        timeTrace = ((fractionChannelsUnderMinPerTime > worker_ctx["fractionChannelsSatToEnforceExclusion"]) | (fractionChannelsOverMaxPerTime > worker_ctx["fractionChannelsSatToEnforceExclusion"])).astype(int)
        extendedTimeTrace = np.convolve(timeTrace, worker_ctx["kernelForSurroundingConvolution"],mode="same")
        extendedTimeTrace = (extendedTimeTrace.astype(bool)).astype(int)
        difftrace = np.diff(extendedTimeTrace)
        ups = np.where(difftrace>0)[0] + frame_start
        downs = np.where(difftrace<0)[0] + frame_start

        # some visualization code for debugging. Comment out most of the time.

        # satRange = [frame_start / worker_ctx["recording"].sampling_frequency,
        #             frame_stop / worker_ctx["recording"].sampling_frequency]
        # plt.plot(extendedTimeTrace)
        # w_ts = sw.plot_timeseries(worker_ctx["recording"], time_range=satRange, clim=(-1955, 1955))
        # plt.show()
        #
        # hhhhh=1+1

        if extendedTimeTrace[0] & extendedTimeTrace[-1]:
            if (any(ups)|any(downs)):
                # then both the front and the back are saturated.
                rangesToSilence = np.zeros((len(ups) + 1, 2))
                rangesToSilence[0] = [frame_start, downs[0]]
                for i in range(0,len(ups)):
                    if not i== len(ups)-1:
                        rangesToSilence[i+1] = [ups[i],downs[i+1]]
                    else:
                        rangesToSilence[i+1] = [ups[i], frame_stop]
            else:
                rangesToSilence = np.zeros((1, 2))
                rangesToSilence[0] = [frame_start, frame_stop]
        elif extendedTimeTrace[0]:
            #then we need to connect a range later.
            rangesToSilence = np.zeros((len(ups) + 1, 2))
            rangesToSilence[0] = [frame_start, downs[0]]
            for i in range(0,len(ups)):
                rangesToSilence[i+1] = [ups[i],downs[i+1]]
        elif extendedTimeTrace[-1]:
            #then we need to connect a range later.
            rangesToSilence = np.zeros((len(ups),2))
            for i in range(0,len(ups)):
                if not i == len(ups)-1:
                    rangesToSilence[i] = [ups[i],downs[i]]
                else:
                    rangesToSilence[i] = [ups[i], frame_stop]
        elif not len(ups) == len(downs):
            raise "unequal number of raises and lowers, but not because of one of them starting at the edge"
        elif len(ups) | len(downs):
            rangesToSilence = np.zeros((len(ups), 2))
            for i in range(0,len(ups)):
                rangesToSilence[i] = [ups[i],downs[i]]
            bluhah = 1
        else:
            raise "hit something weird"
    else:
        rangesToSilence = np.zeros((0, 2))
    for ranges in rangesToSilence:
        if ranges[0] > ranges[1]:
            raise "see what is up"
    return rangesToSilence

def size_pad_to_add(recording, bin_s=6.0):
    """ Function computing the size of the pad to add to a recording in to make it an integer multiple of bin_s
    Parameters:
        recording: recording object (cf spikeinterface doc)
        bin_s: int (in seconds)
    Returns:
        pad_size: int, size of the pad to add to the recording (in number of samples)
        pad_bounds: list, future boundaries of the pad once it's added to be able to replace it with noise
        """
    sampling_freq =recording.get_sampling_frequency()
    bin_samples = int(np.round(bin_s* sampling_freq))
    size_recording = recording.get_num_samples() -1
    pad_size = bin_samples - (size_recording % bin_samples)
    pad_bounds= [size_recording, size_recording + pad_size]
    return pad_size, pad_bounds

def spikesorting_pipeline(recording, output_folder, sorter='kilosort4',concatenated=False):
    working_directory = Path(output_folder) / 'tempDir'
    # if (working_directory / 'binary.json').exists():
    #     recording = si.load_extractor(working_directory)
    # else:
    # job_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)
    # recording = recording.save(folder = working_directory, format='binary', **job_kwargs)
    # picklePath = Path('C:/Jeffrey/Projects/SpeechAndNoise/PythonAnalyses/pickleFolder/driftPickle/')
    # pickleName = 'preSorting.pkl'
    # pickleJar = dict(recording = recording)
    # with open(picklePath / pickleName, 'wb') as file:
    #     pickle.dump(pickleJar, file)
    doDefaultKilosort = True


    if doDefaultKilosort:
        if sorter=='kilosort4':
            sorting = si.run_sorter(
                sorter_name=sorter,
                recording=recording,
                output_folder = working_directory / f'{sorter}_output',
                verbose=True,
                do_CAR=True, # this could maybe also be skipped
                skip_kilosort_preprocessing=False, # skips bandpass filter and whitening.
                do_correction=True, #skips motion correction
                save_extra_vars=True, #save everything
                save_preprocessed_copy=False, #
                delete_recording_dat=True # now true because this is whitened, which makes it useless. By the way, I believe this is before motion correction, the above is after.
                )
        else:
            raise ValueError('Unsupported Spikesorter.')
    else:
        if sorter=='kilosort4':
            sorting = si.run_sorter(
                sorter_name=sorter,
                recording=recording,
                output_folder = working_directory / f'{sorter}_output',
                verbose=True,
                do_CAR=False, # we want to skip CAR because we already have
                skip_kilosort_preprocessing=True, # skips bandpass filter and whitening.
                do_correction=False, #skips motion correction
                save_extra_vars=True, #save everything
                save_preprocessed_copy=False, #
                delete_recording_dat=True # now true because this is whitened, which makes it useless. By the way, I believe this is before motion correction, the above is after.
                )
        else:
            raise ValueError('Unsupported Spikesorter.')
    return sorting


def spikesorting_postprocessing(sorting, output_folder):

    ### It has been a long time since I looked at this and it is basically irrelevant to the unitmatch pipeline. So, watch out.



    output_folder.mkdir(exist_ok=True, parents=True)
    rec = sorting._recording # grab the recording object
    outDir = output_folder / sorting.name
    censored_period_ms = 0.3 ### Jules had this as 2 but he claimed this was above the default. Using 2 would probably miss inhibitory cells.
    si.set_global_job_kwargs(n_jobs=15)
    jobs_kwargs = dict(chunk_duration='1s', progress_bar=True)
    sorting = si.remove_duplicated_spikes(sorting, censored_period_ms=censored_period_ms)
    sorting = si.remove_excess_spikes(sorting, rec)

    if (outDir / 'sortings_folder').exists():
        # we = si.load_waveforms(
        #     outDir / 'sortings_folder',
        #     sorting=sorting,
        #     with_recording=True,
        # )
        we = si.load_sorting_analyzer(outDir / 'sortings_folder', load_extensions=True, format="auto", backend_options=None)
    else:
        # we = si.create_sorting_analyzer() # need to figure this out later.
        # check https://spikeinterface.readthedocs.io/en/latest/tutorials/waveform_extractor_to_sorting_analyzer.html
        # The reason for the update was to not have to extract waveforms for things that didn't need them.
        #
        we = si.create_sorting_analyzer(recording=rec, sorting=sorting, folder=outDir / 'sortings_folder',
                                        format="binary_folder",
                                        sparse=True
                                        )

        ### I've basically translated the compatibility code below.

        # other_kwargs, new_job_qwargs = split_job_kwargs(**jobs_kwargs)

    if not we.has_extension("random_spikes"):
        we.compute("random_spikes",max_spikes_per_unit=300) ### subselects a number of spikes for downstream analysis.
        print('Marker 3')
    if not we.has_extension("waveforms"):
        we.compute("waveforms",ms_before=2,ms_after=3.) ### has the waveforms corresponding to the subset of spikes selected above, on all relevant channels (with relevancy decided by the "sparsity"). You in fact only need to collect these to calculate the templates, which can also be calculated directly from the random spikes and raw data. But, the waveforms file should be smaller than the raw data I think, and also if you use the raw data directly, apparently you can only calculate mean and standard deviation, and not median or percentile. Anyway, I dont mind calculating waveforms as long as it works.
        print('Marker 4')
    # if not we.has_extension("templates"):
    we.compute("templates")  ### used for downstream analysis and are useful to look at to see if your results make sense.
    print('Marker 5')
    if not we.has_extension("noise_levels"):
        we.compute("noise_levels") ### I worry how useful this would be for my concatenated data...
        print('Marker 6')
    # if not we.has_extension("spike_amplitudes"):
    print('Marker 6.5')
    we.compute("spike_amplitudes")
    print('Marker 7')
    # if not we.has_extension("template_similarity"):
    we.compute("template_similarity")
    print('Marker 8')
    if not (outDir / 'report').exists():
        si.export_to_phy(we, outDir / 'phy_folder',
                         verbose=True,
                         compute_pc_features=False,
                         copy_binary=False,
                         remove_if_exists=True,
                         **jobs_kwargs,
                         )

        si.export_report(we, outDir / 'report',
                         format='png',
                         force_computation=True,
                         **jobs_kwargs,
                         )
        # metrics = si.compute_quality_metrics( #### this seems defunct. Seems like I want something related to "ComputeQualityMetrics" here.
        #     we,
        #     n_jobs=jobs_kwargs['n_jobs'],
        #     verbose=True,
        # )