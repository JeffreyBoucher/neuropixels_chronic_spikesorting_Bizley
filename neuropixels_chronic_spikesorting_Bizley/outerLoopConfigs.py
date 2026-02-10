from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import spikeinterface.full as si


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names


import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.

###### all project-specific outer loop variables should be handled by this file. ##########


###### PLOTTING PARAMETERS

if all_VE_config.plottingArguments == 'JeffreyRecommended':
    matplotlibGUItype = 'TkAgg'

    matplotlib.use(matplotlibGUItype) # my preferred plotting gui at the moment. One of the first i tried.

###### SPIKESORTING PARAMETERS

if all_VE_config.generalSpikesortingArguments == 'JeffreyRecommended':
    desired_n_jobs = 16
    si.set_global_job_kwargs(n_jobs=desired_n_jobs)
    doRemoveBadChannels = 1  # currently uses the manual list...
    skipStuffThatKSGUIDoes = 0  # KS GUI does CAR and bandpass filter and it is a bit opaque how to turn off the latter.

if all_VE_config.sessionwiseDriftCorrectionArguments == 'JeffreyRecommended': # contains si arguments. Likely to be nonspecific.

    bin_s_sessionCat = 6.0 # 6 is best. It is the only way I've managed to get something that looks good. Possibly eventually we will want to do this per-week though.
    silenceOrNoiseReplace_sessionwise = 'zeros' # zeros is faster, 'noise' makes more sense theoretically but practically doesn't seem much different
    if silenceOrNoiseReplace_sessionwise == 'silence': silenceOrNoiseReplace = 'zeros' # foolproofing

if all_VE_config.withinSessionSpikesortingArguments == 'JeffreyRecommended':
    silenceOrNoiseReplace_within = 'zeros' # have not tried zeros in actual spikesorting.
    if silenceOrNoiseReplace_within == 'silence': silenceOrNoiseReplace = 'zeros' # foolproofing

###### SAVING PARAMETERS SPIKESORTING

if all_VE_config.projectLabel == 'Jeffrey': # arguments handling how much code to run
    doPreprocessing = 0 # if you want to load your drift maps without recalculating them, turn this off.
    savePreprocessing = 1
    overwritePreprocessing = 1
    resaveMotionIfLoadingPreprocessing = 1 # if you turn off preprocessing, you can either resave new motion correction stuff or not. Kapt off my default for safety.
    checkMotionPlotsOnline = 0 # turn this off if you don't want to view the plots.
    calculateSessionMotionDisplacement = 1 # will probably never be turned off, since this is the whole point of the code.
    testingThings = 0
    if testingThings:
        print('WARNING WARNING TESTING TESTING')
        print('WARNING WARNING TESTING TESTING')
        print('WARNING WARNING TESTING TESTING')
        print('WARNING WARNING TESTING TESTING')
        print('WARNING WARNING TESTING TESTING')
        print('WARNING WARNING TESTING TESTING')
    if not doPreprocessing:
        print('warning: not doing any preprocessing.')


###### IMPLEMENTATION OF FREQUENCY OF CONCATENATION


###### Reorder sessions and prep loop.

if all_VE_config.projectLabel == 'Jeffrey': # finds my sessions on the NAS, sorts them in order, then initializes the SetsOfConcatenatedSessions. I recommend using sort_np_sessions, it is useful for getting things in order.
    SessionsInOrder = sort_np_sessions(list(all_VE_config.session_path.glob(all_VE_config.sessionString)))
    SetsOfConcatenatedSessions = []


if all_VE_config.frequencyOfConcatenation == 'weekly_heuristic':
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
elif all_VE_config.frequencyOfConcatenation == 'do_everything':
    SetsOfConcatenatedSessions =[SessionsInOrder]
    sessionSetName = 'everythingAllAtOnce'

###### stuff from versions of week-driftmapping that might be defunct now

record_motion_sessions = []
record_motion_weeks = []
record_space_bins_sessions = []
record_space_bins_weeks = []#recording drift/motion for UnitMatch later

week_session_correspondance = []
week = 0
last_session_previous_week = 0
setsOfSessionsPerGrouping = []

if all_VE_config.make_multirecording_info == 'JeffreyRecommended':

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
        sessionLoopBreakFlag = False
        sessionsWithinMap = []
        for i,session in enumerate(currentSetOfSessions):
            session_name = session.name
            if (all_VE_config.frequencyOfConcatenation == 'weekly_heuristic') & (not i):
                sessionSetName = 'weekOf' + session_name[4:8] + session_name[2:4] + session_name[0:2]  # name after first day of week. Also, swap to year month day so that things are alphabetical
            elif (not (all_VE_config.frequencyOfConcatenation == 'weekly_heuristic')) & (not i):
                sessionSetName = session_name
            print(f'Processing {sessionSetName}')
            dp = all_VE_config.session_path / session_name
            chan_dict = get_channelmap_names(dp)
            if (session_name + "_" + all_VE_config.stream_id[:-3]) in chan_dict:
                if any(v == all_VE_config.channel_map_to_use for v in chan_dict.values()):
                    sessionsWithinMap.append(session)
            else:
                print('a bug you should solve')
                pass

        if any(sessionsWithinMap):
            setsOfSessionsPerGrouping.append(sessionsWithinMap)