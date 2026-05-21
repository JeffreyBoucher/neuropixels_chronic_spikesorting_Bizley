from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import spikeinterface.full as si


from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions, get_channelmap_names


import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config # this config file should be synced between all your VEs.

###### all project-specific outer loop variables should be handled by this file. ##########

if all_VE_config.projectLabel == 'Jeffrey':
    if all_VE_config.computerUsed == 'JeffreyLabDesktop':
        catgt_location = Path('C:/Users/jeff/PycharmProjects/neuropixels_chronic_spikesorting_Bizley/CatGT-win') # should be local to the VE, but absolute paths are nice.
        OverStrike_location = Path('C:/Users/jeff/PycharmProjects/neuropixels_chronic_spikesorting_Bizley/OverStrike-win')
###### PLOTTING PARAMETERS

if all_VE_config.plottingArguments == 'JeffreyRecommended':
    # matplotlibGUItype = 'TkAgg' # my preferred plotting gui at the moment. One of the first i tried. It isn't that good though.
    pass
    # matplotlib.use(matplotlibGUItype)

###### SPIKESORTING PARAMETERS

if all_VE_config.generalSpikesortingArguments == 'JeffreyRecommended':
    desired_n_jobs = 16 # this is where you can set up parallel processing. -1 for all available cores. Sometimes some code breaks when you try to parallel process; you can set to 1 to bypass those problems in exchange for things running slow.
    si.set_global_job_kwargs(n_jobs=desired_n_jobs)
    doRemoveBadChannels = 1  # currently uses the manual list... I doubt you'll ever want to turn this off.
    skipStuffThatKSGUIDoes = 1  # this variable is fundamentally outdated at this point and will either be irrelevant if you see this, or deleted if you don't.
    beginningAndEndToCutOff = [1, 1] ### saturation cutoff handling, for beginning and end
    loadSatsFromFile = 1 # allow loading saturations from file vs always recalculating
    doSaturationReplace = True


if all_VE_config.sessionwiseDriftCorrectionArguments == 'JeffreyRecommended': # contains si arguments. Likely to be nonspecific.

    bin_s_sessionCat = 6.0 # 6 is best. It is the only way I've managed to get something that looks good. Possibly eventually we will want to do this per-week though.
    silenceOrNoiseReplace_sessionwise = 'zeros' # zeros is significantly faster, 'noise' makes more sense theoretically but practically doesn't seem much different. Might work better with new SI version?
    if silenceOrNoiseReplace_sessionwise == 'silence': silenceOrNoiseReplace = 'zeros' # foolproofing

if all_VE_config.withinSessionSpikesortingArguments == 'JeffreyRecommended':
    silenceOrNoiseReplace_within = 'zeros' # have not systematically evaluated differences wrt spikesorting.
    if silenceOrNoiseReplace_within == 'silence': silenceOrNoiseReplace = 'zeros' # foolproofing

###### SAVING PARAMETERS SPIKESORTING

if all_VE_config.projectLabel == 'Jeffrey': # arguments handling how much code to run
    doPreprocessing = 1 # if you want to load your drift maps without recalculating them, turn this off.
    savePreprocessing = 1
    overwritePreprocessing = 1
    resaveMotionIfLoadingPreprocessing = 0 # if you turn off preprocessing, you can either resave new motion correction stuff or not. Kept off by default for safety.
    checkMotionPlotsOnline = 0 # turn this off if you don't want to view the plots.
    calculateSessionMotionDisplacement = 1 # Decides whether to calculate motion displacement when running get_drift_per_session. Should never not be on.
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

if all_VE_config.projectLabel == 'Jeffrey': # finds my sessions, sorts them in order, then initializes the Local_SetsOfConcatenatedSessions. I recommend using sort_np_sessions, it is useful for getting things in order.
    if any(list(all_VE_config.NAS_session_path.glob(all_VE_config.sessionString))):
        NAS_SessionsInOrder = sort_np_sessions(list(all_VE_config.NAS_session_path.glob(all_VE_config.sessionString)))
    else:
        NAS_SessionsInOrder = [[]] ### possibly in this case whatever is running doesn't need the NAS... Not sure if I want to allow it or not, but if not, then this will simply be a deferred error
    if any(NAS_SessionsInOrder): ### if you have NAS stuff... We are going to assume either you have local stuff or you want to create it.
        Local_SessionsInOrder = []
        for i, session in enumerate(NAS_SessionsInOrder): ### here, we make the folder hierarchy from scratch. We even make the directories.
            Local_SessionsInOrder.append(all_VE_config.local_session_path / Path(*session.parts[-1:]))
            Local_SessionsInOrder[i].mkdir(parents=True, exist_ok=True)
    elif any(list(all_VE_config.local_session_path.glob(all_VE_config.sessionString))): ### If you don't have NAS stuff, maybe you have untouched data locally?
        Local_SessionsInOrder = sort_np_sessions(list(all_VE_config.local_session_path.glob(all_VE_config.sessionString)))
    else:
        Local_SessionsInOrder = [[]] ### This state really should never be reached, and should break the code if you do.

    if not (len(Local_SessionsInOrder) == len(NAS_SessionsInOrder)):
        print('local and nas sessions not equivalent for some reason, and they should be. Figure it out or recreate the local from scratch.')
        breakitonpurpose
    Local_SetsOfConcatenatedSessions = []


if all_VE_config.frequencyOfConcatenation == 'weekly_heuristic': ### a currently unused system where one can batch-run many sets of sessions. The idea here was to, for example, concatenate every week within itself but not between weeks. I'll probably reuse some of this code to make three-week chunks eventually.
    ## aggregate sessions as long as they are less than two days apart. It will fail only to catch if I skep two weekdays. I still need to deal with month though.
    tempPerWeek = []
    countOfConcatenatedSessions = 1  # counting from 1
    week_session_correspondance = []
    week = 0
    for i, session in enumerate(Local_SessionsInOrder):
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
                    Local_SetsOfConcatenatedSessions.append(tempPerWeek)
                    tempPerWeek = []
                    tempPerWeek.append(session)
                    countOfConcatenatedSessions += 1
                else:
                    tempPerWeek.append(session)
            elif (currentDay - priorDay) > 2:  ## in this case, week is over
                Local_SetsOfConcatenatedSessions.append(tempPerWeek)
                tempPerWeek = []
                tempPerWeek.append(session)
                countOfConcatenatedSessions += 1
            else:
                tempPerWeek.append(session)
            priorDay = currentDay
            priorMonth = int(session_name[2:4])
            priorYear = int(session_name[4:8])
    Local_SetsOfConcatenatedSessions.append(tempPerWeek)  # this means that the final week will be appended, I think.
elif all_VE_config.frequencyOfConcatenation == 'do_everything':
    NAS_SetsOfConcatenatedSessions =[NAS_SessionsInOrder]
    Local_SetsOfConcatenatedSessions =[Local_SessionsInOrder]
    sessionSetName = 'everythingAllAtOnce'

###### stuff from versions of week-driftmapping that might be defunct now (I am basically certain it is)

record_motion_sessions = []
record_motion_weeks = []
record_space_bins_sessions = []
record_space_bins_weeks = []#recording drift/motion for UnitMatch later

week_session_correspondance = []
week = 0
last_session_previous_week = 0