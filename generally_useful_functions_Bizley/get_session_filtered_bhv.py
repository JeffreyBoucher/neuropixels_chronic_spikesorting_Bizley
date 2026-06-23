### function to filter trials from a behavior file which had been loaded using load_bhv_file.
from generally_useful_functions_Bizley.apply_behavior_filter import apply_behavior_filter

def get_session_filtered_bhv(filtered_bhv,specific_saturation_tracking_folder,filter_trials,window,evtype,metaFileData,pulseDictThisSession):
    ## IN
    # filtered_bhv: the behavior pandas array
    # specific_saturation_tracking_folder: the location of the saturation exclusion csv file. Should be in the local data folder (session/probe specific)
    # filter_trials: the set of filter strings, used by the "apply_filter" function
    # window: the window that, specifically, you want to be accounted for when you exclude things that do or don't happen in the window
    # evtype: the place where you want to align trials, as a string
    ## OUT
    # filtered_bhv: the bhv file with the trials you don't want removed.
    # ev_times: the "event times" which are just values from within filtered_bhv, extracted using the string evtype.
    if filtered_bhv is None: ### I forget exactly what this means but I imagine it is related to the behavior (bhv) file.
        return None,None ### might break because no ev_times and type_stim ot because copy doesn't work.

    if len(filter_trials) > 0:
        for filter in filter_trials:
            filtered_bhv = apply_behavior_filter(filtered_bhv, filter,window,specific_saturation_tracking_folder,evtype,metaFileData,pulseDictThisSession)
    # Get event times
    #print(filtered_bhv.size)
    if not len(filtered_bhv):
        return filtered_bhv,None ### might break because no ev_times and type_stim
    ev_times = filtered_bhv[evtype].to_numpy()

    return filtered_bhv, ev_times