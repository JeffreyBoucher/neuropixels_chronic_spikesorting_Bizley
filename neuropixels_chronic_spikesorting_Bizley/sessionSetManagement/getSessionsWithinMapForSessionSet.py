### gets sessions within map for session sets extracted with SpecifySessionSetParameters
from pathlib import Path
import neuropixels_chronic_spikesorting_Bizley.outerLoopConfigs as outerLoopConfigs
import neuropixels_chronic_spikesorting_Bizley.all_VE_config as all_VE_config

from neuropixels_chronic_spikesorting_Bizley.helpers.helpers_spikesorting_scripts import sort_np_sessions


def getSessionsWithinMapForSessionSet(sessionSetParams):
    ### note this function won't work unless you have a local and long term storage location. But you always should.
    ### Also I am not actually checking the map at all here! I should be.
    NAS_SessionsInOrder_rebuilt = []
    for i,currentDate in enumerate(sessionSetParams['IncludedDates']): ### included dates should already be in order
        if any(list(all_VE_config.NAS_session_path.glob(currentDate + '*'))): ### sessionstring for this case should be something which gets "everything". Mainly because getting a regex which deals with dates intuitively is needlessly tricky. We'll filter it after.
            NAS_SessionsInOrder = sort_np_sessions(list(all_VE_config.NAS_session_path.glob(currentDate + '*')))
            if len(NAS_SessionsInOrder) > 2:
                print(NAS_SessionsInOrder)
                pass ### so I put this here to watch the order of things when I did a second AM or pm session... Largely though things should be in the right order, because putting a 2 or a g1 somewhere orders things correctly without skipping over the AM PM
            for ii,session in enumerate(NAS_SessionsInOrder):
                NAS_SessionsInOrder_rebuilt.append(session) ### do it like this to avoid lists in lists
    NAS_SessionsInOrder = NAS_SessionsInOrder_rebuilt
    if any(NAS_SessionsInOrder): ### if you have NAS stuff... We are going to assume either you have local stuff or you want to create it.
        Local_SessionsInOrder = []
        for i, session in enumerate(NAS_SessionsInOrder): ### here, we make the folder hierarchy from scratch. We even make the directories.
            Local_SessionsInOrder.append(all_VE_config.local_session_path / Path(*session.parts[-1:]))
            Local_SessionsInOrder[i].mkdir(parents=True, exist_ok=True)
    else:
        breakitonpurpose ### you shouldn't reach this one because the one above this will have already triggered.
    if not (len(Local_SessionsInOrder) == len(NAS_SessionsInOrder)):
        print('local and nas sessions not equivalent for some reason, and they should be. Figure it out or recreate the local from scratch.')
        breakitonpurpose ### should never reach it in this case unless something very buggy is happening.


    return NAS_SessionsInOrder,Local_SessionsInOrder