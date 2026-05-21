from pathlib import Path

###### all project-specific outer loop variables should be handled by this file. ##########

###### Override variables; default to off unless you turn them on.

SurveyOverride = False

###### PERSONALIZATION ARGUMENTS ######
projectLabel = 'Jeffrey' ## put your name or project or something to specify which sets of variables you will access.
sessionString = '[0-9][0-9]*' # this is a default regular expression to get the correct filesets to analyze.

if projectLabel == 'Jeffrey':
    computerUsed = 'JeffreyLabDesktop' # if you plan to analyze on multiple computers and have different local addresses for things, this is how you can account for that.

    # the below are parameters you may want to keep. If you don't, make your own and name it something else.
    sessionwiseDriftCorrectionArguments = 'JeffreyRecommended'
    generalSpikesortingArguments = 'JeffreyRecommended'
    withinSessionSpikesortingArguments = 'JeffreyRecommended'
    plottingArguments = 'JeffreyRecommended'
    make_multirecording_info = 'JeffreyRecommended'

###### FERRET AND FILE NAME PARAMETERS

if projectLabel == 'Jeffrey': ### here will be ferret specific arguments
    ### recordingZones are sets of sessions you want to analyze together, because, for example, they share a probemap and timeframe. Each can have a specific set of parameters associated with it.
    # recordingZone = 'ACx_Boule_top_tipref_March2025'
    #recordingZone = 'PFC_Boule_Borders_Top_GroundrefIThink'
    #recordingZone = 'PFC_Boule_Borders_Top'
    recordingZone = 'PFC_shank0_Challah'
    #recordingZone = 'PFC_shank3_Challah'
    #recordingZone = "ACx_Challah_top_groundref_June2024"
    # recordingZone = "ACx_Challah_top_groundref_May2024"
    # recordingZone = "Challah_Survey" ### intended to be an adaptable set for if I want to spikesort a specific survey
    sessionsToDo = 'all'
    frequencyOfConcatenation = 'do_everything' #'weekly_heuristic' or 'do_everything'. do_everything is contained per probemap.

###### PROBEBANK SPECIFIC PARAMETERS JEFFREY

if projectLabel == 'Jeffrey': # contains probemap specific stuff. First set contains commentary where necessary.
    if recordingZone == 'ACx_Challah_top_groundref_May2024':
        stream_id = 'imec0.ap' # unreliably identifies the probe. Unfortunately can misidentify when small details change. Currently code relies on it, we need to find ways around it though.
        sessionSetLabel = 'All_ACx_Top_groundref_MonthOfMay2024' # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
        channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
        #channel_map_to_use_other_ref = 'Challah_top_b1_horizontal_band_joint_tip.imro' ### this wouldn't actually work...
        badChannelList = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
        ferret = 'F2302_Challah'
        ferret_no_id = 'Challah'
        doMultipleShanks = True # this will never not be true and I'll move it elsewhere pretty soon. Because even with one shank, you want to treat it as multiple.
    elif recordingZone == 'Challah_Survey':
        stream_id = 'imec0.ap' ### imec0.ap for ACx
        sessionSetLabel = 'Challah_Survey_ACx_2026_09_03'
        channel_map_to_use = '' ### don't know how to deal with this yet...
        #channel_map_to_use_other_ref = 'Challah_top_b1_horizontal_band_joint_tip.imro' ### this wouldn't actually work...
        badChannelList = [66,105,149,170,175,209,210,239,354,369]
        ferret = 'F2302_Challah'
        ferret_no_id = 'Challah'
        doMultipleShanks = True
        SurveyOverride = True
        dateStringToLoadUp = '2026_03_09'
    elif recordingZone == 'ACx_Challah_top_groundref_June2024':
        stream_id = 'imec0.ap'
        sessionSetLabel = 'All_ACx_Top_groundref_MonthOfJune2024'
        channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro'
        #channel_map_to_use_other_ref = 'Challah_top_b1_horizontal_band_joint_tip.imro' ### this wouldn't actually work...
        badChannelList = [66,105,149,170,175,209,210,239,354,369]
        ferret = 'F2302_Challah'
        ferret_no_id = 'Challah'
        doMultipleShanks = True
    elif recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        sessionSetLabel = 'PFC_shank0_MonthOfMay2024'#'PFC_shank0'
        channel_map_to_use = 'Challah_top_PFC_shank0.imro'
        badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
        # something else also. Need to read metadata
        ferret = 'F2302_Challah'
        ferret_no_id = 'Challah'
        doMultipleShanks = True
    elif recordingZone == 'PFC_shank3_Challah':
        stream_id = 'imec1.ap'
        sessionSetLabel = 'PFC_shank3'
        channel_map_to_use = 'Challah_top_PFC_shank3.imro'
        badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
        # something else also. Need to read metadata
        ferret = 'F2302_Challah'
        ferret_no_id = 'Challah'
        doMultipleShanks = True
    elif recordingZone == 'ACx_Boule':
        stream_id = 'imec1.ap'
        sessionSetLabel = 'All_ACx_Top'
        channel_map_to_use = 'Boule_top_ACx_tipref.imro'
        # channel_map_to_use_other_ref = ''
        ferret = 'F2301_Boule'
        ferret_no_id = 'Boule'
    elif recordingZone == 'ACx_Boule_top_tipref_March2025':
        stream_id = 'imec1.ap'
        sessionSetLabel = 'All_ACx_Top_tipref_March2025'
        channel_map_to_use = 'Boule_top_ACx_tipref.imro'
        badChannelList = [42,116,143,175]
        ferret = 'F2301_Boule'
        ferret_no_id = 'Boule'
        doMultipleShanks = True
    elif recordingZone == 'PFC_Boule_Center_Top':
        stream_id = 'imec0.ap'
        sessionSetLabel = 'PFC_Shanks_1_2'
        channel_map_to_use = 'Boule_PFC_Shanks_1_2_tipref.imro'
        ferret = 'F2301_Boule'
        ferret_no_id = 'Boule'
    elif recordingZone == "PFC_Boule_Borders_Top":
        stream_id = 'imec0.ap'
        sessionSetLabel = 'PFC_Shanks_0_3'
        channel_map_to_use = 'Boule_PFC_Shanks_0_3_tipRef.imro'
        badChannelList = [19,128,161,291,315]
        ferret = 'F2301_Boule'
        ferret_no_id = 'Boule'
    elif recordingZone == "PFC_Boule_Borders_Top_GroundrefIThink":
        stream_id = 'imec0.ap'
        sessionSetLabel = 'PFC_Shanks_0_3'
        channel_map_to_use = 'Boule_PFC_Shanks_0_3.imro'
        badChannelList = [19, 128, 161, 291, 315]
        # channel_map_to_use_other_ref = ''
        ferret = 'F2301_Boule'
        ferret_no_id = 'Boule'

###### POSSIBLY REDUNDANT SESSION DRILL-DOWN FOR JEFFREY

if projectLabel == 'Jeffrey': # manages the highest-level selection of sessions via regex. Something I will eventually rethink, as this is really only suitable for analyzing small, particular sets of sessions and not analyzing everything in bulk.
    if sessionSetLabel == 'All_ACx_Top':
        sessionString = '[0-9][0-9]*' ### this actually selects more than just the top
    elif sessionSetLabel == 'Tens_Of_June':
        sessionString = '1[0-9]06*'
    elif sessionSetLabel == 'TheFirstDay':
        sessionString = '1305*'
    elif sessionSetLabel == 'TheFirstSession':
        sessionString = '1305*AM*'
    elif 'May2024' in sessionSetLabel:
        sessionString = '[0-9][0-9]052024*'
    elif 'June2024' in sessionSetLabel:
        sessionString = '[0-9][0-9]062024*'
    elif 'March2025' in sessionSetLabel: ### next time I should automate this in some way or change the infrastructure entirely. Probably instead of doing this month based I want to do it three-week-chunk based, and I already have code for that, it'll just be about moving it.
        sessionString = '[0-9][0-9]032025*'
    elif SurveyOverride:
        sessionString = dateStringToLoadUp + '*'
    else:
        sessionString = '[0-9][0-9]*'

###### folder names
if projectLabel == 'Jeffrey':
    if computerUsed == 'JeffreyLabDesktop': # contains folder and file arguments, including higher-level ones like ferret name. Likely to be specific to my project
        output_folder = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output') ### this is the lower-level spot for spikesorted output, specified to ferret after the if statement.
        motionMapFolder = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs/SessionSetDriftmaps')
    elif computerUsed == 'ExternalDriveD':
        output_folder = Path('D:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output')
    sessionSetSortedFolder = output_folder / ferret / sessionSetLabel
    NAS_neural_data = Path('Z:/Data/Neuropixels/')
    if SurveyOverride:
        NAS_neural_data = NAS_neural_data / Path('Surveys')
    NAS_session_path = NAS_neural_data / Path(ferret)
    local_neural_data = Path('D:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs') # I think these are just for saturated zones, though they are in the same format as the folders on the NAS...
    if SurveyOverride:
        local_neural_data = Path('D:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs/Surveys')
    local_session_path = local_neural_data / Path(ferret)

    behavior_path = Path("C:/Users/jeff/Dropbox/Data/" +ferret)
    figure_folder = Path('D:/Jeffrey/Projects/SpeechAndNoise/figures/')
    FolderWithPickles = output_folder  / Path('tempDir/' +ferret + '/' + sessionSetLabel + '/FolderWithPickles') ### I will want to change this Once I undersd



###### very basic and probably unnecessary information, but which is good to have

probeType = 'neuropixels'