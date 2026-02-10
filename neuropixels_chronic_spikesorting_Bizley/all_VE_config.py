from pathlib import Path

###### all project-specific outer loop variables should be handled by this file. ##########

###### PERSONALIZATION ARGUMENTS ######
projectLabel = 'Jeffrey' ## put your name or project or something to specify which sets of variables you will access.

if projectLabel == 'Jeffrey':
    computerUsed = 'JeffreyLabDesktop'

    # the below are parameters you may want to keep. If you don't, make your own and name it something else.
    sessionwiseDriftCorrectionArguments = 'JeffreyRecommended'
    generalSpikesortingArguments = 'JeffreyRecommended'
    withinSessionSpikesortingArguments = 'JeffreyRecommended'
    plottingArguments = 'JeffreyRecommended'
    make_multirecording_info = 'JeffreyRecommended'

###### FERRET AND FILE NAME PARAMETERS

if projectLabel == 'Jeffrey': ### here will be ferret specific arguments
    #recordingZone = 'PFC_Boule_Borders_Top_GroundrefIThink'
    #recordingZone = 'PFC_Boule_Borders_Top'
    #recordingZone = 'PFC_shank0_Challah'
    #recordingZone = 'PFC_shank3_Challah'
    recordingZone = "ACx_Challah_top_groundref"
    sessionsToDo = 'all'
    frequencyOfConcatenation = 'do_everything' #'weekly_heuristic' or 'do_everything'. do_everything is contained per probemap.

###### PROBEBANK SPECIFIC PARAMETERS JEFFREY

if projectLabel == 'Jeffrey': # contains probemap specific stuff. Highly specific to my (Jeffrey's) project.
    if recordingZone == 'ACx_Challah_top_groundref':
        stream_id = 'imec0.ap'
        sessionSetLabel = 'All_ACx_Top_groundref_MonthOfMay2024'
        channel_map_to_use = 'Challah_top_b1_horizontal_band_ground.imro'
        #channel_map_to_use_other_ref = 'Challah_top_b1_horizontal_band_joint_tip.imro' ### this wouldn't actually work...
        badChannelList = [66,105,149,170,175,209,210,239,354,369]
        ferret = 'F2302_Challah'
        doMultipleShanks = True
    elif recordingZone == 'PFC_shank0_Challah':
        stream_id = 'imec1.ap'
        sessionSetLabel = 'PFC_shank0_MonthOfMay2024'#'PFC_shank0'
        channel_map_to_use = 'Challah_top_PFC_shank0.imro'
        badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
        # something else also. Need to read metadata
        ferret = 'F2302_Challah'
        doMultipleShanks = False
    elif recordingZone == 'PFC_shank3_Challah':
        stream_id = 'imec1.ap'
        sessionSetLabel = 'PFC_shank3'
        channel_map_to_use = 'Challah_top_PFC_shank3.imro'
        badChannelList = [21,109,133,170,181,202,295,305,308,310,327,329,339]
        # something else also. Need to read metadata
        ferret = 'F2302_Challah'
        doMultipleShanks = False
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

###### POSSIBLY REDUNDANT SESSION DRILL-DOWN FOR JEFFREY

if projectLabel == 'Jeffrey': # manages the highest-level selection of sessions via regex. A bit outdated now that session-sets are implemented. I still occasionally try and use it.
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

###### folder names

if computerUsed == 'JeffreyLabDesktop': # contains folder and file arguments, including higher-level ones like ferret name. Likely to be specific to my project

    output_folder = Path('C:\\Jeffrey\\Projects\\SpeechAndNoise\\Spikesorting_Output')
    motionMapFolder = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs/SessionSetDriftmaps')
elif computerUsed == 'ExternalDriveD':
    output_folder = Path('D:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output')
session_path = Path('Z:/Data/Neuropixels/' + ferret)  # path to where all relevant sessions are stored
saturatedZonesLocations = Path('C:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Inputs') # I think these are just for saturated zones, though they are in the same format as the folders on the NAS...
