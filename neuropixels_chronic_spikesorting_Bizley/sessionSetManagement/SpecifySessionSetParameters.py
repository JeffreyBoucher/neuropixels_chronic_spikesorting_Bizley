### goal is to take as input a ferret and "session set" name and to gather session set parameters.
# at the time of writing I have not figured out what I want re: regions. I think it would make the most
# sense to have that be another, third input into this function. So that's how I'll start.

### note also that, while at the time of writing I am not using this, there is potential to include set-specific
### spikesorting parameters here, or even session specific. Need to save them into SessionSetParams and then write code
### that can interpret it later. 
from datetime import date, time, datetime, timedelta
import pandas as pd

def SpecifySessionSetParameters(ferret,SessionSetName,BrainRegion):
    SessionSetParams = dict()
    dateFormat ='%d%m%Y' ### This is my, Jeff's, date format. Lower case means two digits, capital means 4 (for years at least).
    if ferret == 'F2302_Challah':
        SessionSetParams['ferret'] = 'F2302_Challah'
        SessionSetParams['ferret_no_id'] = 'Challah'
        if SessionSetName=='SessionSet01':
            ### first month of recording. Saturating signal on left (80% sure, maybe right) spout licks.#
            ### otherwise recordings are good and I have used this as an investigation set.
            ### simultaneous top AC and anterior (shank 0) PFC. 80% sure I have the direction correct re: shank 0.
            SessionSetParams['DateStart'] = '13052024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '25052024'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_groundref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_b1_horizontal_band_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank0_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank0.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet02':
            ### simultaneous top AC and posterior (shank 3) PFC
            SessionSetParams['DateStart'] = '03062024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '22062024'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_groundref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_b1_horizontal_band_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank3_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank3.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet03':
            ### simultaneous top AC and anterior (shank 0) PFC. Yes, again, I wanted more without the saturation. Great decision, I would have not had leftward data otherwise.
            SessionSetParams['DateStart'] = '01072024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '20072024'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_groundref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_b1_horizontal_band_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank0_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank0.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet04':
            ### simultaneous top AC and mid-anterior (shank 1) PFC.
            SessionSetParams['DateStart'] = '29072024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '17082024'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_groundref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_b1_horizontal_band_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                ### I swapped to tipref in the middle of the final week, so I may find I am missing some stuff... I could potentially try unitmatching across or doing both maps.
                SessionSetParams['sessionSetLabel'] = 'PFC_shank1_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank1.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet05':
            ### simultaneous top AC and mid-posterior (shank 2) PFC.
            SessionSetParams['DateStart'] = '26082024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '14092024'
            if BrainRegion == 'AC':
                ### swapped from ground ref to join tip in the middle of the first week. Practically means lost data.
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_b1_horizontal_band_joint_tip.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                # includes a session called "26082024_AM_Challah_g0_badAC" which should probably be excluded.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank2_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank2_tip_ref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet06':
            ### simultaneous bottom AC and anterior (shank 0) PFC.
            SessionSetParams['DateStart'] = '23092024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '12102024'
            if BrainRegion == 'AC':
                ### Swapped to bottom AC!
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Bottom_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_bottom_b1_horizontal_band_joint_tip.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank0_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank0_tip_ref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet07':
            ### simultaneous bottom AC and anterior (shank 0) PFC... Apparently I did it again, possibly noise was getting bad
            SessionSetParams['DateStart'] = '21102024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '09112024'
            if BrainRegion == 'AC':
                ### at the time of writing the final week isn't on the NAS yet and neither is the next session set. I am backing things up again on my local drive and plan to move them via filezilla and mammoth on Monday
                ### also because of this I don't tecnically know what the channel map was, but it was almost cetainly bottom_
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Bottom_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_bottom_b1_horizontal_band_joint_tip.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank0_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank0_tip_ref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet08':
            ### I have not recorded what this channel map was in my spreadsheet, so would need to look manually, but probably won't bother because of the PFC noise.
            SessionSetParams['DateStart'] = '18112024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '06122024'
            if BrainRegion == 'AC':
                ### at the time of writing the final week isn't on the NAS yet and neither is the next session set. I am backing things up again on my local drive and plan to move them via filezilla and mammoth on Monday
                ### also because of this I don't tecnically know what the channel map was, but it was almost cetainly bottom_
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Bottom_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_bottom_b1_horizontal_band_joint_tip.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [66,105,149,170,175,209,210,239,354,369] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_shank0_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Challah_top_PFC_shank0_tip_ref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [21,109,133,170,181,202,295,305,308,310,327,329,339] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
        elif SessionSetName=='SessionSet09':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet10':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet11':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet12':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet13':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet14':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet15':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet16':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet17':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet18':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet19':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet20':
            if BrainRegion == 'AC':
                pass
            elif BrainRegion == 'PFC':
                pass
    elif ferret == 'F2301_Boule':
        ##### FROM HERE ON PROBEMAPS NOT CHECKED
        if SessionSetName=='SessionSet01':
            ### Top of ACx and the top front and back PFC.
            SessionSetParams['DateStart'] = '02122024' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '21122024'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_ground_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_Top_Anterior_Posterior_' + SessionSetName
                SessionSetParams['channel_map_to_use'] = 'Boule_PFC_Shanks_0_3.imro'
                SessionSetParams['badChannelList'] = [19,128,161,291,315]
                pass
        elif SessionSetName=='SessionSet02':
            ### Top of ACx and the top front and back PFC. Looks like I did a second rep of it.
            SessionSetParams['DateStart'] = '06012025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '25012025'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_ground_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_Top_Anterior_Posterior_' + SessionSetName
                SessionSetParams['channel_map_to_use'] = 'Boule_PFC_Shanks_0_3.imro'
                SessionSetParams['badChannelList'] = [19,128,161,291,315]
                pass
        elif SessionSetName=='SessionSet03':
            SessionSetParams['DateStart'] = '03022025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '22022025'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_ground_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_ground.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_Top_Middle_Shanks_' + SessionSetName
                SessionSetParams['channel_map_to_use'] = 'Boule_PFC_Shanks_1_2.imro'
                SessionSetParams['badChannelList'] = [19,128,161,291,315]
                pass
        elif SessionSetName=='SessionSet04':
            ## this session has some issues with freezing which mean I won't want to use it for a bit
            ### independently, I swapped to tipref in everything for this one.
            SessionSetParams['DateStart'] = '03032025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '22032025'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_tipref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_Top_Middle_Shanks_tipref_' + SessionSetName
                SessionSetParams['channel_map_to_use'] = 'Boule_PFC_Shanks_1_2_tipref.imro'
                SessionSetParams['badChannelList'] = [19,128,161,291,315]
                pass
        elif SessionSetName=='SessionSet05':
            ### Top of ACx and the top front and back PFC. Looks like I did a third rep of it... Certainly overkill, though this happens to be my cleanest session set in terms of complications. I could potentially use this to check consistency of responses though, at least with PFC.
            ### it is also tipref.
            ### context clues tell me I might have set it to this without too much thought because of
            ### grant prioritization, so I forgive myself. It's not as if I can't use this data, I am
            ### glad it is here!
            SessionSetParams['DateStart'] = '31032025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '19042025'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_tipref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                SessionSetParams['sessionSetLabel'] = 'PFC_Top_Anterior_Posterior_tipref_' + SessionSetName
                SessionSetParams['channel_map_to_use'] = 'Boule_PFC_Shanks_0_3_tipref.imro'
                SessionSetParams['badChannelList'] = [19,128,161,291,315]
                pass
        elif SessionSetName=='SessionSet06':
            ### yes the large jump in dates is correct
            ### Here is where I start looking into some hippocampus in addition to some AC.
            ### In the short term, I expect this means I should stop because most of my data won't
            ### be AC and I have few enough cells as it is! So I commented things out. To be looked at later.
            SessionSetParams['DateStart'] = '02062025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '21062025'
            if BrainRegion == 'AC':
                # SessionSetParams['sessionSetLabel'] = # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                # SessionSetParams['channel_map_to_use'] = '' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                # SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet07':
            SessionSetParams['DateStart'] = '30062025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '19072025'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_tipref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet08':
            SessionSetParams['DateStart'] = '28072025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '16082025'
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_tipref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                pass
        elif SessionSetName=='SessionSet09':
            SessionSetParams['DateStart'] = '29092025' ## day month year, to my chagrin. My plan will be I think to parse these strings in a very specific way, come up with a date range, and then generate a massive regex or something which allows for fitting to any of these...
            SessionSetParams['DateEnd'] = '13102025' ### after this day (a Monday) the AC probe breaks. PFC still good for some time after, but this is the end of this scope.
            if BrainRegion == 'AC':
                SessionSetParams['sessionSetLabel'] = 'All_ACx_Top_tipref_' + SessionSetName # this will become the folder name for the output, and is also used to create the regular expression that identifies the sessions you want to include.
                SessionSetParams['channel_map_to_use'] = 'Boule_top_ACx_tipref.imro' # the name of the channel map. This is a more reliable way of identifying a probe/map, since we don't tend to use the same maps on multiple probes.
                SessionSetParams['badChannelList'] = [42,116,143,175] # bad channels, manually found and labeled. These tend to be highly reliable from implant to end-of-experiment. They will be removed from all analyses.
                pass
            elif BrainRegion == 'PFC':
                pass
    ### now, calculate all days within each daterange right here! No reason not to.
    startdate = datetime.strptime(SessionSetParams['DateStart'],dateFormat).date() ### the string at the end recognizes MY date format.
    enddate = datetime.strptime(SessionSetParams['DateEnd'],dateFormat).date()
    SessionSetParams['IncludedDates'] = pd.date_range(startdate,enddate-timedelta(days=1),freq='d').strftime("%d%m%Y")
    return SessionSetParams







