# Goal with this script is to load the sorting output of any
# particular spikesorting attempt and then apply functions which
# will help me understand it. The scope will be dependent on whatever
# can or cannot be done with the sortings output. So: pre-phy, and potentially
# excluding useful spikesorting analyses which don't involve the sorting output.


from pathlib import Path

import spikeinterface.full as si

from neuropixels_visualisation.spikesorting import spikesorting_postprocessing
def main():
    ### decide on output folder
    formatToLookAt = 'singleSession'
    if formatToLookAt == 'concatenated':
        ferret = 'F2302_Challah'
        bonusName = 'All_ACx_Top'
        recordingZone = 'ACx_Challah'  # this doesn't seem to work!
        #output_folder_name = 'TestOutputAlsoMidJune' # may also be "Output"
        output_folder_name = 'Output'  # may also be "Output"
        output_folder = Path('/home/sjjgjbo/Scratch/FerretData/') / output_folder_name
        output_folder_sorted = output_folder / 'spikesorted' / ferret / recordingZone / bonusName
        #output_folder_sorted = output_folder / 'testSorted' / ferret / recordingZone / bonusName

        ### load the sorting
        sorter_folder = Path('/home/sjjgjbo/Scratch/FerretData/' +  output_folder_name + '/tempDir/F2302_Challah/ACx_Challah/All_ACx_Top/tempDir/kilosort3_output')
        sorting = si.read_sorter_folder(sorter_folder)
    elif formatToLookAt == 'singleSession':
        session_path = Path('Z:/Data/Neuropixels')
        ferret = 'F2302_Challah'  # chooses the ferret folder.
        # ferret = 'F2105_Clove'  # chooses the ferret folder.
        # ferret = 'F1903_Trifle'  # chooses the ferret folder.
        session_name = '06122024_AM_Challah_g0'  # chooses the session folder. Use this to select specific sessions. You can write something to do this batch later if you want, but seems likely that you'll instead be changing this as you go.
        # session_name = '021122_trifle_am_g0'
        output_folder = Path('D:/Jeffrey/Output/Tones/')
        output_folder_sorted = Path('D:/Jeffrey/Output/Tones/spikesorted/F2302_Challah/06122024_AM_Challah_g0/KiloSortSortingExtractor') ### think this is right
        sorter_folder = Path('D:/Jeffrey/Output/Tones/tempDir/kilosort4_output')
        sorting = si.read_sorter_folder(sorter_folder)
        dp = session_path / ferret / session_name

    rerunPostprocessing = 1
    print('Running this code')
    if rerunPostprocessing:
        sorting = spikesorting_postprocessing(sorting, output_folder=output_folder_sorted)
    print('hahahaha')
if __name__ == '__main__':
    main()