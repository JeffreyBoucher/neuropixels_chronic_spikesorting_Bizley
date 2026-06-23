from pathlib import Path

from generally_useful_functions_Bizley.load_meta_file import load_meta_file


def getCorrectRecordingForRegion(imecFolderList,channel_map_to_use,localData=True):
    foundMeta = False
    for iiii, currentRecording in enumerate(imecFolderList): # go through each recording. Can't assume an imec number means anything, so read all meta files.
        if localData:
            tempRecording = currentRecording / 'catgt_CorrectedRaw'
        else:
            tempRecording = currentRecording
        trial_meta = load_meta_file(list(tempRecording.glob('*.meta'))[0])
        this_imro_file_name = Path(trial_meta['imroFile']).stem + '.imro'
        if channel_map_to_use == this_imro_file_name:
            if not foundMeta:
                foundMeta = True
                metaFileData = trial_meta.copy()
                usedImec = metaFileData['fileName'][-12:-4] ### sketchy but seems like it should always work if you are using an imec, and will throw an error otherwise... Might be an issue for onebox.
                if not (usedImec[0:4] == 'imec'):
                    print('something went wrong determining the imec')
                    breakitonpurpose
                pathToCurrentProbe = tempRecording
            else:
                errorMoreThanOneOfSameMap
        if foundMeta == False:
            metaFileData = None
            usedImec = None
            pathToCurrentProbe = None
            ### we depend on the outside function to deal with these being none. It will certainly happen
            ### in any case where there was a missed recording.
        pass



    return metaFileData,usedImec,pathToCurrentProbe