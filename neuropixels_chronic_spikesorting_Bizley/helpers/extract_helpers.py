import os
import os.path as op
import re
import numpy as np
import scipy.io as sio
import collections
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import date, time, datetime, timedelta

def matstruct2dict(mat_struct):
    """
    A functiom which construct a dict from matlab structure
    Load matlab structure with sio.loadmat(mat_fname, struct_as_record=False, squeeze_me=True)
    """
    
    dict_output = collections.defaultdict(list)
    if np.size(mat_struct) > 1:
        for matobj in mat_struct:
            [dict_output[strg].append(matobj.__dict__[strg]) for strg in matobj._fieldnames]
    else:
        matobj = mat_struct
        [dict_output[strg].append(matobj.__dict__[strg]) for strg in matobj._fieldnames]
    return dict_output


def load_bhv_file(filename):
    ''' load_bhv_file This function loads a ferret behavioural file and returns the data in a pandas dataframe. '''
    matfile = sio.loadmat(filename,
                            struct_as_record=False,
                            squeeze_me=True)
    datadict = matstruct2dict(matfile['data'])
    data = pd.DataFrame(data=datadict)
    if len(data.keys())>10:
        data['fName'] = filename
        tFname= filename.split('\\')[-1]
        levelReg = re.findall(r'\d+',tFname.split()[1])
        data['Level'] = levelReg[0]
        if 'attenOrder' in data.keys():
            data['MultipleDisLvls'] = len(np.unique(data['attenOrder'][len(data['attenOrder'])-1]))
        else:
            data['MultipleDisLvls'] = 1


        currAtten = np.zeros(len(data))
        allTrialCount = np.zeros(len(data))
        for j in range(len(data)):
            if 'attenOrder' in data.keys(): 
                if type(data['attenOrder'][j])==int:
                    currAtten[j] = data['attenOrder'][j]
                elif np.size(data['attenOrder'][j])==0:
                    currAtten[j] = float('NaN')
                elif np.size(data['attenOrder'][j])==1:
                    currAtten[j] = data['attenOrder'][j]
                else:
                    currAtten[j] = data['attenOrder'][j][-1]
            else:
                if 'distractorAtten' in data.keys():                                
                    if type(data['distractorAtten'][j]) == str:
                        currAtten[j] = 120
                    elif type(data['distractorAtten'][j]) == int:
                        currAtten[j] = data['distractorAtten'][j]
                    elif np.size(data['distractorAtten'][j])==0:
                        currAtten[j] = float('NaN')    
                    elif(np.size(data['distractorAtten'][j])>1):
                        currAtten[j] = np.unique(data['distractorAtten'][j])
                    else:
                        currAtten[j] = data['distractorAtten'][j]
                else:
                    currAtten[j] = 120
                    
            allTrialCount[j] = j

        data['currAtten'] = currAtten
        data['allTrialCount'] = allTrialCount

        wordTime = np.zeros(len(data),10)
        wordType = np.zeros(len(data),10)
        for j in range(len(data)):
            wordTime[j,0] = data['startTrialLick'][j]
            for i, word in enumerate(data['distractor'][j]):
                if i < 10:
                    wordTime[j,i+1] = wordTime[j,i] + data['dDurs'][j][i]
                if word == '1':
                    wordType[j,i] = 1
                elif ((word == 10 or word == 26) and data['talker'][j] == 1) or (word == 18 and data['talker'][j] == 2):
                    wordType[j,i] = 2
                else:
                    wordType[j,i] = 3
        data['wordTime'] = wordTime
        data['wordType'] = wordType

        return data

def updateFileInfo(tempFolder):
    '''
    updateFileInfo This function updates the file information of all of the
    ferret behavioural files.  This extracts and stores the date, level and
    filename for each data file to be used for subsequent extraction.
    '''

    # Get the relevant files
    filenames = [fname for fname in os.listdir(tempFolder) 
                if fname.endswith('.mat')]
    rejectTerms = ['PASSIVE','matlab','fileInfo','Recording','Female','Instruments',
                   'sedated','FRA']
    for term in rejectTerms:
        filenames = [fname for fname in filenames if not term in fname]

    # Date files and level
    dataLevel = []
    dataString = []
    dataDates = []
    for fname in filenames:
        f = fname.split('.mat')[0]
        try:
            [dateStr, levelStr, timeStr] = f.split()
        except:
            continue
            # import pdb; pdb.set_trace()
        dateStr = [int(elem) for elem in dateStr.split('_')]
        timeStr = [int(elem) for elem in timeStr.split('_')]
        dateformat = "{:02d}-{:02d}-{:4d} {:02d}:{:02d}".format(
            dateStr[0],dateStr[1],dateStr[2],timeStr[0],timeStr[1])

        levelReg = re.findall(r"\d+",levelStr)
        if not levelReg:
            level = float('NaN')
        else:
            level = levelReg[0]
            

        dataLevel.append(level)
        dataString.append(op.join(tempFolder,fname))
        dataDates.append(datetime.strptime(dateformat,'%d-%m-%Y %H:%M'))

    # save fileInfo file
    fileInfo = dict(dataLevel=np.array(dataLevel),
                    dataString=np.array(dataString),
                    dataDates=np.array(dataDates))
    np.save(op.join(tempFolder,'fileInfo.npy'),fileInfo)

    return fileInfo

def findBhvData(blockname, path):
    # Find the behavioural data for a given block name
    path = Path(path)
    if (path / 'BhvBlockfiles.csv').is_file():
        df = pd.read_csv(path / 'BhvBlockfiles.csv')
    else:
        df = pd.DataFrame(columns=['BehaviourFile', 'BlockRec'])
    
    # Find the block file name
    BehaviourFile = df.loc[df['BlockRec'] == blockname, 'BehaviourFile']

    if blockname == 'BlockNellie-2':
        return('30_6_2022 level41_Trifle15SSN.txt 15_45.mat')

    if blockname == 'BlockNellie-3':
        return(BehaviourFile[81])

    if blockname == 'BlockNellie-4':
        print(blockname)

    if len(BehaviourFile) == 0:
        df = generateBlockBhvDataFrame(path, df)
        df.to_csv(path / 'BhvBlockfiles.csv', index=False)
        BehaviourFile = df.loc[df['BlockRec'] == blockname, 'BehaviourFile']

    # assert len(BehaviourFile) > 0, 'No block file found for block ' + blockname
    if len(BehaviourFile) == 0:
        print(f'No block file found for block {blockname}')
        return None


    BehaviourFile = BehaviourFile.values[-1]
    
    return BehaviourFile

def generateBlockBhvDataFrame(path, df):
    # Generate a dataframe of all the behavioural data files in the folder
    ferret = path.name
    # filenames = [f.name for f in path.glob(f'*{ferret[6:]}*.mat') 
    #             if f.name not in df['BehaviourFile'].values]

    search_str = ferret[6:]

    # Convert ferret[6:] to lowercase for case-insensitive comparison
    search_str = ferret[6:].lower()

    # Check if search_str is 'windolene', if yes, consider both 'windolene' and 'windowlene' in the pattern
    if search_str == 'windolene':
        pattern = re.compile(r'windolene|windowlene', re.IGNORECASE)
    else:
        pattern = re.compile(f'{search_str}', re.IGNORECASE)

    # Iterate over all .mat files in the directory
    filenames = [
        f.name for f in path.glob('*.mat')
        if pattern.search(f.name) and f.name not in df['BehaviourFile'].values
    ]

    # Use a regular expression to find the date at the beginning of each file name
    dates = [re.search(r'\d{1,2}_\d{1,2}_\d{4}', fname).group() for fname in filenames]
    assert len(dates) == len(filenames), 'Not all files have a date in the name (or some have several dates)'
    # Convert the dates to datetime objects
    dates = [datetime.strptime(date, '%d_%m_%Y') for date in dates]
    # Sort the files by date
    filenames = [f for _, f in sorted(zip(dates, filenames))]

    BhvBlockrec = []
    for fname in tqdm(filenames):
        matfile = sio.loadmat(path / fname,
                            struct_as_record=False,
                            squeeze_me=True)
        datadict = matstruct2dict(matfile['data'])
        if not datadict['recBlock'][0] == 'log':
            BhvBlockrec.append([fname, datadict['recBlock'][0]])

    if len(BhvBlockrec) > 0:
        # df = df.append(pd.DataFrame(BhvBlockrec, columns=['BehaviourFile', 'BlockRec']), ignore_index=True)
        df = pd.concat([df, pd.DataFrame(BhvBlockrec, columns=['BehaviourFile', 'BlockRec'])], ignore_index=True, sort=False)
    # df = df.append(pd.DataFrame(BhvBlockrec, columns=['BehaviourFile', 'BlockRec']))

    return df