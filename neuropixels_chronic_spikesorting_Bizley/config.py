from pathlib import Path

#behaviouralDataPath = Path('Y:/Data/Behaviour/Ferret')
#neuropixelDataPath = Path('Z:/Data/Neuropixels')

#neuropixelDataPath = Path('C:/Jeffrey/Projects/SpeechAndNoise/PythonAnalyses/ShamDataHierarchy') # A local copy of one session which I have permission to edit. A temporary solution. I want to change this to allow me to access readonly on the NAS directly
neuropixelDataPath = Path('Z:/Data/Neuropixels')
catgtNeuropixelsDataPath = neuropixelDataPath # catgt is the compressed version, which I don't actually have. I need this here atm because it is asked for later, and presumably I will want to compress my neuropixels stuff eventually.
#behaviouralDataPath = Path('D:/Jeffrey/FakeBehaviorHierarchy')
behaviouralDataPath = Path('C:/Users/jeff/Dropbox/Data')
STIMULI_PATH = Path('C:/Jeffrey/Projects/SpeechAndNoise/stimuli')
bhv_fs = 24414.062500
np_fs = 30000 ### could alternatively be taken per meta file.


