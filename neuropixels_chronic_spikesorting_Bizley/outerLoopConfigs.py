from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



# all project-specific outer loop variables should be handled by this file.




matplotlib.use('TkAgg')
if True: # contains folder and file arguments, including higher-level ones like ferret name. Likely to be specific to my project


    #recordingZone = 'PFC_Boule_Borders_Top_GroundrefIThink'
    #recordingZone = 'PFC_Boule_Borders_Top'
    #recordingZone = 'PFC_shank0_Challah'
    #recordingZone = 'PFC_shank3_Challah'
    recordingZone = "ACx_Challah_top_groundref"
    frequencyOfConcatenation = 'do_everything' #'weekly_heuristic' or 'do_everything'. do_everything is contained per probemap.
    #output_folder = Path('D:/Jeffrey/Projects/SpeechAndNoise/Spikesorting_Output')
    output_folder = Path('C:/Jeffrey/InstrumentsProject/SessionDriftOutput')
    sessionsToDo = 'all'