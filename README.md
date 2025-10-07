
### neuropixels_chronic_spikesorting_Bizley

This package should let you take a set of many recordings and spikesort them in a way that can allow tracking 
of cells across sessions. It involves two main scripts: 

1: "get_drift_per_session", which basically runs the spikesorting
preprocessing, concatenates the results, and uses this concatenation to determine the session-to-session drift. It
outputs the average drift per spatial bin per session. This is later used in the unitmatch script to align cells properly.

2: "spikesort_folder", which iterates though folders and runs spikesorting on individual sessions.

After using each of these, you will be ready to use UnitMatch/Bombcell, which are in a seperate package because they 
require a different virtual environment.

## Installation

You need conda. I should explain how to do that. I don't want to half ass it though, which means I am currently no-assing it.
Next person who needs to do this, maybe we can work on it together and update this section...

After you get conda, you want to run:

conda env create --name neuropixels_chronic_spikesorting_Bizley -f environmentThatWorks.yml

Within that file, I have also said that, for whatever reason, you need to install kilosort seperately, and you should use:

pip install kilosort --upgrade

to do so. Notably, this will update to the most recent version of kilosort. At the time of this writing, the version 
I have installed and working is 4.0.7. The more distant we get from that, the less likely it is to work unless effort is
paid. Similarly, spikeinterface was 103.0 and may also break with distance. If versions ever give you problems, try doing ==0.103.0 
the spikeinterface line in the yml file...  If this doesn't work, contact me so I can update the proper version.

## config.py

ideally, this would be where many of the key file locations would be adjusted, and mostly we do a good job at this.
import config to get them. This will also include, for example, the tdt sample rate. 

## bug fixes for external packages

The best possible way to deal with them is to have your fixed function be called instead of theirs, and
keep that function in your own repository. I will try to do that, but in some cases the bugs will be too
low level for that to be reasonable. A somewhat tolerable alternative is to save the adjusted files in
a special folder and give direction (right here) on where to copy-paste them. So, that's the plan. The version
from which the adjustment was made will be specified by the folder we put it in. 

## list of files to move and where

    - .\neuropixels_chronic_spikesorting_Bizley\externalBugFixes\spike_interface\v_103.0\silence_periods.py

        -from your external packages folder, replace \spikeinterface\preprocessing\silence_periods.py

