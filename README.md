
### neuropixels_chronic_spikesorting_Bizley

This package should let you take a set of many recordings and spikesort them in a way that can allow tracking 
of cells across sessions. It involves two main scripts: 

1: "spikesort_folder", which iterates though folders and runs spikesorting on individual sessions.

2: "get_drift_per_session", which basically runs the spikesorting
preprocessing, concatenates the results, and uses this concatenation to determine the session-to-session drift. It
outputs the average drift per spatial bin per session. This is later used in the unitmatch script to align cells properly.

You only need the latter if you want to try and align across sessions. for spikesorting surveys, for example, there is no need. 

After using each of these, you will be ready to use UnitMatch/Bombcell, which are in a separate package because they 
require a different virtual environment. 

## Installation

You need conda. I should explain how to do that. I don't want to half ass it though, which means I am currently no-assing it.
Next person who needs to do this, maybe we can work on it together and update this section... But basically, just look it
up using a search engine.

After you get conda, you want to run:

conda env create --name neuropixels_chronic_spikesorting_Bizley -f environment.yml

alternatively, use your IDE to create the environment with the yml file. I use pycharm, and pycharm has bugs with
the command line which make using the gui more reliable. 

Within that file, I have also said that, for whatever reason, you need to install kilosort seperately, and you should use:

pip install kilosort --upgrade

to do so, or use your IDE package manager

After this, if you want to use a gpu, you need to do some stuff with torch:

(doesn't work yet)

pip uninstall torch (you for sure need to do this because you currently will have the cpu torch installed. Make sure
you unintall it from your virtual environment, and not from anywhere else

Then, you want to install pytorch with cuda. The best way to do this will be to follow directions on the website. Reportedly,
these days cuda is backward compatible, so you should just be able to use this link here for the correct comman line stuff:
https://pytorch.org/get-started/locally/
if not, you may want to look into previous versions, and to figure out which cuda version your gpu supports
by using the command line function nvidia-smi. I think there is another slightly more accurate function, but have lost
track of it...



## config.py

ideally, this would be where many of the key file locations would be adjusted, and mostly we do a good job at this.
import config to get them. This will also include, for example, the tdt sample rate. 

## catgt

catgt will be a part of this package. We can do this because the license they put it under allows that. The version I
put only has a windows and linux version; if there is ever insistence on mac, that person will need to find a mac version.


## bug fixes for external packages

The best possible way to deal with them is to have your fixed function be called instead of theirs, and
keep that function in your own repository. I will try to do that, but in some cases the bugs will be too
low level for that to be reasonable. A somewhat tolerable alternative is to save the adjusted files in
a special folder and give direction (right here) on where to copy-paste them. So, that's the plan. The version
from which the adjustment was made will be specified by the folder we put it in. 

## list of files to move and where (try without adjusting if you are using a more recent version of the relevant packages)
    (currently no need for any because spikeinterface fixed the silence_periods bug in the most recent version)
