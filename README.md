
### neuropixels_chronic_spikesorting_Bizley

This package should let you take a set of many recordings and spikesort them in a way that can allow tracking 
of cells across sessions. It involves two main scripts: 

1: "get_drift_per_session", which basically runs the spikesorting
preprocessing, concatenates the results, and uses this concatenation to determine the session-to-session drift. It
outputs the average drift per spatial bin per session. This is later used in the unitmatch script to align cells properly.

2: "spikesort_folder", which iterates though folders and runs spikesorting on individual sessions.

After using each of these, you will be ready to use UnitMatch/Bombcell, which are in a seperate package because they 
require a different virtual environment.



In order to get my spikesorting to work, I needed to manually edit some of spikeinterface's code... I still need to figure
out how to provide these changes (which are good) but also allow for potentially updating si. I should update this section
of the readme when I figure out what I am doing here; yell at me (Jeff) if you ever read this.


## Installation

You need conda. I should explain how to do that. 

After you get conda, I believe you want to run:

conda env create --name neuropixels_chronic_spikesorting_Bizley -f environmentThatWorks.yml

Within that file, I have also said that, for whatever reason, you need to install kilosort seperately, and you should use:

pip install kilosort --upgrade

to do so. Notably, this will update to the most recent version of kilosort. At the time of this writing, the version 
I have installed and working is 4.0.7. The more distant we get from that, the less likely it is to work unless effort is
paid.

Similarly, the spikeinterface version may cause problems. The one I got working most recently was 102.3, but I know I
will want to update this and so am hoping to get things working on the most recent version, 103.0, soon. At any point things
may break though, this will be hard to keep stable. If versions ever give you problems, try doing ==0.103.0 after the spikeinterface
line in the yml file... This assumes I will have gotten 103 working. If this doesn't work, contact me so I can update the proper version.

## config.py

ideally, this would be where many of the key file locations would be adjusted, and mostly we do a good job at this.
import config to get them. This will also include, for example, the tdt sample rate. 


