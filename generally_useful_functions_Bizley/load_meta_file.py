# Function adapted from NeuroPyxels package (https://github.com/m-beau/NeuroPyxels)
# To read metadata from neuropixels recordings

from pathlib import Path
import os
import numpy as np

def load_meta_file(metafile):
    ''' Read a .meta file from spikeglx and return a dictionnary of its content'''
    # load metafile
    meta = {}
    with open(metafile, 'r') as f:
        for ln in f.readlines():
            tmp = ln.split('=')
            k, val = tmp[0], ''.join(tmp[1:])
            k = k.strip()
            val = val.strip('\r\n')
            if '~' in k:
                meta[k] = val.strip('(').strip(')').split(')(')
            else:
                try:  # is it numeric?
                    meta[k] = float(val)
                except:
                    meta[k] = val
    return meta