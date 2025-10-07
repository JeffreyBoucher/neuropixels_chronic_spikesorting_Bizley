from neo.io.basefromrawio import BaseFromRaw

from .phynpxlrawio import PhyNpxlRawIO


class PhyNpxlIO(PhyNpxlRawIO, BaseFromRaw): # some kind of object that does IO for phy.
    name = 'Phy IO'
    description = "Phy IO"
    mode = 'dir'

    def __init__(self, 
                dirname, 
                sync_around_pulses = True,
                window_around_pulses = -2):
        PhyNpxlRawIO.__init__(self, 
                    dirname=dirname, 
                    sync_around_pulses=sync_around_pulses, 
                    window_around_pulses=window_around_pulses)
        BaseFromRaw.__init__(self, dirname)
