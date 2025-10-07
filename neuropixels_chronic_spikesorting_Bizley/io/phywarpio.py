from neo.io.basefromrawio import BaseFromRaw

from .phywarprawio import PhyWarpRawIO


class PhyWarpIO(PhyWarpRawIO, BaseFromRaw):
    name = 'Phy IO'
    description = "Phy IO"
    mode = 'dir'

    def __init__(self, dirname):
        PhyWarpRawIO.__init__(self, dirname=dirname)
        BaseFromRaw.__init__(self, dirname)
