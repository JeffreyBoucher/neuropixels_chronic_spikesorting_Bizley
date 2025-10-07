
"""
ExampleRawIO is a class of a  fake example.
This is to be used when coding a new RawIO.


Rules for creating a new class:
  1. Step 1: Create the main class
    * Create a file in **neo/rawio/** that endith with "rawio.py"
    * Create the class that inherits from BaseRawIO
    * copy/paste all methods that need to be implemented.
    * code hard! The main difficulty is `_parse_header()`.
      In short you have a create a mandatory dict than
      contains channel informations::

            self.header = {}
            self.header['nb_block'] = 2
            self.header['nb_segment'] = [2, 3]
            self.header['signal_streams'] = signal_streams
            self.header['signal_channels'] = signal_channels
            self.header['spike_channels'] = spike_channels
            self.header['event_channels'] = event_channels

  2. Step 2: RawIO test:
    * create a file in neo/rawio/tests with the same name with "test_" prefix
    * copy paste neo/rawio/tests/test_examplerawio.py and do the same

  3. Step 3 : Create the neo.io class with the wrapper
    * Create a file in neo/io/ that ends with "io.py"
    * Create a class that inherits both your RawIO class and BaseFromRaw class
    * copy/paste from neo/io/exampleio.py

  4.Step 4 : IO test
    * create a file in neo/test/iotest with the same previous name with "test_" prefix
    * copy/paste from neo/test/iotest/test_exampleio.py


PhyNpxlRawIO is a class to handle neuropixel spike sorted data with ecephys_spike_sorting .

"""

from neo.rawio.baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)

import numpy as np
from pathlib import Path
import re
import csv
import ast
from datetime import datetime, timedelta

from neuropixels_visualisation.helpers.extract_helpers import load_bhv_file

class PhyRawIO(BaseRawIO):
    """
    Class for reading Phy data.

    Usage:
        >>> import neo.rawio
        >>> r = neo.rawio.PhyNpxlRawIO(dirname='/dir/to/phy/folder')
        >>> r.parse_header()
        >>> print(r)
        >>> spike_timestamp = r.get_spike_timestamps(block_index=0,
        ... seg_index=0, spike_channel_index=0, t_start=None, t_stop=None)
        >>> spike_times = r.rescale_spike_timestamp(spike_timestamp, 'float64')

    """
    extensions = []
    rawmode = 'one-dir'

    def __init__(self, dirname=''):
        BaseRawIO.__init__(self)
        self.dirname = dirname

    def _source_name(self):
        return self.dirname

    def _parse_header(self):
        phy_folder = Path(self.dirname)
        
        # spike_times_in_sec=True

        # if (phy_folder / 'spike_times_sec_adj.npy').is_file():
        #     self._spike_times = np.load(phy_folder / 'spike_times_sec_adj.npy')
        #     self._spike_times = self._spike_times.reshape(len(self._spike_times), 1)
        # elif (phy_folder / 'spike_times_sec.npy').is_file():
        #     self._spike_times = np.load(phy_folder / 'spike_times_sec.npy')
        # else:
        self._spike_times = np.load(phy_folder / 'spike_times.npy')
        spike_times_in_sec=False

        self._spike_templates = np.load(phy_folder / 'spike_templates.npy')

        if (phy_folder / 'spike_clusters.npy').is_file():
            self._spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
        else:
            self._spike_clusters = self._spike_templates

        if len(self._spike_clusters.shape) == 1:
            self._spike_clusters = self._spike_clusters.reshape(len(self._spike_clusters), 1)

        if (phy_folder / 'mean_waveforms.npy').is_file():
            self._mean_waveform = np.load(phy_folder / 'mean_waveforms.npy')

        

        # TODO: Add this when array_annotations are ready
        if (phy_folder / 'amplitudes.npy').is_file():
            self._amplitudes = np.squeeze(np.load(phy_folder / 'amplitudes.npy'))
        else:
            self._amplitudes = np.ones(len(self._spike_times))
        
        if (phy_folder / 'pc_features.npy').is_file():
            self._pc_features = np.squeeze(np.load(phy_folder / 'pc_features.npy', mmap_mode='r'))
        else:
            self._pc_features = None

        if (phy_folder / 'channel_map.npy').is_file():
            self._channel_map = np.load(phy_folder / 'channel_map.npy')
        else:
            self._channel_map = None
        
        if (phy_folder / 'channel_positions.npy').is_file():
            self._channel_positions = np.load(phy_folder / 'channel_positions.npy')
        else:
            self._channel_positions = None

        # SEE: https://stackoverflow.com/questions/4388626/
        #  python-safe-eval-string-to-bool-int-float-none-string
        if (phy_folder / 'params.py').is_file():
            with (phy_folder / 'params.py').open('r') as f:
                contents = f.read()
            metadata = dict()
            contents = contents.replace('\n', ' ')
            pattern = re.compile(r'(\S*)[\s]?=[\s]?(\S*)')
            elements = pattern.findall(contents)
            for key, value in elements:
                metadata[key.lower()] = ast.literal_eval(value)

        self._sampling_frequency = metadata['sample_rate']

        if spike_times_in_sec:
            self._spike_times = self._spike_times * self._sampling_frequency

        clust_ids = np.unique(self._spike_clusters)
        self.unit_labels = list(clust_ids)
        self.clust_ids = clust_ids

        self._t_start = 0.
        self._t_stop = np.max(self._spike_times) / self._sampling_frequency

        signal_streams = []
        signal_streams = np.array(signal_streams, dtype=_signal_stream_dtype)

        signal_channels = []
        signal_channels = np.array(signal_channels,
                                   dtype=_signal_channel_dtype)

        spike_channels = []
        # Check this info and modify accordingly
        for i, clust_id in enumerate(clust_ids):
            unit_name = f'unit {clust_id}'
            unit_id = f'{clust_id}'
            wf_units = ''
            wf_gain = 1
            wf_offset = 0.
            wf_left_sweep = 0
            wf_sampling_rate = 0
            spike_channels.append((unit_name, unit_id, wf_units, wf_gain,
                                  wf_offset, wf_left_sweep, wf_sampling_rate))
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        event_channels = []
        # event_channels.append(('trials_epochs', 'ep_0','epoch'))

        event_channels.append(('trials_start', 'ev_0','event'))


        # event_channels.append(())

        # event_channels.append(('trials', 'ev_0','event'))
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [1]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()

        #csv_tsv_files = [x for x in phy_folder.iterdir() if
        #                 x.suffix == '.csv' or x.suffix == '.tsv']

        cluster_info = [x for x in phy_folder.iterdir() if 
                        (x.name.startswith('cluster')) and 
                        (x.suffix == '.csv' or x.suffix == '.tsv')
                        and (not x.name == 'cluster_info.tsv')
                        # and not x.name=='cluster_KSLabel.tsv'
                        ]

        # annotation_lists is list of list of dict (python==3.8)
        # or list of list of ordered dict (python==3.6)
        # SEE: https://docs.python.org/3/library/csv.html#csv.DictReader
        self.annotation_lists = [self._read_csv_as_dict(file)
                            for file in cluster_info]

        bl_ann = self.raw_annotations['blocks'][0]
        bl_ann['name'] = "Block #0"
        seg_ann = bl_ann['segments'][0]
        seg_ann['name'] = 'Seg #0 Block #0'
        seg_ann['channel_map'] = self._channel_map
        seg_ann['channel_positions'] = self._channel_positions
        seg_ann['sampling_frequency'] = self._sampling_frequency


    def _segment_t_start(self, block_index, seg_index):
        assert block_index == 0
        return self._t_start

    def _segment_t_stop(self, block_index, seg_index):
        assert block_index == 0
        return self._t_stop

    def _get_signal_size(self, block_index, seg_index, channel_indexes=None):
        return None

    def _get_signal_t_start(self, block_index, seg_index, channel_indexes):
        return None

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                channel_indexes):
        return None

    def _spike_count(self, block_index, seg_index, spike_channel_index):
        assert block_index == 0
        spikes = self._spike_clusters
        unit_label = self.unit_labels[spike_channel_index]
        mask = spikes == unit_label
        nb_spikes = np.sum(mask)
        return nb_spikes

    def _get_spike_timestamps(self, block_index, seg_index,
                              spike_channel_index, t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        unit_label = self.unit_labels[spike_channel_index]
        mask = self._spike_clusters == unit_label
        spike_timestamps = self._spike_times[mask.reshape(len(mask))]
        spike_timestamps = spike_timestamps.reshape(len(spike_timestamps))

        if t_start is not None:
            start_frame = int(t_start * self._sampling_frequency)
            spike_timestamps = \
                spike_timestamps[spike_timestamps >= start_frame]
        if t_stop is not None:
            end_frame = int(t_stop * self._sampling_frequency)
            spike_timestamps = spike_timestamps[spike_timestamps < end_frame]

        return spike_timestamps

    def _rescale_spike_timestamp(self, spike_timestamps, dtype):
        spike_times = spike_timestamps.astype(dtype)
        spike_times /= self._sampling_frequency
        return spike_times

    def _get_spike_raw_waveforms(self, block_index, seg_index,
                                 spike_channel_index, t_start, t_stop):
        return None

    def _event_count(self, block_index, seg_index, event_channel_index):
        return None

    def _get_event_timestamps(self, block_index, seg_index,
                              event_channel_index, t_start, t_stop):

        assert block_index == 0
        seg_t_start = self._segment_t_start(block_index, seg_index)
        if event_channel_index == 0:
            timestamp = self._trial_times + seg_t_start
            durations = None
            labels = None


        else:
            timestamp = np.array([0])
            durations = None
            labels = None

        if t_start is not None:
            keep = timestamp >= t_start
            timestamp = timestamp[keep]
            if labels is not None:
                labels = labels[keep]
            if durations is not None:
                durations = durations[keep]

        if t_stop is not None:
            keep = timestamp <= t_stop
            timestamp = timestamp[keep]
            if labels is not None:
                labels = labels[keep]
            if durations is not None:
                durations = durations[keep]

        return timestamp, durations, labels

    def _rescale_event_timestamp(self, event_timestamps, dtype, event_channel_index):
        # must rescale to second a particular event_timestamps
        # with a fixed dtype so the user can choose the precisino he want.

        # really easy here because in our case it is already seconds
        event_times = event_timestamps.astype(dtype)
        return event_times

    def _rescale_epoch_duration(self, raw_duration, dtype):
        return None

    def _get_behaviouralSessionTime(self,session_time,fileInfo,threshold=1):
        """
        Find the corresponding behavioural file based on time created
        Parameters:
            - session_time: str, creation time of the dataset, must be in the format
             yyyy-mm-ddTHH-MM-SS
            - fileInfo: dict, info relative to the behavioural files
            - threshold: float, maximum difference in creation time between session time and the
                returned behavioural file (in mn)
        """

        session_datetime = datetime.strptime(session_time, '%Y-%m-%dT%H:%M:%S')
        mask = (session_datetime >= fileInfo['dataDates'] - timedelta(minutes=threshold)) & \
            (session_datetime <= fileInfo['dataDates'] + timedelta(minutes=threshold))
        behaviouralSession  = fileInfo['dataString'][mask]
        assert behaviouralSession.size == 1, \
            f'WARNING {behaviouralSession.size} behavioural files found for {self.dirname}, check threshold'

        return behaviouralSession.tolist()[0]

    def _extract_cluster_metrics(self, cluster_id, metrics):
        clus_ids = [float(i) for i in metrics['cluster_id']]
        idx = clus_ids==cluster_id
        clus_metrics = {}
        keys = [k for k in metrics if k!='cluster_id']
        for k in keys:
            try:
                clus_metrics[k] = np.array(metrics[k],dtype=float)[idx]
            except:
                clus_metrics[k] = np.array(metrics[k],dtype=str)[idx]

        return clus_metrics


    def _define_quality(self, st_an, 
                        ISI_violations_trhs = 0.5,
                        amplitude_cutoff_trsh = 0.1,
                        presence_ratio_trsh = 0.9
                        ):
        '''
        Define cluster quality based on kilosort output and quality metrics
        If group = noise, then quality = noise
        Else, if KSLabel = mua, then quality = mua
        Else, a unit is good if: 
            - ISI_violations < ISI_violations_trhs
            - amplitude_cutoff < amplitude_cutoff_trsh
            - presence_ratio > 0.9

        For more info see https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
        '''
        if st_an['group']=='noise':
            quality = 'noise'

        # if st_an['kslabel'] == 'mua':
        #     quality = 'mua'

        elif 'quality_metrics' not in st_an.keys():
            quality = 'unsorted'

        elif (st_an['quality_metrics']['isi_viol'] <= ISI_violations_trhs) \
                & (st_an['quality_metrics']['amplitude_cutoff'] <= amplitude_cutoff_trsh) \
                & (st_an['quality_metrics']['presence_ratio'] >= presence_ratio_trsh):

            quality = 'good'

        else:
            quality = 'mua'

        return quality

    def load_bhv_data(self, bhv_file, fs=24414.062500):
        '''
        Load the behavioural session data from the file
        '''
        currData = load_bhv_file(bhv_file)
        currData['targTimes'] = currData['timeToTarget']/fs
        
        currData['centreRelease'] = currData['lickRelease']-currData['startTrialLick']
        currData['relReleaseTimes'] = currData['centreRelease']-currData['targTimes']
        currData['realRelReleaseTimes'] = currData['relReleaseTimes'] - currData['absentTime']
        currData['absoluteTargTimes'] = currData.startTrialLick + currData.targTimes
        currData['absoluteRealLickRelease'] = currData.lickRelease - currData.absentTime
        wordTime = np.zeros(len(currData), 10)
        wordType = np.zeros(len(currData), 10)
        for j in range(len(currData)):
            wordTime[j, 0] = currData['startTrialLick'][j]
            for i, word in enumerate(currData['distractor'][j]):
                if i < 10:
                    wordTime[j, i + 1] = wordTime[j, i] + currData['dDurs'][j][i]
                if word == '1':
                    wordType[j, i] = 1
                elif ((word == 10 or word == 26) and currData['talker'][j] == 1) or (word == 18 and currData['talker'][j] == 2):
                    wordType[j, i] = 2
                else:
                    wordType[j, i] = 3
        currData['wordTime'] = wordTime
        currData['wordType'] = wordType


        return currData        


    @staticmethod
    def _parse_tsv_or_csv_to_list_of_dict(filename):
        list_of_dict = list()
        letter_pattern = re.compile('[a-zA-Z]')
        float_pattern = re.compile(r'\d*\.')
        with open(filename) as csvfile:
            if filename.suffix == '.csv':
                reader = csv.DictReader(csvfile, delimiter=',')
            elif filename.suffix == '.tsv':
                reader = csv.DictReader(csvfile, delimiter='\t')
            else:
                raise ValueError("Function parses only .csv or .tsv files")
            line = 0

            for row in reader:
                if line == 0:
                    key1, key2 = tuple(row.keys())
                # Convert cluster ID to int
                row[key1] = int(row[key1])
                # Convert strings without letters
                if letter_pattern.match(row[key2]) is None:
                    if float_pattern.match(row[key2]) is None:
                        row[key2] = int(row[key2])
                    else:
                        row[key2] = float(row[key2])

                list_of_dict.append(row)
                line += 1

        return list_of_dict


    @staticmethod
    def _read_csv_as_dict(filename, delimiter=None):
        # csv_data = open(filename, 'r')
        if delimiter is None:
            if filename.suffix == '.csv':
                delimiter=','
            elif filename.suffix == '.tsv':
                delimiter='\t'
            else:
                raise ValueError("Function parses only .csv or .tsv files")

        with open(filename, 'r') as csv_data :
            data = list(csv.reader(csv_data, delimiter=delimiter))
            # get list of dict with keys as first row of csv
            keys = data[0]
            data = data[1:]
            list_of_dict = [dict(zip(keys, row)) for row in data]

            # ind_dict = dict(zip(data[0],get_mulList(*data[1:]))) 
        return list_of_dict


def get_mulList(*args):
    return map(list,zip(*args))