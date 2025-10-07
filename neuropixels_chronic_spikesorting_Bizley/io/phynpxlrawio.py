
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

import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import spikeinterface.full as si

from .phyrawio import PhyRawIO
# from .helpers_io import read_metadata
from .helpers_io import get_npix_sync
from ..config import behaviouralDataPath, neuropixelDataPath
from ..helpers.extract_helpers import updateFileInfo


class PhyNpxlRawIO(PhyRawIO):
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

    def __init__(self, 
                dirname='',
                curr_neural_data_path = neuropixelDataPath,
                sync_around_pulses = True,
                window_around_pulses = -2,
                ):
        """
        Initializes the class.
        Parameters:
            - dirname: str, path to the phy folder containing the spike sorted data
            - sync_around_pulses: bool, will align all the spike from window_around_pulses to the next pulse
            - window_around_pulses: list, window around trial onset to sync the spikes to (in seconds)
                Example: if window_around_pulses = 2
                spikes from i-th pulse time -2s to i+1-th time -2s pulse with be aligned to the i-th pulse
        """

        PhyRawIO.__init__(self, dirname)
        self.curr_neural_data_path = curr_neural_data_path
        self.sync_around_pulses = sync_around_pulses
        self.window_around_pulses = window_around_pulses
    
    def _source_name(self):
        return self.dirname

    def _parse_header(self): # this function is heavily non-robust. It depends on aspects being in specific positions.
        PhyRawIO._parse_header(self)
        
        phy_folder = Path(self.dirname)
        self._name = phy_folder.parents._parts[-4] # this was once -3. It led to the "name" being imec0.ap, which is incorrect, and moreover the code seems to have been written to not expect it. I need to rewrite it.

        ferret = [parent.name for parent in phy_folder.parents if parent.name.startswith('F')]
        assert len(ferret) == 1, \
            f'Could not find the ferret name in {phy_folder}, maybe two folders start with F?'
        self._ferret = ferret[0]

        if (phy_folder.parents[0] / 'report' / 'quality metrics.csv').is_file():
            self._quality_metrics = self._read_csv_as_dict(phy_folder.parents[0] / 'report' / 'quality metrics.csv')
            if '' in self._quality_metrics:
                # 0 based indexing
                self._quality_metrics['cluster_id'] = self._quality_metrics.pop('')
        else:
            self._quality_metrics = None

        if (phy_folder.parents[0] / 'report' / 'unit list.csv').is_file():
            self._peak_info = self._read_csv_as_dict(phy_folder.parents[0] / 'report' / 'unit list.csv',
                                                     delimiter='\t')
        else:
            self._peak_info = None

        if (phy_folder / 'cluster_info.tsv').is_file():
            self._cluster_info = self._read_csv_as_dict(phy_folder / 'cluster_info.tsv')
        else:
            self._cluster_info = None

        bl_ann = self.raw_annotations['blocks'][0]
        seg_ann = bl_ann['segments'][0]
        seg_ann['ferret'] = self._ferret
        seg_ann['channel_map'] = self._channel_map
        seg_ann['channel_positions'] = self._channel_positions
        seg_ann['sampling_frequency'] = self._sampling_frequency
        seg_ann['name'] = self._name

        rec_path = self.curr_neural_data_path / self._ferret / self._name

        seg_ann['rec_path'] = str(rec_path)

        if (rec_path / f'{rec_path.name[6:]}_tcat.nidq.XD_4_5_100.txt').exists():
            seg_ann['imec_pulse_time'] = np.genfromtxt(rec_path / f'{rec_path.name[6:]}_tcat.nidq.XD_4_5_100.txt')
        else:
            onsets, offsets = get_npix_sync(rec_path)
            seg_ann['imec_pulse_time'] = onsets[5]
            
        meta = self._load_metadata(next(rec_path.glob("*.meta")))
        fileInfo = updateFileInfo(behaviouralDataPath / seg_ann['ferret'])
        behaviouralSession = self._get_behaviouralSessionTime(
            # meta['highpass']['fileCreateTime_original'],
            meta['fileCreateTime'],
            fileInfo,
            threshold=3,
            )

        seg_ann['bhv_file'] = behaviouralSession

        seg_ann['bhv_data'] = self.load_bhv_data(behaviouralSession).to_dict(orient='list')
        seg_ann['trial_times'] = np.array(seg_ann['bhv_data']['startTrialLick'])
        self._trial_times = seg_ann['trial_times']
        seg_ann['tdt_pulse_time'] = np.array(seg_ann['bhv_data']['npPulseTime'])

        if seg_ann['imec_pulse_time'] is not None and len(seg_ann['imec_pulse_time']) != 0:
            # assert len(seg_ann['trial_times']) == len(seg_ann['imec_pulse_time']), \
            #     f'Number of trials in behavioural file ({len(seg_ann["trial_times"])}) does not match number of sync pulses data ({len(self._imec_pulse_time)})'

            tdt_pulse_time = seg_ann['tdt_pulse_time'].copy()
            imec_pulse_time = seg_ann['imec_pulse_time'].copy()

            # if len(seg_ann['tdt_pulse_time']) != len(seg_ann['imec_pulse_time']):
            #     print('debug')
            try:
                if len(seg_ann['tdt_pulse_time']) > len(seg_ann['imec_pulse_time']):
                    tdt_pulse_time = seg_ann['trial_times'][:len(seg_ann['imec_pulse_time'])]
                    imec_pulse_time = seg_ann['imec_pulse_time']
                elif len(seg_ann['tdt_pulse_time']) < len(seg_ann['imec_pulse_time']):
                    imec_pulse_time = seg_ann['imec_pulse_time'][:len(seg_ann['tdt_pulse_time'])]
                    tdt_pulse_time = seg_ann['tdt_pulse_time']

                # Just remove the last pulses if there is a mismatch and check if it works
                # If this is not the case, then we need to find a better way to match the pulses
                assert np.allclose(np.diff(imec_pulse_time), np.diff(tdt_pulse_time), atol=0.1), \
                    f'Behavioural file and sync pulses data do not match'
                
                seg_ann['tdt_pulse_time'] = tdt_pulse_time
                seg_ann['imec_pulse_time'] = imec_pulse_time
            except:
                if len(seg_ann['tdt_pulse_time']) > len(seg_ann['imec_pulse_time']):
                    tdt_pulse_time = seg_ann['trial_times'][
                        len(seg_ann['tdt_pulse_time']) - len(seg_ann['imec_pulse_time']):
                        ]
                    imec_pulse_time = seg_ann['imec_pulse_time']
                elif len(seg_ann['tdt_pulse_time']) < len(seg_ann['imec_pulse_time']):
                    imec_pulse_time = seg_ann['imec_pulse_time'][
                        len(seg_ann['imec_pulse_time']) - len(seg_ann['tdt_pulse_time']):
                        ]
                    tdt_pulse_time = seg_ann['tdt_pulse_time']

                assert np.allclose(np.diff(imec_pulse_time), np.diff(tdt_pulse_time), atol=0.2), \
                    f'Behavioural file and sync pulses data do not match (probably because of a missing pulse somewhere)'
                
                seg_ann['tdt_pulse_time'] = tdt_pulse_time
                seg_ann['imec_pulse_time'] = imec_pulse_time

        # Sync spikes times with behaviour time
        # pulse_diff = self._imec_pulse_time[0] - seg_ann['bhv_data']['npPulseTime'][0]
        # self._spike_times = self._spike_times - round(pulse_diff*self._sampling_frequency)

        if self._spike_times[0] < 0.:
            self._t_start = self._spike_times[0] / self._sampling_frequency
        # if self._spike_times[-1] > self._t_stop * self._sampling_frequency:
        #     self._t_stop = (self._spike_times[-1] + 1) / self._sampling_frequency
        last_pulse_diff = tdt_pulse_time[-1] - imec_pulse_time[-1]
        if (self._spike_times[-1] / self._sampling_frequency) + last_pulse_diff > self._t_stop:
            self._t_stop = ((self._spike_times[-1] + 1) / self._sampling_frequency) + last_pulse_diff
            

        for index, clust_id in enumerate(self.clust_ids):
            spiketrain_an = seg_ann['spikes'][index]

            # Add cluster_id annotation
            spiketrain_an['cluster_id'] = clust_id

            # Loop over list of list of dict and annotate each st
            for annotation_list in self.annotation_lists:
                clust_key, property_name = tuple(annotation_list[0].
                                                 keys())
                # if property_name == 'KSLabel':
                #     annotation_name = 'quality'
                # else:
                annotation_name = property_name.lower()
                for annotation_dict in annotation_list:
                    if int(annotation_dict[clust_key]) == clust_id:
                        spiketrain_an[annotation_name] = \
                            annotation_dict[property_name]
                        break

            si_clust_id = int(spiketrain_an['si_unit_id'])
            if self._cluster_info is not None:
                clust_info = [clu for clu in self._cluster_info if int(clu['cluster_id'])==clust_id][0]
                assert int(clust_info['cluster_id']) == clust_id, \
                    f'inconsistent {clust_id} (and clust_info: {clust_info["cluster_id"]}), please check...'

                for annotation in clust_info:
                    if annotation not in spiketrain_an:
                        spiketrain_an[annotation] = clust_info[annotation]

            # Add peak info annotation
            if self._peak_info is not None:
                clus_peak_info = [pinfo for pinfo in self._peak_info if int(pinfo['unit_id'])==si_clust_id][0]
                ann_dict = {}
                for annotation in clus_peak_info:
                    if annotation != 'unit_id':
                        ann_dict[annotation] = clus_peak_info[annotation]
                spiketrain_an['peak_info'] = ann_dict

            # Add quality annotation
            if self._quality_metrics is not None:
                clus_quality_metrics = [qmet for qmet in self._quality_metrics if int(qmet[''])==si_clust_id][0]
                ann_dict = {}
                for annotation in clus_quality_metrics:
                    if annotation != '':
                        ann_dict[annotation] = clus_quality_metrics[annotation]
                spiketrain_an['quality_metrics'] = ann_dict
            # spiketrain_an['mean_waveform'] = np.squeeze(
            #     self._mean_waveform[clust_id,:,:]
            # )

            # if self._waveform_metrics is not None:
            #     clus_waveform_metrics = [metric for metric in self._waveform_metrics if int(metric['cluster_id']) == clust_id][0]
            #     spiketrain_an['waveform_metrics'] = clus_waveform_metrics

            # if self._quality_metrics is not None:
            #     clus_quality_metrics = [metric for metric in self._quality_metrics if int(metric['cluster_id']) == clust_id][0]
            #     spiketrain_an['quality_metrics'] = clus_quality_metrics

            #     # spiketrain_an['quality_metrics'] = self._extract_cluster_metrics(clust_id,
            #     #     self._quality_metrics)

            #     quality = self._define_quality(spiketrain_an)


            # else:
                # if spiketrain_an['group']=='noise':
                #     quality = 'noise'
                # elif spiketrain_an['kslabel']=='mua':
                #     quality = 'mua'
                # else:
                #     quality = 'good'

            spiketrain_an['quality'] = spiketrain_an['group']

    def _get_spike_timestamps(self, block_index, seg_index,
                              spike_channel_index, t_start, t_stop):
        assert block_index == 0
        assert seg_index == 0

        unit_label = self.unit_labels[spike_channel_index]
        mask = self._spike_clusters == unit_label
        spike_timestamps = self._spike_times[mask.reshape(len(mask))]
        spike_timestamps = spike_timestamps.reshape(len(spike_timestamps))

        if self.sync_around_pulses:
            seg_ann = self.raw_annotations['blocks'][block_index]['segments'][seg_index]
            spike_timestamps = self._sync_spikes_around_pulses(spike_timestamps, seg_ann)

            # if spike_timestamps[0] / self._sampling_frequency < 0:
            #     self._t_start = spike_timestamps[0] / self._sampling_frequency
            # if spike_timestamps[-1] / self._sampling_frequency > self._t_stop:
            #     self._t_stop = (spike_timestamps[-1] + 1) / self._sampling_frequency

        if t_start is not None:
            start_frame = int(t_start * self._sampling_frequency)
            spike_timestamps = \
                spike_timestamps[spike_timestamps >= start_frame]
        if t_stop is not None:
            end_frame = int(t_stop * self._sampling_frequency)
            spike_timestamps = spike_timestamps[spike_timestamps < end_frame]

        return spike_timestamps


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

    def _sync_spikes_around_pulses(self, seg_spike_timestamps, seg_ann):
        # get pulse times
        pulse_times = seg_ann['imec_pulse_time']
        # get trial times
        tdt_pulse_times = seg_ann['tdt_pulse_time']

        # get spike times (in seconds)
        spike_times = seg_spike_timestamps / self._sampling_frequency



        # align spike time to behaviour time and keep only spikes around pulses
        spike_times_around_pulses = []
        for i, (pulse_time, tdt_pulse_time) in enumerate(zip(pulse_times, tdt_pulse_times)):
            sync_spike_times = spike_times - pulse_time + tdt_pulse_time

            # avoid overlapping windows between trials
            if (i < len(pulse_times) - 1):
                curr_window = [self.window_around_pulses, (pulse_times[i+1] - pulse_time) + self.window_around_pulses]

            # Deal with first and last trial
            if i==0:
                mask = (sync_spike_times >= 0) & (sync_spike_times <= tdt_pulse_time + curr_window[1])
            elif i==len(pulse_times)-1:
                mask = (sync_spike_times >= tdt_pulse_time + self.window_around_pulses)
            else:
                mask = (sync_spike_times >= tdt_pulse_time + curr_window[0]) & (sync_spike_times <= tdt_pulse_time + curr_window[1])
            
            spike_times_around_pulses.extend(sync_spike_times[mask])

        # Go back to samples
        spike_times_around_pulses = np.round(np.array(spike_times_around_pulses) * self._sampling_frequency)
        spike_times_around_pulses = spike_times_around_pulses.astype(int)

        return spike_times_around_pulses


    def _define_quality(self, st_an, 
                        ISI_violations_trhs = 0.5,
                        amplitude_cutoff_trsh = 0.1,
                        presence_ratio_trsh = 0.9
                        ):
        '''
        /!\ In progress, check manually the spikesorting output /!\
        Define cluster quality based on kilosort output and quality metrics
        If group = noise, then quality = noise
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

        elif (float(st_an['quality_metrics']['isi_viol']) <= ISI_violations_trhs) \
                & (float(st_an['quality_metrics']['amplitude_cutoff']) <= amplitude_cutoff_trsh) \
                & (float(st_an['quality_metrics']['presence_ratio']) >= presence_ratio_trsh):

            quality = 'good'

        else:
            quality = 'mua'

        return quality
    
    def _load_metadata(self, metafile):
        meta_glx = {}
        with open(metafile, 'r') as f:
            for ln in f.readlines():
                tmp = ln.split('=')
                k, val = tmp[0], ''.join(tmp[1:])
                k = k.strip()
                val = val.strip('\r\n')
                if '~' in k:
                    meta_glx[k] = val.strip('(').strip(')').split(')(')
                else:
                    try:  # is it numeric?
                        meta_glx[k] = float(val)
                    except:
                        meta_glx[k] = val
        return meta_glx