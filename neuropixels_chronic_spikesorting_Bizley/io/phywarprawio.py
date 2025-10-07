from pathlib import Path
import numpy as np
import scipy.io as sio

from .phyrawio import PhyRawIO
from ..config import behaviouralDataPath
from ..helpers.extract_helpers import findBhvData


class PhyWarpRawIO(PhyRawIO):

    def __init__(self, dirname=''):
        PhyRawIO.__init__(self, dirname)

    def _parse_header(self):
        PhyRawIO._parse_header(self)

        phy_folder = Path(self.dirname)

        if (phy_folder.parents[0] / 'report/quality metrics.csv').is_file():
            self._quality_metrics = self._read_csv_as_dict(phy_folder.parents[0] / 'report/quality metrics.csv')
            if '' in self._quality_metrics:
                # 0 based indexing
                self._quality_metrics['cluster_id'] = self._quality_metrics.pop('')
        else:
            self._quality_metrics = None

        if (phy_folder.parents[0] / 'report/unit list.csv').is_file():
            self._peak_info = self._read_csv_as_dict(phy_folder.parents[0] / 'report/unit list.csv',
                                                    delimiter='\t')
        else:
            self._peak_info = None

        bl_ann = self.raw_annotations['blocks'][0]
        seg_ann = bl_ann['segments'][0]
        seg_ann['ferret'] = phy_folder.parents[2].name
        seg_ann['recording_session'] = phy_folder.parents[1].name

        behaviouralSession = findBhvData(seg_ann['recording_session'], behaviouralDataPath / seg_ann['ferret'])
        seg_ann['bhv_file'] = str(behaviouralDataPath / seg_ann['ferret'] / behaviouralSession)

        seg_ann['bhv_data'] = self.load_bhv_data(seg_ann['bhv_file']).to_dict(orient='list')

        # matfile = sio.loadmat(seg_ann['bhv_file'],
        #                     struct_as_record=False,
        #                     squeeze_me=True)

        # datadict = matstruct2dict(matfile['data'])
        self._trial_times = np.array(seg_ann['bhv_data']['startTrialLick'])

        cluster_info = [x for x in phy_folder.iterdir() if 
                        (x.name.startswith('cluster')) and 
                        (x.suffix == '.csv' or x.suffix == '.tsv')
                        # and not x.name=='cluster_KSLabel.tsv'
                        ]

        # annotation_lists is list of list of dict (python==3.8)
        # or list of list of ordered dict (python==3.6)
        # SEE: https://docs.python.org/3/library/csv.html#csv.DictReader
        annotation_lists = [self._parse_tsv_or_csv_to_list_of_dict(file)
                            for file in cluster_info]


        for index, clust_id in enumerate(self.clust_ids):
            spiketrain_an = seg_ann['spikes'][index]

            # Add cluster_id annotation
            spiketrain_an['cluster_id'] = clust_id

            # Loop over list of list of dict and annotate each st
            for annotation_list in annotation_lists:
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
            
            
            # Add peak info annotation
            if self._peak_info is not None:
                spiketrain_an['peak_info'] = [info for info in self._peak_info 
                    if int(info['unit_id'])==clust_id+1][0]
                # spiketrain_an['peak_info'] = self._extract_peak_info(clust_id + 1,
                #     self._peak_info)

            # Add quality annotation
            if self._quality_metrics is not None:
                # add +1 for cluster_id as it is 0-based
                spiketrain_an['quality_metrics'] = [qm for qm in self._quality_metrics
                    if int(qm[''])==clust_id+1][0]
                # spiketrain_an['quality_metrics'] = self._extract_cluster_metrics(clust_id + 1,
                #     self._quality_metrics)

                # quality = self._define_quality(spiketrain_an)

            spiketrain_an['quality'] = self._define_quality(spiketrain_an)

    def _extract_peak_info(self, cluster_id, peak_info):
        clus_ids = [float(i) for i in peak_info['unit_id']]
        idx = clus_ids==cluster_id
        clus_metrics = {}
        keys = [k for k in peak_info if k!='unit_id']
        for k in keys:
            try:
                clus_metrics[k] = np.array(peak_info[k],dtype=float)[idx]
            except:
                clus_metrics[k] = np.array(peak_info[k],dtype=str)[idx]

        return clus_metrics


    def _define_quality(self,
                        spiketrain_an,
                        quality_thresholds={'good': {'presence_ratio': (0.9, 'above')},
                                            'mua': {'presence_ratio': (0.9, 'above')}}):
        '''
        Define cluster quality based on quality metrics
        For more info see https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
        Params:
            quality thresholds: dict of dicts with keys 'good' and 'mua'
            ex: {'good': {'presence_ratio': (0.9, 'above'),
                        'isi_violations_ratio': (0.2, 'below')},
                'mua': {'presence_ratio': (0.9, 'above')}}

            parameters to define quality            

        '''

        quality_metrics = spiketrain_an['quality_metrics']

        for qual, thresholds in quality_thresholds.items():
            curr_quality = True
            for metric, (threshold, operator) in thresholds.items():
                if operator == 'above':
                    if float(quality_metrics[metric]) < threshold:
                        curr_quality=False
                        continue
                elif operator == 'below':
                    if float(quality_metrics[metric]) > threshold:
                        curr_quality=False
                        continue
                else:
                    raise ValueError('Operator not recognized')
            if curr_quality:
                return qual

        return 'noise'