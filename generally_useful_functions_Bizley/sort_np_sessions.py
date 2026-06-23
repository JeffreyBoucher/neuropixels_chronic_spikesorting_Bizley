
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import shutil

from generally_useful_functions_Bizley.load_meta_file import load_meta_file

def sort_np_sessions(sessions_list,minimum_duration_s=-1):
    """
    Sorts a list of Neurophysiology (NP) session directories based on the file creation time of their metadata files.

    Parameters:
    -----------
    sessions_list : list of pathlib.Path objects or list of str
        A list of pathlib.Path objects or a list of strings representing the directories of NP sessions.
    minimum_duration_s : int, optional
        The minimum duration (in seconds) of sessions to be included in the sorted list. Defaults to -1, which includes all sessions.

    Returns:
    --------
    numpy.ndarray
        A 1-dimensional array of pathlib.Path objects representing the directories of NP sessions, sorted in ascending order
        of their file creation times.
    """

    if isinstance(sessions_list[0], str):
        sessions_list = [Path(s) for s in sessions_list]

    meta_dicts = []
    for session in sessions_list:
        metafile = [f for f in session.glob('*.meta')][0]
        meta = load_meta_file(metafile)
        meta['session_name'] = session
        meta_dicts.append(meta)

    df_meta = pd.DataFrame.from_dict(meta_dicts)
    df_meta['fileCreateTime'] = pd.to_datetime(df_meta['fileCreateTime'])
    df_meta = df_meta.sort_values('fileCreateTime', ignore_index=True)

    df_meta = df_meta.loc[df_meta.fileTimeSecs > minimum_duration_s]

    return df_meta.session_name.to_numpy()