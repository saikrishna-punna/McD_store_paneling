import pandas as pd
import numpy as np
import os
import sp_utils as ut

def dummy_fe():
    path = r'D:\McDonald\refactoring\StorePanelling\Russia\input\sample_fe'
    files = os.listdir(path)
    df = pd.concat([ut.load_flat_file(os.path.join(path, file_)) for file_ in files])
    return df