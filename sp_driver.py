import pandas as pd
import numpy as np
import os
import yaml

from panelling import clustering



obj = clustering()
obj.process_data()

obj.elbow(4)
obj.get_clusters()

print('END')