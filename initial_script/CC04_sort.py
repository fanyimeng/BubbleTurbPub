import pandas as pd
import numpy as np
from bubturb import *


# reassign_ids_by_dec(df_path = '0610-tp.tab', output_file_path = '0610-gstp.tab')
reassign_ids_by_dec(df_path = '0705.tab', output_file_path = '0705-s.tab', sort_col='r_kpc')
