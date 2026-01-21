import pandas as pd
import numpy as np
from bubturb import *


reassign_ids_by_dec(df_path = '0503-good.tab', output_file_path = '0524-gs.tab')
compute_center_velocity('0524-gs.tab', '0524-gstp.tab')
compute_expansion_velocity('0524-gstp.tab', '0524-gstp.tab')
compute_radius_pc('0524-gstp.tab', '0524-gstp.tab')