import pandas as pd
import numpy as np
from bubturb import *


# reg2df(fits_image_path = '../data/jcomb_vcube_scaled2_mom0.fits', 
# 	ds9_reg_file_path='0328-final_xy.reg', 
# 	output_file_path='0328-final_xy.tab')

# normalize_ellipses('0328-final_xy.tab','0328-final_xy_panorm.tab')

# add_ra_dec_strings(df_path = '0328-final_xy_panorm.tab', 
# 	fits_path = '../data/jcomb_vcube_scaled2_mom0.fits', 
# 	output_file_path = '0328-final_xy_panorm_hmsdms.tab')

# major_minor_2arcsec(df_path = '0328-final_xy_panorm_hmsdms.tab', 
# 	output_file_path = '0328-final_xy_panorm_hmsdms_as.tab',
# 	factor = 5)


# pandas2ds9('0328-final_xy_panorm_hmsdms_as.tab',
# 	'0328-final_xy_panorm_hmsdms_as.reg')

reassign_ids_by_dec(df_path = '0328-final_with_collision_fields.tab', output_file_path = '0407-sorted.tab')
reassign_ids_by_dec(df_path = '0407-sorted.tab', output_file_path = '0407-sorted.tab')
# process_hi_hole_with_ellipses("b86/table2.dat","b86_J2000.tab",reg_output_path="b86_J2000.reg")

# # 用法示例
# process_hi_hole_table(
#     input_path="b86/table2.dat",
#     output_path="b86_J2000.tab"
# )