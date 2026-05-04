# This file contains paths to multiple experiment records
# main.py takes these constants from here -
# ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL, RAW_IMAGE_FOLDER
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent



# PSF retrieval inputs

# beads from 16/02/2026
 # Beads in air (on coverslip) for 21/01/2026
# inputs mes3 (main microtubuls) - behind_obj_exc_640nm_oil_x100_145_step0.2um_027
ZSTACK_FILE = str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "center_x477_y573_2um.tif")
CENTRAL_BEAD_COORDINATES_PIXEL = [573, 477]  # optical axis reference ([y,x])

OFFAXIS_ZSTACK_FILES = [
    # str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "center_x477_y573_2um.tif"),
    str(PROJECT_DIR /"2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "bottomRight_x866_y765_2um.tif"),
    str(PROJECT_DIR /"2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "topRight_x851_y173_2um.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/"  / "top_x364_y152_2um.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/"  / "Left_x146_y417_2um.tif"),

    # str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "centerRight_x710_y618.tif"),  #around the centers
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "centerTop_x582_y358.tif"),
    # str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "centerBottomRight_x670_y780.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "centerBottom_x368_y826.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "bottom_x682_y982.tif")  # bottom - area of interest so 2um_025
]

OFFAXIS_COORDS_PIXEL = [
    # [573, 477],
    [765, 866],   # [r,c] in full camera coordinates
    [180, 851],
    [152, 364],
    [417, 146],

    # [618, 710],  # around the center
    [358, 582],
    # [780, 670],
    [826, 368],
    [982, 682]
]

# Raw data (blinking images)

#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes3/" / "behindObj_exc_640nm_oil_x100_145_026")  # mes3 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes2/" / "behind_obj_010")  # mes2 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_02_16_microtubules/mes4_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_044")  # mes4 of 16/02/2025 experiment

#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes3/" / "behind_obj_016 - Copy")  # mes3 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes3/" / "behind_obj_018")  # mes3 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_02_16_microtubules/mes3_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_026 - Copy")  # mes3 of 16/02/2025 experiment
RAW_IMAGE_FOLDER = str(PROJECT_DIR / "2026_02_16_microtubules/mes3_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_031")  # mes3 of 16/02/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "training_data_res19/" /  "reconstructed_full_fov")  # mes3 of 16/02/2025 experiment

'''
 # inputs mitochondria
ZSTACK_FILE     = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif")
CENTRAL_BEAD_COORDINATES_PIXEL = [561, 647]  # optical axis reference ([y,x])

OFFAXIS_ZSTACK_FILES = [
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_bottom_-2um_to_2um_x295_y1027.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_top_-2um_to_2um_x815_y183.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_right_-2um_to_2um_x1048_y726.tif")
   ]

OFFAXIS_COORDS_PIXEL = [
    #[573, 477],
    [1027, 295],   # [r,c] in full camera coordinates
    [183, 815],
    [726, 1048]
]

'''
# Ori's edit added on 2/01/2026 for mask displacement optimization

# beads in water #2
'''
 #Beads in air (on coverslip) for 21/01/2026
ZSTACK_FILE     = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif")

OFFAXIS_ZSTACK_FILES = [
    #str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_right_-2um_to_2um_x1048_y726.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_left_-2um_to_2um_x209_y513.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_top_-2um_to_2um_x815_y183.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_bottom_-2um_to_2um_x295_y1027.tif"),
]

OFFAXIS_COORDS_PIXEL = [
    #[561, 647],
    [726, 1048],   # [r,c] in full camera coordinates
    [513, 209],
    [183, 815],
    [1027, 295],
]
'''
'''
# for mask design:

 #Beads in air (on coverslip)
ZSTACK_FILE     = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif")

OFFAXIS_ZSTACK_FILES = [
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),

]

OFFAXIS_COORDS_PIXEL = [
    #[561, 647],
    [726, 1048],   # [r,c] in full camera coordinates
    [513, 209],
    [183, 815],
    [1027, 295],
] 

CENTRAL_BEAD_COORDINATES_PIXEL = [651, 647] # optical axis reference ([y,x])
'''

# End  Ori's edit added on 2/01/2026 for mask displacement optimization
