# main.py
# Run AutoDS3D steps without the GUI. Edit the PARAMS below and run any step.

# Steps to install:
# 1. create a virtual environment and activate it:
#     (python -m venv .venv; .\.venv\Scripts\activate on Windows, or source .venv/bin/activate on Linux/Mac)
# 2. pip install -r requirements.txt
# 3. python main.py

import os
from pathlib import Path
# --- Project paths and constants ---------------------------------------------
from emitter_centers import (
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL, RAW_IMAGE_FOLDER
)
from func_utils import characterize_PSF

# TODO (RK): Delete and remove import if unnecessary
# Avoid GUI backends on a headless server
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_DIR = Path(__file__).resolve().parent

# --- Fixed parameters (EDIT THESE ONCE) -------------------------------------
# Optical + acquisition
M               = 100
NA              = 1.45
n_immersion     = 1.518
lamda           = 0.67           # um (emission)
n_sample        = 1.33
f_4f            = 200000        # um
ps_camera       = 11             # um
ps_BFP          = 80           # um
external_mask   = "None"         # or absolute path to a .mat mask

#nfp_text        = "-7, -3, 21"  # start,end,count (um) #  works good!
nfp_text        = "-7.5, -3.5, 21"  # start,end,count (um) # best!! works good!

#NFP = -5.5 + 1.5/1.33 # res7 and res 8. because there is a NFP shift in the system
#zrange = "-0.2, 2.8"   # res7 and res 8. 0.3 is "focus" when nfp bead [-7,3] and measured nfp is -3.9 res8 mes3

NFP = -5.5 + 2.0/1.33 # up to res15. because there is a NFP shift in the system
NFP = -5.5 + 1.6 #2.2/1.33 # res16
zrange = "0.0, 3.2"   # res7 and res 8. 0.3 is "focus" when nfp bead [-7,3] and measured nfp is -3.9 res8 mes3
#zrange = "0.3,1.1"
#zrange = "-0.7, 2.4"   # 0.3 is "focus" when nfp bead [-7,3] and measured nfp is -3.9 res8 mes3


snr_roi         = "550, 550, 650, 650"     # r0,c0,r1,c1 (pixels)
max_pv          = 100 #80     # camera saturation-ish  #30
projection_01   = 0                  # 0 = no 0-1 projection, 1 = yes
#centralBeadCoordinates_pixel = [849, 854] # [r, c] ori's edit defined in ds3d_utils
# Training data & training
num_z_voxel     = 161 #81
training_im_size= 1200 #/8 # 121
us_factor       = 1  # up-scaling factor
max_num_particles = 25*(8*8)  # 35*(8*8)?
num_training_images = 5000 #400 #400 # 500//50
test_idx        = 1000
threshold       = 20 #20 #30

# TODO (RK): Add a "save conf to file support"

def run_characterize_PSF():
    state = {"param_dict": {}}
    state["param_dict"]["centralBeadCoordinates_pixel"] = CENTRAL_BEAD_COORDINATES_PIXEL
    state["param_dict"]["offaxis_zstack_files"] = OFFAXIS_ZSTACK_FILES
    state["param_dict"]["offaxis_coords_pixel"] = OFFAXIS_COORDS_PIXEL
    state["param_dict"]["debug_max_emitters"] = len(OFFAXIS_COORDS_PIXEL) + 1

    state["param_dict"]["mask_fit_save_dir"] = str(PROJECT_DIR / "mask_fit_outputs")

    msg = characterize_PSF(M, NA, n_immersion, lamda, n_sample, f_4f, ps_camera,
                            ps_BFP, external_mask, ZSTACK_FILE, nfp_text, NFP,
                            zrange, state)
    print("\n=== STEP OUTPUT ===\n", msg)

if __name__ == "__main__":
    run_characterize_PSF()
