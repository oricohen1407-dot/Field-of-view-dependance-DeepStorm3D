# debug_pipeline.py
# Run AutoDS3D steps without the GUI. Edit the PARAMS below and run any step.
# Use your env2 interpreter. From project root:
#   /bigdata/ori_cohen/anaconda3/envs/env2/bin/python debug_pipeline.py --step psf

# debugger.py  (place this file in your repo root)
import os, sys
from pathlib import Path
import pickle
import argparse
from pathlib import Path



# Avoid GUI backends on a headless server
os.environ.setdefault("MPLBACKEND", "Agg")

# --- Project paths -----------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent   # put this file in the repo root
os.chdir(PROJECT_DIR)                           # important: many functions use CWD
print("CWD:", Path.cwd())

# --- Import pipeline functions ----------------------------------------------
from func_utils import (
    func1, func2, func3, func4, func5, func6_1, func6_2, func7
)

# --- Fixed parameters (EDIT THESE ONCE) -------------------------------------
# Optical + acquisition
M               = 100
NA              = 1.45
n_immersion     = 1.518
lamda           = 0.67           # um (emission)
n_sample        = 1.33
f_4f            = 200000        # um
ps_camera       = 11             # um
ps_BFP          = 80             # um
external_mask   = "None"         # or absolute path to a .mat mask
#camera_size_px = 1400, 1400  # number of piixels in camera H,W

# PSF retrieval inputs
zstack_file     = str(PROJECT_DIR / "test2" / "scan_640nm_range_8um_step0.2um_mes_007_croped_range3.4.tif")

#zstack_file     = str(PROJECT_DIR / "test2" / "scan_640nm_range_8um_step0.2um_mes_007_flipped_croped_range_3.6.tif")

#nfp_text        = "-1.8, 1.6, 18"  # start,end,count (um)
nfp_text        = "1.8, -1.6, 18"  # start,end,count (um) "flipped bead zstack"

#NFP             = 2.5               # um
NFP             = 1.8*1               # um
#zrange          = "0.4, 2.4"            # expected z-range (um)
zrange          = "-0.4, 3.4"            # expected z-range (um)
#zrange          = "0.5, 3.5"            # expected z-range (um)
#zrange          = "0.2, 3.6"            # expected z-range (um)
#zrange = "0.05 , 0.15"
#zrange = "3.65 , 3.65"
#zrange = "1.4 , 1.4"  #focus when NFP is 1.8
#zrange = "0.4 , 0.4"
#zrange = "-0.4 , -0.4"
#zrange = "2.4 , 2.4"

#NFP             = 1.8*0               # um
#zrange          = "-1.0, 1.0"            # expected z-range (um) "-1.1,1.1"
#zrange          = "-1.4, 1.6"            # expected z-range (um) "-1.1,1.1"
#zrange = "1.4,1.45"
#zrange = "0,0.05"
# Raw data (blinking images)
raw_image_folder= str(PROJECT_DIR / "test2" / "mask_off_005_exp50__illuminationx1_010_2")  # folder with frames (mask_off_005_exp50__illuminationx1_010_2_1000_images for faster experiment)
snr_roi         = "520, 450, 610, 650"     # r0,c0,r1,c1 (pixels)
max_pv          = 40 *1        # camera saturation-ish  #30
projection_01   = 0                  # 0 = no 0-1 projection, 1 = yes
#centralBeadCoordinates_pixel = [849, 854] # [r, c] ori's edit defined in ds3d_utils
# Training data & training
num_z_voxel     = 81
training_im_size= 1400 #/8 # 121
us_factor       = 1  # up-scaling factor
max_num_particles = 35*50
num_training_images = 400 # 500//50
test_idx        = 1000
threshold       = 15*1 #30

mask_offset_in_um = 40000  * 0#38000 need to modify inside the function! this doesn't do anything

# Optional: reuse a previous param_dict pickle produced by training (func5)
previous_param_dict = "None"   # or e.g. "param_dict_01-23_17-02.pickle"

# Where we cache GUI-like state between steps
STATE_PICKLE = PROJECT_DIR / ".debug_state.pkl"

def load_state():
    if STATE_PICKLE.exists():
        with open(STATE_PICKLE, "rb") as f:
            return pickle.load(f)
    return {"param_dict": {}}

def save_state(state):
    with open(STATE_PICKLE, "wb") as f:
        pickle.dump(state, f)

def run_step(step):
    state = load_state()

    args = (
        M, NA, n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
        zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
        num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images,
        previous_param_dict, test_idx, threshold, state
    )

    if step == "psf":
        msg = func1(*args)
    elif step == "preproc":
        msg = func2(*args)
    elif step == "snr":
        msg = func3(*args)
    elif step == "td":  # training data
        msg = func4(*args)
    elif step == "train":
        msg = func5(*args)
    elif step == "test":
        msg = func6_1(*args)
    elif step == "localize":
        msg = func6_2(*args)
    elif step == "all":
        msg = func7(*args)
    else:
        raise ValueError("Unknown step")

    # Persist state (so the next step sees updated param_dict)
    save_state(state)
    print("\n=== STEP OUTPUT ===\n", msg)

if __name__ == "__main__":
    run_all = True

    import argparse
    if not run_all:
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="test",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)

    else:

        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        print(" ~~~~~~~~~~~~~~~~~~~~~~ running psf stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="psf",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)


        print(" ~~~~~~~~~~~~~~~~~~~~~~ running preproc stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="preproc",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)

        print(" ~~~~~~~~~~~~~~~~~~~~~~ running snr stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="snr",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)

        print(" ~~~~~~~~~~~~~~~~~~~~~~ running training data (td) stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="td",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)

        print(" ~~~~~~~~~~~~~~~~~~~~~~ running train stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="train",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)

        try:
            print(" ~~~~~~~~~~~~~~~~~~~~~~ running test stage ~~~~~~~~~~~~~~~~~~~~~~~~")
            p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
            p.add_argument("--step",
                           choices=["psf","preproc","snr","td","train","test","localize","all"],
                           default="test",   # <-- default if omitted
                           help="Which pipeline step to run")
            args = p.parse_args()
            run_step(args.step)

            '''
            print(" ~~~~~~~~~~~~~~~~~~~~~~ running localization stage ~~~~~~~~~~~~~~~~~~~~~~~~")
            p.add_argument("--step",
                           choices=["psf","preproc","snr","td","train","test","localize","all"],
                           default="localize",   # <-- default if omitted
                           help="Which pipeline step to run")
            args = p.parse_args()
            run_step(args.step)
            '''
        except:
                  print('failed to run test. running localization')

        import time

        print(" ~~~~~~~~~~~~~~~~~~~~~~ running localization stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        iter = 0
        tryLoopsNum = 3
        TryLoopsFlag = True
        while iter<tryLoopsNum and TryLoopsFlag:
            p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
            iter += 1
            TryLoopsFlag = False
            try:
                print("Execution paused for 5 seconds.")
                time.sleep(5)  # Pause for 5 seconds
                print("Execution resumed after 5 seconds.")
                print(" try {" + str(iter) +"} to execute localization")

                p.add_argument("--step",
                               choices=["psf","preproc","snr","td","train","test","localize","all"],
                               default="localize",   # <-- default if omitted
                               help="Which pipeline step to run")
                args = p.parse_args()
                run_step(args.step)

            except:
                TryLoopsFlag = True



