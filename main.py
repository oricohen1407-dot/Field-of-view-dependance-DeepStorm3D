# main.py
# Run AutoDS3D steps without the GUI. Edit the PARAMS below and run any step.

# Steps to install:
# 1. create a virtual environment and activate it:
#     (python -m venv .venv; .\.venv\Scripts\activate on Windows, or source .venv/bin/activate on Linux/Mac)
# 2. pip install -r requirements.txt
# 3. python main.py

import os
from pathlib import Path
import pickle
import argparse
from pathlib import Path


# Avoid GUI backends on a headless server
os.environ.setdefault("MPLBACKEND", "Agg")

# --- Project paths and constants ---------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
from emitter_centers import (
    ZSTACK_FILE, CENTRAL_BEAD_COORDINATES_PIXEL, OFFAXIS_ZSTACK_FILES, OFFAXIS_COORDS_PIXEL, RAW_IMAGE_FOLDER
)

# --- Import pipeline functions ----------------------------------------------
from func_utils import (
    func1, func2, func3, func4, func5, func6_1, func6_2, func7
)

# --- Project paths -----------------------------------------------------------
os.chdir(PROJECT_DIR)                           # important: many functions use CWD
print("CWD:", Path.cwd())

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
        test_idx, threshold, state
    )

    state.setdefault("param_dict", {})
    state["param_dict"]["centralBeadCoordinates_pixel"] = centralBeadCoordinates_pixel
    state["param_dict"]["offaxis_zstack_files"] = offaxis_zstack_files
    state["param_dict"]["offaxis_coords_pixel"] = offaxis_coords_pixel
    state["param_dict"]["debug_max_emitters"] = len(offaxis_coords_pixel) + 1

    state["param_dict"]["mask_fit_save_dir"] = str(PROJECT_DIR / "mask_fit_outputs")

    if step == "psf":
        msg = func1(*args)
    elif step == "preproc":
        msg = func2(*args)
    elif step == "snr":
        msg = func3(*args)
    elif step == "td":  # training data
        msg = func4(*args)
    elif step == "train":
        #msg = func5(*args)
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
    startFromSNR = False  # only if "start from training" is false
    StartFromTraining = True
    import argparse

    if not run_all:
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf", "preproc", "snr", "td", "train", "test", "localize", "all"],
                       default="localize",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)

    else:
        if not StartFromTraining:
            if not startFromSNR:
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


            print(" ~~~~~~~~~~~~~~~~~~~~~~~ running snr stage ~~~~~~~~~~~~~~~~~~~~~~~~")
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
        else:
            print(" ~~~~~~~~~~~~~~~~~~~~~~ Starting from traing stage! ~~~~~~~~~~~~~~~~~~~~~~~~")


        print(" ~~~~~~~~~~~~~~~~~~~~~~ running train stage ~~~~~~~~~~~~~~~~~~~~~~~~")
        p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
        p.add_argument("--step",
                       choices=["psf","preproc","snr","td","train","test","localize","all"],
                       default="train",   # <-- default if omitted
                       help="Which pipeline step to run")
        args = p.parse_args()
        run_step(args.step)
        # '''
        try:
            print(" ~~~~~~~~~~~~~~~~~~~~~~ running test stage ~~~~~~~~~~~~~~~~~~~~~~~~")
            p = argparse.ArgumentParser(description="Debug AutoDS3D pipeline without GUI")
            p.add_argument("--step",
                           choices=["psf","preproc","snr","td","train","test","localize","all"],
                           default="test",   # <-- default if omitted
                           help="Which pipeline step to run")
            args = p.parse_args()
            run_step(args.step)

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



