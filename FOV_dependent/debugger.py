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
ps_BFP          = 80           # um
external_mask   = "None"         # or absolute path to a .mat mask

# PSF retrieval inputs
#zstack_file     = str(PROJECT_DIR / "test2" / "scan_640nm_range_8um_step0.2um_mes_007_croped_range3.4.tif")
#zstack_file     = str(PROJECT_DIR / "2026_01_13_Mitochondria_flat/Beads_before/behind_objective" / "Beads_range_-2um_to_3um_x600_y562_pixels.tif")
#zstack_file = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_left_-2um_to_2um_x209_y513.tif")

# beads from 16/02/2026
 #Beads in air (on coverslip) for 21/01/2026
#inputs mes3 (main microtubuls)
zstack_file     = str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_center_x477_y573_2um.tif")
centralBeadCoordinates_pixel = [573, 477]  # optical axis reference ([y,x])

offaxis_zstack_files = [
    #str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_center_x477_y573_2um.tif"),
    str(PROJECT_DIR /"2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_bottomRight_x866_y765_2um.tif"),
    str(PROJECT_DIR /"2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_topRight_x851_y173_2um.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/"  / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_top_x364_y152_2um.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/"  / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_Left_x146_y417_2um.tif"),

    #str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_centerRight_x710_y618.tif"),  #around the centers
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_centerTop_x582_y358.tif"),
    #str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_centerBottomRight_x670_y780.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_027_centerBottom_x368_y826.tif"),
    str(PROJECT_DIR / "2026_02_16_microtubules/beads_shiftedFOV_afterExperiment/" / "behind_obj_exc_640nm_oil_x100_145_step0.2um_025_bottom_x682_y982.tif")  # bottom - area of interest
]

offaxis_coords_pixel = [
    #[573, 477],
    [765, 866],   # [r,c] in full camera coordinates
    [180, 851],
    [152, 364],
    [417, 146],

    #[618, 710],  # around the center
    [358, 582],
    #[780, 670],
    [826, 368],
    [982, 682]
]
'''
 # inputs mitochondria
zstack_file     = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif")
centralBeadCoordinates_pixel = [561, 647]  # optical axis reference ([y,x])

offaxis_zstack_files = [
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_bottom_-2um_to_2um_x295_y1027.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_top_-2um_to_2um_x815_y183.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_right_-2um_to_2um_x1048_y726.tif")
   ]

offaxis_coords_pixel = [
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
zstack_file     = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif")

offaxis_zstack_files = [
    #str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_right_-2um_to_2um_x1048_y726.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_left_-2um_to_2um_x209_y513.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_top_-2um_to_2um_x815_y183.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/"  / "BehindObj_001_bottom_-2um_to_2um_x295_y1027.tif"),
]

offaxis_coords_pixel = [
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
zstack_file     = str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif")

offaxis_zstack_files = [
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),
    str(PROJECT_DIR / "2026_01_21_Mitochondria/stack_before/" / "BehindObj_001_center__-2um_to_2um_x647_y561.tif"),

]

offaxis_coords_pixel = [
    #[561, 647],
    [726, 1048],   # [r,c] in full camera coordinates
    [513, 209],
    [183, 815],
    [1027, 295],
] 
 
'''
#centralBeadCoordinates_pixel = [651, 647]  # optical axis reference

# End  Ori's edit added on 2/01/2026 for mask displacement optimization


#nfp_text        = "-7, -3, 21"  # start,end,count (um) #  works good!
nfp_text        = "-7.5, -3.5, 21"  # start,end,count (um) # best!! works good!


#NFP = -5.5 + 1.5/1.33 # res7 and res 8. because there is a NFP shift in the system
#zrange = "-0.2, 2.8"   # res7 and res 8. 0.3 is "focus" when nfp bead [-7,3] and measured nfp is -3.9 res8 mes3

NFP = -5.5 + 2.0/1.33 # up to res15. because there is a NFP shift in the system
NFP = -5.5 + 1.6 #2.2/1.33 # res16
zrange = "0.0, 3.2"   # res7 and res 8. 0.3 is "focus" when nfp bead [-7,3] and measured nfp is -3.9 res8 mes3
#zrange = "0.3,1.1"

#zrange = "-0.7, 2.4"   # 0.3 is "focus" when nfp bead [-7,3] and measured nfp is -3.9 res8 mes3

# Raw data (blinking images)

#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes3/" / "behindObj_exc_640nm_oil_x100_145_026")  # mes3 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes2/" / "behind_obj_010")  # mes2 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_02_16_microtubules/mes4_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_044")  # mes4 of 16/02/2025 experiment

#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes3/" / "behind_obj_016 - Copy")  # mes3 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_01_21_Mitochondria/mes3/" / "behind_obj_018")  # mes3 of 21/01/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "2026_02_16_microtubules/mes3_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_026 - Copy")  # mes3 of 16/02/2025 experiment
raw_image_folder= str(PROJECT_DIR / "2026_02_16_microtubules/mes3_shiftedFOV/" /  "behindObj_exc_640nm_oil_x100_145_031")  # mes3 of 16/02/2025 experiment
#raw_image_folder= str(PROJECT_DIR / "training_data_res19/" /  "reconstructed_full_fov")  # mes3 of 16/02/2025 experiment




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



