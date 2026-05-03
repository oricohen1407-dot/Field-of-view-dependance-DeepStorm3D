import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from skimage import io
import gradio as gr
from func_utils import func1, func2, func3, func4, func5, func6_1, func6_2, func7
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from app_utils import (phase_retrieval, background_removal, mu_std_p, training_data_func, training_func,
                       inference_func1, inference_func2)
from app_utils import show_z_psf
import numpy as np
import torch
import os
import time
import scipy.io as sio
import pickle
from skimage import io

print('Entry point')
M = 100
NA = 1.45
n_immersion = 1.518
lamda = 0.67
n_sample = 18
f_4f = 200000
ps_camera = 11
ps_BFP = 40
external_mask = False
zstack_file = 'test2/scan_640nm_range_8um_step0.2um_mes_007_croped_range3.4.tif'
nfp_text = '-1.8, 1.6, 18'
NFP = 2.5
zrange = '0.5, 3.5'
raw_image_folder = 'test2/mask_off_005_exp50__illuminationx1_010'
snr_roi = '770, 770, 850, 850'
max_pv = 250
projection_01 = 0
num_z_voxel = 61
training_im_size = 61
us_factor = 1
max_num_particles = 20
num_training_images = 10000
previous_param_dict = 'None'
test_idx = 1000
threshold = 20
STATE_PICKLE = "/debug_state.pkl"

func4(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state)  # training data
