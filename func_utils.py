
from app_utils import (
    phase_retrieval,
    show_z_psf,
    fit_mask_offset_from_offaxis_stacks
)
import numpy as np
import torch
import scipy.io as sio

def characterize_PSF(M, NA,  n_immersion, lamda, n_sample, f_4f, ps_camera, ps_BFP, external_mask,
          zstack_file, nfp_text, NFP, zrange, raw_image_folder, snr_roi, max_pv, projection_01,
          num_z_voxel, training_im_size, us_factor, max_num_particles, num_training_images, previous_param_dict, test_idx, threshold,
          state):
    # fetch param_dict
    if 'param_dict' not in state.keys():  # in the case of preprocessing images before characterizing PSF
        state['param_dict'] = dict()
    param_dict = state['param_dict']
    # update param_dict
    param_dict['M'] = M
    param_dict['NA'] = NA
    param_dict['lamda'] = lamda
    param_dict['n_immersion'] = n_immersion
    param_dict['n_sample'] = n_sample
    param_dict['f_4f'] = f_4f
    param_dict['ps_camera'] = ps_camera
    param_dict['ps_BFP'] = ps_BFP
    param_dict['NFP'] = NFP
    #param_dict['g_sigma'] = 1.1 # when bead where in air. ori's edit on 17/12/2025 original:0.6
    param_dict['g_sigma'] = 1.2 #1.4 #1.5 #1.2 # ori's edit on 17/12/2025 original:0.6

    param_dict["circ_scale"] = 5.3/5.8 # 5.3/5.8 # 5.4/5.8 #5.4/5.8  # example 26/01/2026

    # ori's edit
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
    print(f'device used (characterize_PSF): {device}')
    param_dict['device'] = device
    #end ori's edit
    #param_dict['device'] = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')

    # a dict for phase retrieval
    nfp_text = nfp_text.split(',')
    nfps = np.linspace(float(nfp_text[0]), float(nfp_text[1]), int(nfp_text[2]))
    param_dict["nfps"] = nfps

    pr_dict = dict(
        # zstack_file_path=os.path.join(os.getcwd(), zstack_file),
        zstack_file_path=zstack_file,
        nfps=nfps,
        r_bead=0.02,  # a default value, not critical
        epoch_num=250//1,  # optimization iterations
        epochs = 250//1,
        loss_label=1,  # 1: gauss log likelihood, 2: l2
        learning_rate=0.001,  # 0.005?

        # added on 15/03/2026:
        # bead-wise robust alignment
        fine_defocus_range_um=0.2,
        fine_defocus_step_um=0.1,
        max_shift_px=10,
    )
    if external_mask == 'None':


        phase_mask, g_sigma, ccs = phase_retrieval(param_dict, pr_dict)
        print(f'Phase mask is retrieved. blue sigma: {np.round(g_sigma, decimals=2)}.')
        print(f'PSF modeling accuracy: average cc of {np.round(np.mean(ccs), decimals=4)}.')

    else:
        mask_dict = sio.loadmat(external_mask)
        mask_name = list(mask_dict.keys())[3]
        phase_mask = mask_dict[mask_name]
        g_sigma = 0.6

    param_dict['g_sigma'] = (np.round(0.8*g_sigma, decimals=2), np.round(1.0*g_sigma, decimals=2))
    param_dict['phase_mask'] = phase_mask

    # Ori's edit added on 2/01/2026 for mask displacement optimization
    param_dict['nfps'] = nfps  # save z positions used for stacks

    # optional stage: fit mask distance using off-axis bead stacks
    #if param_dict.get('offaxis_zstack_files', None):
    #    from app_utils import fit_mask_offset_from_offaxis_stacks
    #    fit_mask_offset_from_offaxis_stacks(param_dict)


    #param_dict['centralBeadCoordinates_pixel'] = [r_center, c_center]  # optical axis reference (the one used for phase retrieval)
    param_dict['centralBeadCoordinates_pixel'] = param_dict.get('centralBeadCoordinates_pixel', [600, 600])

    #param_dict['offaxis_zstack_files'] = [file1, file2, ...]
    #param_dict['offaxis_coords_pixel'] = [[r1, c1], [r2, c2], ...]  # global camera coords of each ROI center
    # Off-axis stacks are optional. If provided (e.g. by debugger via state['param_dict']),
    # they will be used to fit mask offset.
    param_dict.setdefault('offaxis_zstack_files', [])
    param_dict.setdefault('offaxis_coords_pixel', [])
    param_dict.setdefault('centralBeadCoordinates_pixel', [600, 600])

    param_dict['nfps'] = nfps
    # End  Ori's edit added on 2/01/2026 for mask displacement optimization

    # show z-PSF regarding the NFP
    param_dict['NFP'] = NFP  # now it's not bead
    zrange = zrange.split(',')
    zrange = (float(zrange[0]), float(zrange[1]))
    param_dict['zrange'] = zrange  # a tuple
    param_dict['baseline'] =  None   # required by imaging model, None for now
    param_dict['read_std'] =  None
    param_dict['bg'] = None
    show_z_psf(param_dict)  # generate PSFs.jpg
    param_dict['non_uniform_noise_flag'] = True
    param_dict['bitdepth'] = 16


    save_dir = param_dict.get("mask_fit_save_dir", None)
    NFP_exp = param_dict["NFP"]
    param_dict["NFP"] = 0.0
    fit_mask_offset_from_offaxis_stacks(param_dict, save_dir=save_dir)
    param_dict["NFP"] = NFP_exp
    # save param_dict for other blocks
    if 'param_dict' in state.keys():
        state['param_dict'] = {**state['param_dict'], **param_dict}
    else:
        state['param_dict'] = param_dict

    return ("PSF characterization is done. Check "
            "\nphase_retrieval_results.jpg "
            "\nPSFs.jpg")
