
import numpy as np
from numpy import pi
from skimage import io
import csv
import time
import os
import shutil
import pickle
from scipy import ndimage
from datetime import datetime
import torch
import torch.fft as fft
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DS3Dplus.ds3d_utils import ImModel, Sampling, MyDataset, Volume2XYZ, calc_jaccard_rmse, KDE_loss3D, ImModelBead, ImModelBase, ImModelTraining
from DS3Dplus.ds3d_utils import LON as Net
from DS3Dplus.training_utils import TorchTrainer
import matplotlib.pyplot as plt
from DS3Dplus.ds3d_utils import asm_propagate

# Ori's edit added on 2/01/2026 for mask displacement optimization
import numpy as np
import os
import csv
import numpy as np
from datetime import datetime
from skimage import io
import matplotlib.pyplot as plt
import math

# cocorellation loss addition 29/01/2026
import torch
import torch
from scipy import ndimage
import numpy as np

from skimage.registration import phase_cross_correlation
from scipy import ndimage


from torch.utils.data import WeightedRandomSampler
import re

def _norm_zm_unit(x, eps=1e-6):
    x = x - x.mean()
    return x / (x.std() + eps)

@torch.no_grad()
def phasecorr_shift_int(a, b, max_shift_px=None, eps=1e-6):
    """
    Returns (dy, dx) integer shift that best aligns b to a (translation only),
    using phase correlation. a,b: [H,W] torch tensors.
    """
    H, W = a.shape
    a0 = _norm_zm_unit(a.float(), eps)
    b0 = _norm_zm_unit(b.float(), eps)

    A = torch.fft.fftn(a0)
    B = torch.fft.fftn(b0)
    R = A * torch.conj(B)
    R = R / (torch.abs(R) + eps)
    cc = torch.fft.ifftn(R).real  # [H,W]

    # peak index
    k = torch.argmax(cc)
    py = (k // W).item()
    px = (k %  W).item()

    # wrap to signed
    if py > H // 2: py -= H
    if px > W // 2: px -= W

    # optional: clamp shifts (prevents crazy jumps)
    if max_shift_px is not None:
        py = int(max(-max_shift_px, min(max_shift_px, py)))
        px = int(max(-max_shift_px, min(max_shift_px, px)))

    return py, px

@torch.no_grad()
def cc_score(a, b, eps=1e-6):
    a = _norm_zm_unit(a.float(), eps).flatten()
    b = _norm_zm_unit(b.float(), eps).flatten()
    return (a @ b) / (a.norm() * b.norm() + eps)


def recenter_stack_per_slice(zst, ref_mode="midz", upsample=10):
    Z, H, W = zst.shape
    ref = zst[Z//2] if ref_mode == "midz" else zst.sum(axis=0)
    out = np.empty_like(zst)
    shifts = []
    for i in range(Z):
        shift, _, _ = phase_cross_correlation(ref, zst[i], upsample_factor=upsample)
        # shift is (dy, dx): how to move zst[i] to match ref
        out[i] = ndimage.shift(zst[i], shift=shift, order=1, mode="constant", cval=0.0)
        shifts.append(shift)
    return out, np.array(shifts)


def recenter_stack_by_centroid(zst, mode="sumz", eps=1e-12):
    """
    zst: (Z,H,W) float
    Returns: zst_shifted, (dy, dx) where positive dy means shift DOWN.
    """
    Z, H, W = zst.shape

    if mode == "sumz":
        ref = zst.sum(axis=0)
    elif mode == "midz":
        ref = zst[Z // 2]
    else:
        raise ValueError("mode must be 'sumz' or 'midz'")

    ref = ref.astype(np.float64, copy=False)
    s = ref.sum()
    if s < eps:
        return zst, (0.0, 0.0)

    yy, xx = np.indices((H, W))
    cy = (ref * yy).sum() / s
    cx = (ref * xx).sum() / s

    # desired center (pixel coords)
    cy0 = (H - 1) / 2.0
    cx0 = (W - 1) / 2.0

    dy = cy0 - cy
    dx = cx0 - cx

    # shift every z-slice the same amount
    zst_shifted = np.stack(
        [ndimage.shift(zst[i], shift=(dy, dx), order=1, mode="constant", cval=0.0) for i in range(Z)],
        axis=0
    ).astype(zst.dtype, copy=False)

    return zst_shifted, (dy, dx)
'''
def estimate_shift_phasecorr(pred, target, eps=1e-8):
    """
    pred, target: [B,H,W] real tensors
    returns: (dy, dx) as int tensors of shape [B]
    """
    # remove DC to reduce background sensitivity
    pred0 = pred - pred.mean(dim=(-2, -1), keepdim=True)
    targ0 = target - target.mean(dim=(-2, -1), keepdim=True)

    Fp = torch.fft.fft2(pred0)
    Ft = torch.fft.fft2(targ0)

    R = Fp * torch.conj(Ft)  # cross-power spectrum
    R = R / (torch.abs(R) + eps)  # phase correlation normalization

    corr = torch.fft.ifft2(R).real  # [B,H,W], peak at shift (circular)
    B, H, W = corr.shape

    flat_idx = corr.reshape(B, -1).argmax(dim=1)
    peak_y = flat_idx // W
    peak_x = flat_idx % W

    # convert from [0..H-1] circular to signed shift in [-H/2..H/2]
    dy = peak_y.clone()
    dx = peak_x.clone()
    dy = torch.where(dy > H // 2, dy - H, dy)
    dx = torch.where(dx > W // 2, dx - W, dx)

    return dy.to(torch.int64), dx.to(torch.int64)
'''
'''
def _autocorr2d(x, eps=1e-8):
    """
    x: [B,H,W] real tensor
    returns: [B,H,W] real autocorrelation (fftshifted), normalized
    """
    # remove DC to reduce background sensitivity
    x = x - x.mean(dim=(-2, -1), keepdim=True)

    # FFT-based autocorrelation: ifft(|FFT(x)|^2)
    X = torch.fft.fft2(x)
    ac = torch.fft.ifft2((X.conj() * X)).real  # [B,H,W]

    # center the peak
    ac = torch.fft.fftshift(ac, dim=(-2, -1))

    # normalize (shape-only)
    ac = ac - ac.amin(dim=(-2, -1), keepdim=True)
    ac = ac / (ac.sum(dim=(-2, -1), keepdim=True) + eps)
    return ac

def autocorr_mse_loss(pred, target):
    ap = _autocorr2d(pred)
    at = _autocorr2d(target)
    return torch.mean((ap - at) ** 2)
'''
# end autocorrelation addition

def _center_crop(im, out_hw):
    out_h, out_w = out_hw
    h, w = im.shape
    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    return im[y0:y0 + out_h, x0:x0 + out_w]


def _norm01_sum(im):
    im = im.astype(np.float32, copy=False)
    im = im - im.min()
    s = float(im.sum())
    if s > 0:
        im /= s
    return im


def _cc(a, b):
    a = a.astype(np.float32, copy=False).ravel()
    b = b.astype(np.float32, copy=False).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / den)


def _to_uint16_stack(stack, mode="per_slice_max"):
    st = stack.astype(np.float32, copy=False)

    if mode == "global_max":
        mx = float(st.max())
        if mx <= 0:
            return np.zeros_like(st, dtype=np.uint16)
        return (np.clip(st / mx, 0, 1) * 65535.0).astype(np.uint16)

    # per_slice_max
    out = np.zeros_like(st, dtype=np.uint16)
    mx = st.reshape(st.shape[0], -1).max(axis=1)
    for i in range(st.shape[0]):
        if mx[i] > 0:
            out[i] = (np.clip(st[i] / float(mx[i]), 0, 1) * 65535.0).astype(np.uint16)
    return out


def fit_mask_offset_from_offaxis_stacks(
    param_dict,
    #d_search_um=(00000.0, 70000.0),#d_search_um=(00000.0, 80000.0),
    d_search_um=(00000.0, 70000.0),#d_search_um=(00000.0, 80000.0),
    d_coarse_step_um=5000.0,
    d_fine_step_um=250.0,
    photons_for_sim=1e4,
    save_dir=None,
    save_uint16_mode="per_slice_max",
    make_montage=True,
):
    if not param_dict.get("offaxis_zstack_files"):
        print("[fit d] No off-axis stacks provided. Skipping.")
        return None

    # --- output directory (ONE place only) ---
    if save_dir is None:
        time_now = datetime.today().strftime("%Y%m%d_%H%M%S")
        #save_dir = os.path.join(os.getcwd(), f"mask_offset_fit_{time_now}")
        save_dir = os.path.join(os.getcwd(), f"mask_fit_outputs")
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # --- load nfps & enforce monotonic (fixes “flip”) ---
    #nfps_raw = np.asarray(param_dict["nfps"], dtype=np.float32)
    #Z_expected = len(nfps_raw)

    nfps = np.asarray(param_dict["nfps"], dtype=np.float32)




    # If z values are strictly decreasing, reverse them AND reverse every exp z-stack
    if np.all(np.diff(nfps) < 0):
        nfps = nfps[::-1].copy()
        reverse_exp_stacks = True
    else:
        reverse_exp_stacks = False

    Z_expected = len(nfps)

    '''
    # If nfps is not strictly increasing, replace with a monotonic grid
    if not np.all(np.diff(nfps_raw) > 0):
        z0, z1 = float(nfps_raw.min()), float(nfps_raw.max())
        nfps = np.linspace(z0, z1, Z_expected, dtype=np.float32)
        print(f"[fit d] WARNING: nfps not monotonic -> using linspace({z0:.3f}, {z1:.3f}, {Z_expected})")
    else:
        nfps = nfps_raw
    '''
    # --- model ---
    from DS3Dplus.ds3d_utils import ImModelTraining
    model = ImModelTraining(param_dict)
    model.eval()

    r0, c0 = map(float, param_dict["centralBeadCoordinates_pixel"])
    ps_cam = float(param_dict["ps_camera"])
    M = (param_dict["M"])

    # --- load stacks once ---
    stacks = []
    for f, (rr, cc) in zip(param_dict["offaxis_zstack_files"], param_dict["offaxis_coords_pixel"]):
        zstack = io.imread(f).astype(np.float32)  # (Z,H,W)
        #if reverse_exp_stacks:
            #zstack = zstack[::-1].copy()
            #nfps = nfps[::-1].copy()


        if zstack.shape[0] != Z_expected:
            raise ValueError(f"[fit d] Z mismatch: {f} has Z={zstack.shape[0]} but nfps has {Z_expected}.")

        dx_pix = float(cc)# - c0
        dy_pix = float(rr)# - r0
        x_um = dx_pix * (ps_cam/M)
        y_um = dy_pix * (ps_cam/M)

        stacks.append({
            "file": f,
            "name": os.path.splitext(os.path.basename(f))[0],
            "exp": zstack,
            "H": zstack.shape[1],
            "W": zstack.shape[2],
            "x_um": float(x_um),
            "y_um": float(y_um),
        })


    def simulate_stack_for_d(d_um, st):
        model.mask_offset_in_um = float(d_um)

        Z, Hroi, Wroi = st["exp"].shape[0], st["H"], st["W"]
        sim_stack = np.zeros((Z, Hroi, Wroi), dtype=np.float32)
        cc_per_z = np.zeros((Z,), dtype=np.float32)

        x_um, y_um = st["x_um"], st["y_um"]

        oldNFP = float(model.NFP)

        for zi, nfp_um in enumerate(nfps):
            exp_im = _norm01_sum(st["exp"][zi])
            model.NFP = float(nfp_um)  # scan -> NFP
            xyzp = np.array([x_um, y_um, 0.0, float(photons_for_sim)], dtype=np.float32)

            sim = model.psf_patch_clean(xyzp)
            # IMPORTANT FIX: float32 (prevents Float vs Double mismatch in torch)
            #xyzp = np.array([x_um, y_um, float(z_um), float(photons_for_sim)], dtype=np.float32)

            sim = _center_crop(sim, (Hroi, Wroi))
            sim = _norm01_sum(sim)


            Blur = True
            if Blur:
                g_sigma = param_dict["g_sigma"]
                g_sigma = torch.tensor(g_sigma)
                g_size = 9 #hard coded! to fix
                g_r = int(g_size / 2)
                #g_xs = torch.linspace(-g_r, g_r, g_size, device=device).type(torch.float64)
                g_xs = torch.linspace(-g_r, g_r, g_size).type(torch.float64)
                g_xx, g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

                # blur
                # blur (batched)
                blur_kernel = 1 / (2 * pi * g_sigma[0] ** 2) * (
                    torch.exp(-0.5 * (g_xx ** 2 + g_yy ** 2) / g_sigma[0] ** 2)   )
                sim_tensor = torch.tensor(sim)
                sim_tensor = F.conv2d(sim_tensor.unsqueeze(0).unsqueeze(0), blur_kernel.type_as(sim_tensor).unsqueeze(0).unsqueeze(0), padding='same' ).squeeze(1)
                '''sim = F.conv2d(
                    sim.unsqueeze(1),
                    blur_kernel.unsqueeze(0).unsqueeze(0).type_as(sim),
                    padding='same'
                ).squeeze(1)'''
                # photon normalization
                #sim = sim / torch.sum(psfs, dim=(1, 2), keepdims=True) * xyzps[:, 3:4].unsqueeze(         1)  # photon normalization
                # sim = sim[:, self.idx05 - self.h05:self.idx05 + self.h05 + 1, self.idx05 - self.w05:self.idx05 + self.w05 + 1]
                #sim = sim[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]
            sim_tensor = sim_tensor.squeeze(0)
            sim_stack[zi] = sim_tensor
            cc_per_z[zi] = _cc(exp_im, sim_tensor.numpy())



        model.NFP = oldNFP  # turning it back to experimental nfp
        return sim_stack, cc_per_z

    def score_for_d(d_um):
        ccs = []
        for st in stacks:
            _, cc_per_z = simulate_stack_for_d(d_um, st)
            ccs.append(cc_per_z)
        ccs = np.concatenate(ccs) if ccs else np.array([-1e9], dtype=np.float32)
        return float(ccs.mean())

    # ---- coarse search ----
    d_search_um = param_dict["mask_offset_in_um"], param_dict["mask_offset_in_um"]+1e-6
    d0, d1 = map(float, d_search_um)

    d_vals = np.arange(d0, d1 + 1e-6, float(d_coarse_step_um), dtype=np.float32)
    scores = [score_for_d(d) for d in d_vals]
    best_d = float(d_vals[int(np.argmax(scores))])

    # ---- fine search ----
    lo = max(d0, best_d - 2 * float(d_coarse_step_um))
    hi = min(d1, best_d + 2 * float(d_coarse_step_um))
    d_vals2 = np.arange(lo, hi + 1e-6, float(d_fine_step_um), dtype=np.float32)
    scores2 = [score_for_d(d) for d in d_vals2]
    best_d2 = float(d_vals2[int(np.argmax(scores2))])
    best_s2 = float(max(scores2))

    # save to param_dict
    param_dict["mask_offset_in_um"] = best_d2
    param_dict["mask_offset_fit_info"] = {
        "best_d_um": best_d2,
        "best_cc": best_s2,
        "coarse": {"d": d_vals.tolist(), "cc": [float(x) for x in scores]},
        "fine": {"d": d_vals2.tolist(), "cc": [float(x) for x in scores2]},
        "save_dir": save_dir,
        "nfps_used": nfps.tolist(),
    }

    print(f"[fit d] best mask_offset_in_um = {best_d2:.1f} um, mean CC={best_s2:.4f}")
    print(f"[fit d] saving outputs to: {save_dir}")

    # ---- save curves ----
    with open(os.path.join(save_dir, "cc_curve_coarse.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d_um", "mean_cc"])
        w.writerows([[float(d), float(s)] for d, s in zip(d_vals, scores)])

    with open(os.path.join(save_dir, "cc_curve_fine.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["d_um", "mean_cc"])
        w.writerows([[float(d), float(s)] for d, s in zip(d_vals2, scores2)])

    plt.figure(figsize=(6, 4))
    plt.plot(d_vals, scores, marker="o", linewidth=1)
    plt.plot(d_vals2, scores2, marker="o", linewidth=1)
    plt.axvline(best_d2, linestyle="--")
    plt.xlabel("mask_offset_in_um (d) [um]")
    plt.ylabel("mean CC")
    plt.title(f"Best d = {best_d2:.1f} um, mean CC={best_s2:.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cc_curve.png"), dpi=200)
    plt.close()

    # ---- save exp/sim/montage for best d (ALL in same dir) ----
    for st in stacks:
        sim_stack, cc_per_z = simulate_stack_for_d(best_d2, st)  # <-- removed bogus save_results

        with open(os.path.join(save_dir, f"cc_per_z_{st['name']}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["z_um", "cc"])
            w.writerows([[float(z), float(cc)] for z, cc in zip(nfps, cc_per_z)])

        exp_u16 = _to_uint16_stack(st["exp"], mode=save_uint16_mode)
        sim_u16 = _to_uint16_stack(sim_stack, mode=save_uint16_mode)

        io.imsave(os.path.join(save_dir, f"exp_stack_{st['name']}.tif"), exp_u16, check_contrast=False)
        io.imsave(os.path.join(save_dir, f"sim_stack_bestd_{st['name']}.tif"), sim_u16, check_contrast=False)

        if make_montage:
            Z, H, W = exp_u16.shape
            montage = np.zeros((Z, H, 2 * W), dtype=np.uint16)
            montage[:, :, :W] = exp_u16
            montage[:, :, W:] = sim_u16
            io.imsave(os.path.join(save_dir, f"comparison_montage_{st['name']}.tif"), montage, check_contrast=False)

    return best_d2

'''
'''
# End Ori's edit added on 2/01/2026 for mask displacement optimization

class ImModel_pr(nn.Module):
    def __init__(self, params):
        """
        a scalar model for air or oil objective in microscopy
        """
        super().__init__()
        device = params['device']
        # ori's edit
        device = torch.device(
            "cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
        print(f'device used: {device}')
        # end ori's edit
        ################### set parameters: unit:um
        # oil objective
        M = params['M']  # magnification
        NA = params['NA']  # NA
        n_immersion = params['n_immersion']  # refractive index of the immersion of the objective
        lamda = params['lamda']  # wavelength
        n_sample = params['n_sample']   # refractive index of the sample
        f_4f = params['f_4f']  # focal length of 4f system
        ps_camera = params['ps_camera']  # pixel size of the camera
        ps_BFP = params['ps_BFP']  # pixel size at back focal plane
        # ori's edit from 26/01/2026 - phase retrival with d
        #self.mask_offset_in_um = float(params.get('mask_offset_in_um', 0.0)) #27/01/2026
        #self.lamda = lamda
        #self.ps_BFP = ps_BFP
        self.f_4f = f_4f
        # 27/01/2026 - improved pr
        self.lamda = float(lamda)
        self.ps_BFP = float(ps_BFP)

        # ---- learnable d (mask displacement) ----
        self.d_min_um = 8000 *1 #float(params.get("d_min_um", 10000.0))
        self.d_max_um = float(params.get("d_max_um", 16000.0*0 ))
       #d_init = float(params.get("mask_offset_in_um", 0.0))
        if "mask_offset_in_um" in params:
            if params["mask_offset_in_um"]!=0:  # bandage! to fix!
                d_init = float(params["mask_offset_in_um"])
            else:
                d_init = 0.5 * (self.d_min_um + self.d_max_um)  # midrange default
        else:
            d_init = 0.5 * (self.d_min_um + self.d_max_um)  # midrange default
        self.mask_offset_in_um = float(params.get("mask_offset_in_um", 0.0))

        self.centralBeadCoordinates_pixel = list(params.get('centralBeadCoordinates_pixel', [600,600]))  # 27/01/2026
        self.ps_camera = params['ps_camera']
        self.NFP = params['NFP']  # location of the nominal focal plane
        self.n_immersion = params['n_immersion']
        self.NA = params['NA']
        self.M = params['M']


        # map initial d into an unconstrained parameter via inverse-sigmoid (logit)
        eps = 1e-3
        p = (d_init - self.d_min_um) / (self.d_max_um - self.d_min_um + 1e-12)
        p = min(max(p, eps), 1.0 - eps)
        d_raw_init = math.log(p / (1.0 - p))

        self.d_raw = nn.Parameter(torch.tensor(d_raw_init, device=device, dtype=torch.float32))
        # end improved pr 27/01/2026

        # end ori's edit from 26/01/2026 - phase retrival with d

        # image
        H, W = params['H'], params['W']  # FOV size
        g_size = 9  # size of the gaussian blur kernel
        g_sigma = params['g_sigma']  # std of the gaussian blur kernel

        ###################

        N = np.floor(f_4f * lamda / (ps_camera * ps_BFP))  # simulation size
        N = int(N + 1 - (N % 2))  # make it odd
        print(f'Simulation size of the imaging model is {N} which must be larger than image size (PSF z-stack and training images)!')

        # pupil/aperture at back focal plane
        d_pupil = 2 * f_4f * NA / np.sqrt(M ** 2 - NA ** 2)  # diameter [um]
        #print('d_pupil = ' + str(d_pupil))
        pn_pupil = d_pupil / ps_BFP  # pixel number of the pupil diameter should be smaller than the simulation size N
        if N < pn_pupil:
            raise Exception('Simulation size is smaller than the pupil!')
        # cartesian and polar grid in BFP
        x_phys = np.linspace(-N / 2, N / 2, N) * ps_BFP
        xi, eta = np.meshgrid(x_phys, x_phys)  # cartesian physical coordinates
        r_phys = np.sqrt(xi ** 2 + eta ** 2)
        pupil = (r_phys < d_pupil / 2).astype(np.float32)

        x_ang = np.linspace(-1, 1, N) * (N / pn_pupil) * (NA / n_immersion)  # angular coordinate
        xx_ang, yy_ang = np.meshgrid(x_ang, x_ang)
        r = np.sqrt(
            xx_ang ** 2 + yy_ang ** 2)  # normalized angular coordinates, s.t. r = NA/n_immersion at edge of E field support

        k_immersion = 2 * pi * n_immersion / lamda  # [1/um]
        sin_theta_immersion = r
        #circ_NA = (sin_theta_immersion < (NA / n_immersion)).astype(
        #    np.float32)  # the same as pupil, NA / n_immersion < 1

        # ori's edit - added on 26/01/2026
        circ_scale = float(params.get("circ_scale", 1.0))  # 1.0 = default behavior
        r_lim = (NA / n_immersion)
        r_lim_scaled = (NA / n_immersion) * circ_scale
        #r_lim_scaled = circ_scale
        print("rlim_scaled = " + str(r_lim_scaled))
        print("r_lim = " + str(r_lim))
        r_lim = min(r_lim, 1.0)  # can't exceed sin(theta)=1
        #r_lim_scaled = min(r_lim, 1.0)

        circ_NA = (sin_theta_immersion < r_lim).astype(np.float32)
        circ_NA_scaled = (sin_theta_immersion < r_lim_scaled).astype(np.float32)  # 28/01/2026

        # end ori's edit from 26/01/2026

        cos_theta_immersion = np.sqrt(1 - (sin_theta_immersion * circ_NA) ** 2) * circ_NA

        k_sample = 2 * pi * n_sample / lamda
        sin_theta_sample = n_immersion / n_sample * sin_theta_immersion
        # note: when circ_sample is smaller than circ_NA, super angle fluorescence apears
        circ_sample = (sin_theta_sample < 1).astype(np.float32)  # if all the frequency of the sample can be captured
        cos_theta_sample = np.sqrt(1 - (sin_theta_sample * circ_sample) ** 2) * circ_sample * circ_NA

        # circular aperture to impose on BFP, SAF is excluded
        circ = circ_NA * circ_sample
        circ_scaled = circ_NA_scaled
        #circ = circ_NA   # include SAF! 25/01/2026
        #print(str(np.sum(circ)))
        #print(str(np.sum(circ_NA * circ_sample)))

        pn_circ = np.floor(np.sqrt(np.sum(circ) / pi) * 2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * pi * xi * M / (lamda * f_4f)
        Ygrid = 2 * pi * eta * M / (lamda * f_4f)
        Zgrid = k_sample * cos_theta_sample
        NFPgrid = k_immersion * (-1) * cos_theta_immersion  # -1

        self.device = device
        # ori's edit
        self.device = torch.device(
            "cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
        print(f'device used (ImModel_pr): {device}')
        # end ori's edit
        self.Xgrid = torch.from_numpy(Xgrid).to(device)
        self.Ygrid = torch.from_numpy(Ygrid).to(device)
        self.Zgrid = torch.from_numpy(Zgrid).to(device)
        self.NFPgrid = torch.from_numpy(NFPgrid).to(device)
        self.circ = torch.from_numpy(circ).to(device)
        self.circ_NA = torch.from_numpy(circ_NA).to(device)
        self.circ_sample = torch.from_numpy(circ_sample).to(device)
        self.idx05 = int(N / 2)
        self.N = N
        self.pn_pupil = pn_pupil
        self.pn_circ = pn_circ
        self.circ_scaled = torch.from_numpy(circ_scaled).to(device)
        # for a blur kernel
        g_r = int(g_size / 2)
        g_xs = torch.linspace(-g_r, g_r, g_size, device=device).type(torch.float64)
        self.g_xx, self.g_yy = torch.meshgrid(g_xs, g_xs, indexing='xy')

        # crop settings
        self.r0, self.c0 = int(np.round((N-H)/2)), int(np.round((N-W)/2))
        # h05, w05 = int(H / 2), int(W / 2)
        # self.h05, self.w05 = h05, w05
        self.H, self.W = H, W

        # -------------------------
        # DEBUG: BFP logging  #27/01/2026
        # -------------------------
        self.debug_bfp = bool(params.get("debug_bfp", True))  # hard-code True if you want
        self.debug_every = int(params.get("debug_every", 500//2))  # every N forward calls
        self.debug_dir = str(params.get("debug_dir", os.path.join("debug", "bfp")))
        self.debug_max_emitters = int(params.get("debug_max_emitters", 5 ))  # save first K in batch  number of beads
        self._debug_call_idx = 0
        #self.phase_mask = torch.tensor(circ, device=device, requires_grad=True)
        self.phase_mask = torch.zeros((N, N), device=device, requires_grad=True)

        self.g_sigma = torch.tensor(g_sigma, device=device, requires_grad=True)



    # ori's edit from 27/01/2026 for improved pr with displacement
    def d_um(self):
        # bounded to [d_min_um, d_max_um]
        return self.d_min_um + (self.d_max_um - self.d_min_um) * torch.sigmoid(self.d_raw)
    # end ori's edit from 27/01/2026 for improved pr with displacement

    # added on 27/01/2026
    def _maybe_save_debug(self, ef_bfp_eff, psfs, xyzps, NFPs):
        if not self.debug_bfp:
            return

        self._debug_call_idx += 1
        if (self._debug_call_idx % self.debug_every) != 0:
            return

        # lazy import so training isn't slowed when debug off
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        os.makedirs(self.debug_dir, exist_ok=True)

        d_now = float(self.d_um().detach().cpu().item())
        g_now = float(self.g_sigma.detach().cpu().item()) if hasattr(self, "g_sigma") else float("nan")

        B = ef_bfp_eff.shape[0]
        K = self.debug_max_emitters #min(B, self.debug_max_emitters)
        if B == K:
            num_of_stacks_per_bead = B // K
        else:
            num_of_stacks_per_bead = B
            K=1

        for i in range(K):
            #subdir = os.path.join(self.debug_dir, f"emitter_{i:03d}")
            label = f"emitter_{i:03d}"
            if hasattr(self, "debug_names") and i < len(self.debug_names):
                label = str(self.debug_names[i])
            subdir = os.path.join(self.debug_dir, label)

            os.makedirs(subdir, exist_ok=True)
            #inx = int( (i+0.5*1) * num_of_stacks_per_bead)
            if B == K:
                inx = int( (0.5*1) * num_of_stacks_per_bead) + i
            else:
                inx = int( (0.5*1) * num_of_stacks_per_bead) + 1

            #phase_eff = (torch.angle(ef_bfp_eff[inx]) * self.circ).detach().cpu().numpy()
            phase_eff = (torch.angle(ef_bfp_eff[inx])).detach().cpu().numpy()

            # PSF normalize for display (not for training!)
            psf = psfs[inx].detach().cpu().numpy()
            psf_disp = psf / (psf.max() + 1e-12)

            # xyz + nfp for title
            x_um = float(xyzps[inx, 0].detach().cpu().item())
            y_um = float(xyzps[inx, 1].detach().cpu().item())
            z_um = float(xyzps[inx, 2].detach().cpu().item())
            nfp = float(NFPs[inx].detach().cpu().item()) if NFPs is not None else float("nan")

            fig = plt.figure(figsize=(10, 4))

            ax1 = fig.add_subplot(1, 2, 1)
            im1 = ax1.imshow(phase_eff, cmap="twilight")
            ax1.set_title("effective BFP phase")
            ax1.axis("off")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = fig.add_subplot(1, 2, 2)
            im2 = ax2.imshow(psf_disp, cmap="gray")
            ax2.set_title("PSF (display norm)")
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            fig.suptitle(
                f"call={self._debug_call_idx}  d={d_now:.1f}um  g={g_now:.3f}  "
                f"x={x_um:.3f} y={y_um:.3f} z={z_um:.3f}  NFP={nfp:.3f}"
            )
            fig.tight_layout()

            out = os.path.join(subdir, f"call_{self._debug_call_idx:06d}.png")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
    #end added on 27/01/2026

    def forward(self, xyzps, NFPs):
        import torch
        import torch.nn.functional as F
        from math import pi

        def shift2d_integer(img2d: torch.Tensor, shift_x_px: int, shift_y_px: int):
            """
            Integer-pixel shift with zero fill.
            Positive shift_x_px -> right
            Positive shift_y_px -> down
            """
            H, W = img2d.shape
            out = torch.zeros_like(img2d)

            src_x0 = max(0, -shift_x_px)
            src_x1 = min(W, W - shift_x_px) if shift_x_px >= 0 else W
            dst_x0 = max(0, shift_x_px)
            dst_x1 = min(W, W + shift_x_px) if shift_x_px < 0 else W

            src_y0 = max(0, -shift_y_px)
            src_y1 = min(H, H - shift_y_px) if shift_y_px >= 0 else H
            dst_y0 = max(0, shift_y_px)
            dst_y1 = min(H, H + shift_y_px) if shift_y_px < 0 else H

            out[dst_y0:dst_y1, dst_x0:dst_x1] = img2d[src_y0:src_y1, src_x0:src_x1]
            return out

        def shift_complex_field_integer(field2d: torch.Tensor, shift_x_px: int, shift_y_px: int):
            real_shifted = shift2d_integer(field2d.real, shift_x_px, shift_y_px)
            imag_shifted = shift2d_integer(field2d.imag, shift_x_px, shift_y_px)
            return torch.complex(real_shifted, imag_shifted)

        xyzp = xyzps  # [B,4], um in object space

        # -----------------------------------
        # coordinates
        # -----------------------------------
        x_pix = xyzp[:, 0:1] / self.ps_camera * self.M
        y_pix = xyzp[:, 1:2] / self.ps_camera * self.M

        z = xyzp[:, 2:3]
        photons = xyzp[:, 3:4].unsqueeze(1)
        # already relative to optical axis!
        cx = 0 # float(self.centralBeadCoordinates_pixel[1])  # col
        cy = 0 # float(self.centralBeadCoordinates_pixel[0])  # row

        x_pix_rel = x_pix - cx
        y_pix_rel = y_pix - cy

        x_rel = x_pix_rel * self.ps_camera / self.M
        y_rel = y_pix_rel * self.ps_camera / self.M

        x_coarse = torch.round(x_pix_rel) * self.ps_camera / self.M
        y_coarse = torch.round(y_pix_rel) * self.ps_camera / self.M

        x_sub = x_rel - x_coarse
        y_sub = y_rel - y_coarse

        # -----------------------------------
        # BFP phase: axial + delicate sub-pixel lateral phase only
        # -----------------------------------
        NFPs_b = NFPs.to(self.NFPgrid.dtype).view(-1, 1, 1)

        phase_axial = self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b
        phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)

        circ_final_bfp = self.circ_NA
        ef_bfp = torch.exp(1j * (phase_axial + phase_lateral_sub_pixel)).to(torch.complex64)
        ef_bfp = ef_bfp * circ_final_bfp
        ef_bfp = torch.where(circ_final_bfp > 0.5, ef_bfp, 0)

        # optional debug field
        #ebfp_on_axis = torch.exp(1j * phase_axial).to(torch.complex64) * self.circ

        # -----------------------------------
        # propagate to mask plane
        # -----------------------------------
        d = self.d_um() #if callable(self.d_um) else self.d_um
        d_scalar = d # float(d.detach().cpu().item()) if torch.is_tensor(d) else float(d)

        if abs(d_scalar) > 0:
            ef_mask = asm_propagate( ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_scalar, n=1.0, bandlimit=True).to(torch.complex64)
        else:
            ef_mask = ef_bfp

        # -----------------------------------
        # convert coarse lateral position to mask-plane shift
        # NOTE:
        # This mapping is the part you should calibrate physically.
        # Current version uses a small-angle geometric approximation.
        #
        # x_coarse, y_coarse are in object-space um
        # f_4f is used here as a placeholder effective propagation geometry. for a microscopic system with no 4f: the 4f value should be tube lens/M (e.g. f_obj)
        # -----------------------------------
        theta_x = x_coarse / (self.f_4f / self.M)
        theta_y = y_coarse / (self.f_4f / self.M)

        dx_mask_um = d_scalar * theta_x
        dy_mask_um = d_scalar * theta_y

        dx_mask_px = (dx_mask_um / self.ps_BFP).squeeze(1)
        dy_mask_px = (dy_mask_um / self.ps_BFP).squeeze(1)

        dx_mask_px = torch.round(dx_mask_px).to(torch.int64)
        dy_mask_px = torch.round(dy_mask_px).to(torch.int64)
        # -----------------------------------
        # shift complex field at mask plane
        ef_mask_shifted = []
        for i in range(ef_mask.shape[0]):
            #ef_mask_shifted.append(shift_complex_field(ef_mask[i], dx_mask_px[i], dy_mask_px[i]))
            ef_mask_shifted.append(shift_complex_field_integer(ef_mask[i], int(dx_mask_px[i].item()), int(dy_mask_px[i].item())))
        ef_mask_shifted = torch.stack(ef_mask_shifted, dim=0)

        # -----------------------------------
        # apply phase mask at mask plane
        phase = torch.exp(1j * self.phase_mask.to(ef_mask.device).to(torch.float32))
        circ_phase = (self.circ_scaled > 0.5).unsqueeze(0)

        ef_mask_shifted = ef_mask_shifted * phase * circ_phase
        ef_mask_shifted = torch.where(circ_phase > 0.5, ef_mask_shifted, 0)

        # -----------------------------------
        # shift back
        ef_mask_unshifted = []
        for i in range(ef_mask_shifted.shape[0]):
            #ef_mask_unshifted.append(shift_complex_field(ef_mask_shifted[i], -dx_mask_px[i], -dy_mask_px[i]))
            ef_mask_unshifted.append(shift_complex_field_integer(ef_mask_shifted[i], int(-dx_mask_px[i].item()), int(-dy_mask_px[i].item())))
        ef_mask_unshifted = torch.stack(ef_mask_unshifted, dim=0)

        # -----------------------------------
        # propagate back to BFP
        if abs(d_scalar) > 0:
            ef_bfp_after = asm_propagate(ef_mask_unshifted,self.lamda,self.ps_BFP,self.ps_BFP,-d_scalar, n=1.0,bandlimit=True).to(torch.complex64)
        else:
            ef_bfp_after = ef_mask_unshifted

        ef_bfp_after = torch.where(circ_final_bfp > 0.5, ef_bfp_after, 0)

        # -----------------------------------
        # image plane FFT
        psf_field = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ef_bfp_after, dim=(1, 2)), dim=(1, 2)),dim=(1, 2))
        psf = torch.abs(psf_field) ** 2
        psfs = psf / (torch.sum(psf, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # blur
        #if len(self.g_sigma) == 1:
        #g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
        #else:
        #    g_sigma = self.g_sigma
        #sigma = g_sigma[0] + torch.rand(1).to(self.device) * (g_sigma[1] - g_sigma[0])

        sigma = self.g_sigma
        blur_kernel = 1 / (2 * pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs), padding='same').squeeze(1)

        # renormalize after blur
        psfs = psfs / (torch.sum(psfs, dim=(1, 2), keepdims=True) + 1e-12) * photons

        # crop
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        # debug
        if self.debug_bfp:
            self._maybe_save_debug(ef_bfp_after, psfs, xyzps, NFPs)

        return psfs

    ''' removed on 05/04/2026 (ditching s approach)
    def forward(self, xyzps, NFPs):

        import torch
        import torch.nn.functional as F
        from math import pi
        ''''''
        xx = np.linspace(-0.5,0.5,self.circ.shape[1]) * self.circ.shape[1]
        yy = np.linspace(-0.5,0.5,self.circ.shape[0]) * self.circ.shape[0]
        XX,YY = np.meshgrid(xx,yy)
        circ_new = torch.ones_like(self.circ)
        circ_new[((XX)**2+(YY)**2 > (160/2)**2)] = 0
        #circ_new[((XX+40)**2+(YY-40)**2 < (130/2)**2)] = 1
        #circ_new[((XX-40)**2+(YY+40)**2 < (130/2)**2)] = 1
        ''''''
        xyzp = xyzps  # um in object space
        # pixel coordinates
        x_pix = xyzp[:, 0:1] / self.ps_camera * self.M
        y_pix = xyzp[:, 1:2] / self.ps_camera * self.M

        # optical axis in pixels (col, row)
        cx = self.centralBeadCoordinates_pixel[1]
        cy = self.centralBeadCoordinates_pixel[0]

        # pixel offset from optical axis
        x_pix_rel = x_pix# - cx #already centered here
        y_pix_rel = y_pix# - cy

        # convert to sample-plane micrometers
        x = x_pix_rel * self.ps_camera /self.M #
        y = y_pix_rel * self.ps_camera /self.M

        z = xyzp[:, 2:3]

        photons = xyzp[:, 3:4].unsqueeze(1)

        # --- finite differences (center pixel) ---
        i0 = self.Xgrid.shape[0] // 2
        j0 = self.Xgrid.shape[1] // 2
        dX_col = (self.Xgrid[i0, j0 + 1] - self.Xgrid[i0, j0]).abs()  # phase coef step per +1 col
        dY_row = (self.Ygrid[i0 + 1, j0] - self.Ygrid[i0, j0]).abs()  # phase coef step per +1 row

        # true per-pixel phase increments
        dphi_col = dX_col * i0  # ~ |ΔX|*|x|
        dphi_row = dY_row * j0  # ~ |ΔY|*|y|

        # cycles/pixel
        ax = (dphi_col / (2 * torch.pi)).item()
        ay = (dphi_row / (2 * torch.pi)).item()

        ax = np.floor(np.abs(ax)) * 2 + 1
        ay = np.floor(np.abs(ay)) * 2 + 1
        s = int(np.sqrt(ax ** 2 + ay ** 2))  # ori's edit from 16/12/2025! to fix error in edges

        # --- scale lateral tilt & distance only ---
        x_s = x / s
        y_s = y / s
        z_s = xyzp[:, 2:3] / s  # note -should be absolute value??
        d = self.d_um()  # torch scalar, has grad
        d_eff = d * s
        #print('Debug!! d=' + str(d))

        # calculating round phase shift to keep sub pixel shift

        x_sub = x - torch.round(x *self.M / self.ps_camera) * self.ps_camera / self.M
        y_sub = y - torch.round(y *self.M / self.ps_camera) * self.ps_camera / self.M
        phase_lateral_sub_pixel = self.Xgrid * x_sub.unsqueeze(1) + self.Ygrid * y_sub.unsqueeze(1)
        phase_lateral_sub_pixel = phase_lateral_sub_pixel
        #phase_lateral_sub_pixel *=0
        #

        # phases with scaled lateral tilt, original axial
        NFPs_b = NFPs.to(self.NFPgrid.dtype).view(-1, 1, 1)
        NFPs_s = NFPs_b / s
        phase_axial = (self.Zgrid * z_s.unsqueeze(1) + self.NFPgrid * NFPs_s)
        phase_nfp = (self.NFPgrid * NFPs_b) .to(torch.complex64)

        actual_phase_axial = (self.Zgrid * z.unsqueeze(1) + self.NFPgrid * NFPs_b)
        phase_lateral = self.Xgrid * x_s.unsqueeze(1) + self.Ygrid * y_s.unsqueeze(1)
        circ_final_bfp = self.circ_NA # *self.circ_sample or self.circ_NA
        #circ_final_bfp = self.circ_sample # for mask design

        #ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)
        ebfp_on_axis = torch.exp(1j * (actual_phase_axial)).to(torch.complex64)

        # adding amplitude of bfp 22/12/2025
        # xi = self.Xgrid/2/np.pi/self.M *(self.lamda*self.f_4f) / self.ps_BFP / (self.N//2)
        # eta = self.Ygrid/2/np.pi/self.M *(self.lamda*self.f_4f) / self.ps_BFP
        inx1 = torch.where(self.circ[np.shape(self.circ)[0] // 2, :] == 1)
        inx1 = inx1[0][0]
        inx1 = (np.shape(self.circ)[0] / 2 - inx1).to(self.device)

        x_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
        x_norm = x_phys / inx1
        y_phys = torch.linspace(-self.N / 2, self.N / 2, self.N).to(self.device)
        y_norm = y_phys / inx1

        # xx, yy = meshgrid(x_norm, y_norm); need to do meshgrid
        # rho2 = xx ** 2 + yy ** 2
        rho2 = x_norm ** 2 + y_norm ** 2
        amp = 1 / torch.abs(((1 - rho2 * (self.NA / self.n_immersion) ** 2 + 1e-16)) ** (1 / 4))
        ef_bfp = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64) * circ_final_bfp
        ef_bfp[torch.isnan(ef_bfp)] = 0

        ef_bfp = torch.where(circ_final_bfp>0.5, ef_bfp, 0)  #  removed on 12/17/2025 - probably it is better this way!
        ef_bfp_nocirc = torch.exp(1j * (phase_axial + phase_lateral)).to(torch.complex64)

        # forward propagate
        if d_eff != 0.0:
            ef_off = asm_propagate(ef_bfp, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)
            ef_off_nocirc = asm_propagate(ef_bfp_nocirc, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)
            #ef_off_on_axis =  asm_propagate(ebfp_on_axis, self.lamda, self.ps_BFP, self.ps_BFP, +d_eff, n=1.0, bandlimit=True)

        else:
            ef_off = ef_bfp
            ef_off_nocirc = ef_bfp_nocirc

        # mask & circ
        phase = torch.exp(1j * self.phase_mask.to(ef_off.device).to(torch.float32))
        # correcting phase so mask will be without axial offset

        circ_phase = (self.circ_scaled > 0.5).unsqueeze(0)

        ef_off = ef_off * phase * circ_phase
        ef_off = torch.where(circ_phase>0.5, ef_off, 0)

        # back propagate
        if d_eff != 0.0:
            ef_bfp = asm_propagate(ef_off, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0, bandlimit=True).to(
                torch.complex64)

            ef_bfp_nocirc = asm_propagate(ef_off_nocirc, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0, bandlimit=True).to(
                torch.complex64)

            ef_phase_mask_only = phase * circ_phase * torch.exp(1j * phase_lateral)
            ef_phase_mask_only =asm_propagate(ef_phase_mask_only, self.lamda, self.ps_BFP, self.ps_BFP, -d_eff, n=1.0, bandlimit=True).to(torch.complex64)


        else:
            ef_bfp = ef_off
            ef_bfp_nocirc = ef_off_nocirc
            ef_phase_mask_only = (phase * circ_phase) .to(torch.complex64)

        # remove only the (scaled) lateral phase before FFT (same as your original flow)
        ef_bfp = ef_bfp * torch.exp(1j * (-phase_lateral + phase_lateral_sub_pixel)).to(torch.complex64)  # * self.circ
        #ef_bfp = ef_bfp * torch.exp(1j * (actual_phase_axial - phase_axial)).to(torch.complex64) * circ_final_bfp
        ef_bfp = ef_bfp * torch.exp(1j * (actual_phase_axial).to(torch.complex64)) * circ_final_bfp /((ef_bfp_nocirc)* torch.exp(1j * (-phase_lateral).to(torch.complex64) ))
        ef_bfp = torch.where(circ_final_bfp>0.5, ef_bfp, 0)  #  removed on 12/17/2025 - probably it is better this way!


        psf_field = torch.fft.fftshift( torch.fft.fftn(torch.fft.ifftshift(ef_bfp, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))
        psf_on_axis = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ebfp_on_axis * self.circ, dim=(1, 2)), dim=(1, 2)), dim=(1, 2))

        psf = torch.abs(psf_field) ** 2

        psfs = psf / torch.sum(torch.abs(psf), dim=(1, 2), keepdims=True) * photons

        MEAN = psfs.mean()
        STD = psfs.std()
        psfs = torch.where(psfs>MEAN+0*STD, psfs, 0) # are we sure?


        # blur (batched)
        g_sigma = (torch.round(0.8 * self.g_sigma, decimals=2), torch.round(1.0 * self.g_sigma, decimals=2))
        sigma = g_sigma[0]+torch.rand(1).to(self.device)*(g_sigma[1]-g_sigma[0])

        blur_kernel = 1 / (2 * pi * sigma ** 2) * ( torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2)  )
        psfs = F.conv2d(psfs.unsqueeze(1), blur_kernel.unsqueeze(0).unsqueeze(0).type_as(psfs),padding='same').squeeze(1)

        psfs = psfs / torch.sum(torch.abs(psfs), dim=(1, 2), keepdims=True) * photons


        # photon normalization
        psfs = psfs[:, self.r0:self.r0 + self.H, self.c0:self.c0 + self.W]

        if self.debug_bfp:
            ef_bfp_axial_phase_removed = ef_bfp
            ef_phase_mask_only = ef_phase_mask_only * torch.exp(-1j * phase_lateral)
            ef_phase_mask_only = torch.where(circ_phase, ef_phase_mask_only, 0)
            ef_phase_mask_only = ef_phase_mask_only * torch.exp(1j * phase_nfp.to(torch.complex64)) * circ_phase
            self._maybe_save_debug(ef_phase_mask_only, psfs, xyzps, NFPs)

        return psfs
     ''' # end 05/04/2026

def calculate_cc(output, target):
    # output: rank 3, target: rank 3
    output_mean = np.mean(output, axis=(1, 2), keepdims=True)
    target_mean = np.mean(target, axis=(1, 2), keepdims=True)
    ccs = (np.sum((output - output_mean) * (target - target_mean), axis=(1, 2)) /
           (np.sqrt(np.sum((output - output_mean) ** 2, axis=(1, 2)) * np.sum((target - target_mean) ** 2,
                                                                              axis=(1, 2))) + 1e-9))
    return ccs

'''
# added on 16/03/2026 to try force smoothness
import torch

def phasor_tv_loss(phi, aperture=None, eps=1e-8):
    """
    Total-variation-like smoothness on the complex phasor exp(i*phi).
    Wrap-safe. Penalizes rapid spatial phase changes.
    """
    if aperture is None:
        aperture = torch.ones_like(phi)

    z = torch.exp(1j * phi) * aperture  # complex

    dx = z[:, 1:] - z[:, :-1]   # (N, N-1)
    dy = z[1:, :] - z[:-1, :]   # (N-1, N)

    # L2 TV (smooth, stable)
    loss = (dx.abs()**2).mean() + (dy.abs()**2).mean()
    return loss


def phasor_laplacian_loss(phi, aperture=None):
    """
    Stronger smoothness: penalize Laplacian of phasor.
    """
    if aperture is None:
        aperture = torch.ones_like(phi)

    z = torch.exp(1j * phi) * aperture

    # 5-point Laplacian
    lap = (-4*z
           + torch.roll(z,  1, dims=0) + torch.roll(z, -1, dims=0)
           + torch.roll(z,  1, dims=1) + torch.roll(z, -1, dims=1))
    return (lap.abs()**2).mean()
# end
'''
def phase_retrieval(param_dict, pr_dict, fig_flag=True):
    device = param_dict['device']

    nfps = np.asarray(param_dict['nfps'], dtype=np.float32)
    Z = len(nfps)

    # ----------------------------
    # Collect stacks: on-axis + off-axis
    # ----------------------------
    stacks = []

    # on-axis stack (x=y=0)
    zstack_on = io.imread(pr_dict['zstack_file_path']).astype(np.float32)
    stacks.append(("onaxis", zstack_on, 0.0, 0.0))

    # off-axis stacks (if provided)
    if 'offaxis_zstack_files' in param_dict and len(param_dict['offaxis_zstack_files']) > 0:
        r0, c0 = param_dict['centralBeadCoordinates_pixel']
        ps_cam = float(param_dict['ps_camera'])
        M = float(param_dict['M'])

        for f, (rr, cc) in zip(param_dict['offaxis_zstack_files'], param_dict['offaxis_coords_pixel']):
            zst = io.imread(f).astype(np.float32)
            dx_pix = float(cc) - float(c0)
            dy_pix = float(rr) - float(r0)

            # ✅ correct physical conversion (sample-plane um)
            x_um = dx_pix * (ps_cam / M)
            y_um = dy_pix * (ps_cam / M)

            stacks.append((os.path.splitext(os.path.basename(f))[0], zst, x_um, y_um))

    # ----------------------------
    # Normalize and pack into one training batch
    # ----------------------------
    y_list = []
    xyz_list = []
    nfp_list = []
    bead_id_list = []
    stack_index = 0

    # ori added on 27/01/2026
    is_onaxis_list = []  # <-- add

    for name, zst, x_um, y_um in stacks:
        stack_index += 1
        bead_id = stack_index - 1  # added on 15/03/2026
        if zst.shape[0] != Z:
            raise ValueError(
                f"{name}: Z mismatch. stack has {zst.shape[0]} but nfps has {Z}"
            )

        # ---------- ORIGINAL PR BACKGROUND CLEANUP ----------
        corner_size = max(7, int(0.1 * zst.shape[1]))

        patches = np.concatenate(
            (
                np.concatenate(
                    (zst[:, :corner_size, :corner_size],
                     zst[:, :corner_size, -corner_size:]),
                    axis=2
                ),
                np.concatenate(
                    (zst[:, -corner_size:, :corner_size],
                     zst[:, -corner_size:, -corner_size:]),
                    axis=2
                ),
            ),
            axis=1
        )

        means = np.mean(patches, axis=(1, 2), keepdims=True)
        stds = np.std(patches, axis=(1, 2), keepdims=True)

        zst = zst - means
        mask = (zst > stds*1)

        struct = ndimage.generate_binary_structure(2, 1)
        mask = np.array([
            ndimage.binary_dilation(
                ndimage.binary_erosion(mask[i], struct),
                struct
            )
            for i in range(mask.shape[0])
        ], dtype=np.float32)

        zst = zst * mask
        # ---------- END CLEANUP ----------
        # Autocorrelation
        '''# after:
        # zst = zst * mask

        if pr_dict.get("recenter_offaxis", True) and (name != "onaxis"):
            zst, shifts = recenter_stack_per_slice(zst, ref_mode="midz", upsample=10)
        #    if name == "onaxis":
        #        print(f"[recenter] {name}: dy={dy:.2f}px dx={dx:.2f}px")'''
        # end autocorrelation


        # normalize AFTER cleanup (as in original PR)
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)
        zst = np.clip(zst, 0.0, None).astype(np.float32)
        z_photons = np.sum(zst, axis=(1, 2)).astype(np.float32)
        for zi in range(Z):
            #y_list.append(zst[zi])
            #y_list.append(zst[zi]/z_photons[Z//2])  # normalize according to center

            #if stack_index == 1:
            #norm_factor = z_photons[Z//2]
            norm_factor = z_photons[zi]
            y_list.append(zst[zi] / norm_factor)  # normalize according to center
            #xyz_list.append([x_um, y_um, 0.0, float(z_photons[zi])])  # <-- photons restored
            xyz_list.append([x_um, y_um, 0.0, 1.0])
            nfp_list.append(float(nfps[zi]))
            bead_id_list.append(bead_id)
            is_onaxis_list.append(name == "onaxis")  # <-- add

        # end addition by ori 27/01/2026

        ''' # replaced on 27/01/2026
        
        for name, zst, x_um, y_um in stacks:
            if zst.shape[0] != Z:
                raise ValueError(f"{name}: Z mismatch. stack has {zst.shape[0]} but nfps has {Z}")
    
                # ---------- ORIGINAL PR BACKGROUND CLEANUP ----------
                corner_size = max(7, int(0.1 * zst.shape[1]))
    
                patches = np.concatenate(
                    (
                        np.concatenate((zst[:, :corner_size, :corner_size],
                                        zst[:, :corner_size, -corner_size:]), axis=2),
                        np.concatenate((zst[:, -corner_size:, :corner_size],
                                        zst[:, -corner_size:, -corner_size:]), axis=2),
                    ),
                    axis=1
                )
    
                means = np.mean(patches, axis=(1, 2), keepdims=True)
                stds = np.std(patches, axis=(1, 2), keepdims=True)
    
                zst = zst - means
                mask = (zst > stds)
    
                struct = ndimage.generate_binary_structure(2, 1)
                mask = np.array([
                    ndimage.binary_dilation(
                        ndimage.binary_erosion(mask[i], struct),
                        struct
                    ) for i in range(mask.shape[0])
                ], dtype=np.float32)
    
                zst = zst * mask
                # ---------- END CLEANUP ----------
                '''
        # normalize AFTER masking
        #zst = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)



    y_true = torch.from_numpy(np.stack(y_list, 0)).to(device)  # [B,H,W]
    xyzps = torch.from_numpy(np.asarray(xyz_list, np.float32)).to(device)  # [B,4]

    # NFP per sample (constant)
    #NFPs = torch.full((xyzps.shape[0],), float(param_dict['NFP']), device=device)  # removed on 26/01/2026
    NFPs = torch.tensor(np.asarray(nfp_list, np.float32), device=device)
    bead_ids = torch.tensor(np.asarray(bead_id_list, np.int64), device=device)  # added on 15/03/2026
    is_onaxis = torch.tensor(is_onaxis_list, device=device)  # [B] bool

    # ----------------------------
    # Build PR model (now includes d via ASM)
    # ----------------------------

    # zstack is numpy (Z,Hroi,Wroi)
    Hroi, Wroi = y_true.shape[-2], y_true.shape[-1]  # <-- ALWAYS matches the training target
    param_dict['H'] = int(Hroi)
    param_dict['W'] = int(Wroi)
    '''
    params_pr = dict(param_dict)
    params_pr['H'] = int(Hroi)
    params_pr['W'] = int(Wroi)
    '''
    # ori's edit from 26/01/2026 for improved pr with displacement
    params_pr = dict(param_dict)
    params_pr['H'] = int(Hroi)
    params_pr['W'] = int(Wroi)

    # bounds for d:
    params_pr['d_min_um'] = float(pr_dict.get("d_bounds_um", (15000, 30000.0*1))[0])
    params_pr['d_max_um'] = float(pr_dict.get("d_bounds_um", (15000, 30000.0*1))[1])
    # initial d:
    params_pr['mask_offset_in_um'] = float(param_dict.get("mask_offset_in_um", 0.0))
    # end ori's edit from 26/01/2026 for improved pr with displacement

    im_model = ImModel_pr(params_pr).to(device)

    im_model.train()

    # ori's edit from 26/01/2026 for improved pr with displacement
    opt = torch.optim.Adam(
        [
            {'params': [im_model.phase_mask], 'lr': float(pr_dict.get("lr_phase", 100000 *pr_dict['learning_rate']))},
            #{'params': [im_model.phase_mask], 'lr': float(pr_dict.get("lr_phase", pr_dict['learning_rate']))},
            {'params': [im_model.g_sigma], 'lr': float(pr_dict.get("lr_sigma", 5 * 0*pr_dict['learning_rate']))},
            {'params': [im_model.d_raw], 'lr': float(pr_dict.get("lr_d", 500 * pr_dict['learning_rate']))},
        ],
        betas=(0.9, 0.99)
    )
    ''' used when beads where in air 
       opt = torch.optim.Adam(
        [
            {'params': [im_model.phase_mask], 'lr': float(pr_dict.get("lr_phase", 1000000 *pr_dict['learning_rate']))},
            #{'params': [im_model.phase_mask], 'lr': float(pr_dict.get("lr_phase", pr_dict['learning_rate']))},
            {'params': [im_model.g_sigma], 'lr': float(pr_dict.get("lr_sigma", 0.1 * pr_dict['learning_rate']))},
            {'params': [im_model.d_raw], 'lr': float(pr_dict.get("lr_d", 500 * pr_dict['learning_rate']))},
        ],
        betas=(0.9, 0.99)
    )'''

    ccs = []
    # added on 15/03/2026 to make pr robust to small defocus
    fine_defocus_range_um = float(pr_dict.get("fine_defocus_range_um", 0.6))
    fine_defocus_step_um = float(pr_dict.get("fine_defocus_step_um", 0.1))
    max_shift_px = int(pr_dict.get("max_shift_px", 10))

    delta_candidates = np.arange(
        -fine_defocus_range_um,
        fine_defocus_range_um + 0.5 * fine_defocus_step_um,
        fine_defocus_step_um,
        dtype=np.float32
    )
    # end
    for epoch in range(pr_dict['epochs']):
        opt.zero_grad()
        apply_off_axis_space_invariance = True

        if not apply_off_axis_space_invariance:
            pred = im_model(xyzps, NFPs)  #original
            loss = F.mse_loss(pred, y_true)  #original
            # added on 15/03/2026 for small defocus robustness in pr
        else:
            # --------------------------------------------------
            # bead-wise robust alignment:
            # 1) one fixed shift per bead across z
            # 2) one fixed fine-defocus offset per bead across z
            # --------------------------------------------------
            with torch.no_grad():
                y_aligned = y_true.clone()
                nfp_offsets = torch.zeros_like(NFPs)

                unique_beads = torch.unique(bead_ids)

                for bid in unique_beads.tolist():
                    idx = torch.where(bead_ids == bid)[0]

                    # keep on-axis fixed
                    if bool(is_onaxis[idx[0]].item()):
                        continue

                    target_bead = y_true[idx]  # [Z,H,W]

                    best_loss = None
                    best_dd = 0.0
                    best_target_shifted = target_bead.clone()

                    for dd in delta_candidates:
                        nfp_cand = NFPs[idx] + float(dd)
                        pred_cand = im_model(xyzps[idx], nfp_cand)  # [Z,H,W] removed on 15/03/2026



                        target_shifted = target_bead.clone()

                        # allow a different shift for every z slice
                        for zi in range(pred_cand.shape[0]):
                            a = pred_cand[zi]
                            b = target_bead[zi]

                            dy, dx = phasecorr_shift_int(a, b, max_shift_px=max_shift_px)

                            b1 = torch.roll(b, shifts=(dy, dx), dims=(0, 1))
                            b2 = torch.roll(b, shifts=(-dy, -dx), dims=(0, 1))
                            if cc_score(a, b2) > cc_score(a, b1):
                                b1 = b2

                            target_shifted[zi] = b1

                        eps = 1e-12
                        pred_cand_n = pred_cand / (pred_cand.sum(dim=(1, 2), keepdim=True) + eps)
                        cand_loss = F.mse_loss(pred_cand_n, target_shifted).item()

                        if (best_loss is None) or (cand_loss < best_loss):
                            best_loss = cand_loss
                            best_dd = float(dd)
                            best_target_shifted = target_shifted.clone()

                    nfp_offsets[idx] = best_dd
                    y_aligned[idx] = best_target_shifted
            ''' replaced to make shift invariant per slice rather than per bead
            with torch.no_grad():
                y_aligned = y_true.clone()
                nfp_offsets = torch.zeros_like(NFPs)

                unique_beads = torch.unique(bead_ids)

                for bid in unique_beads.tolist():
                    idx = torch.where(bead_ids == bid)[0]

                    # on-axis bead: keep nominal NFP, no shift search
                    if bool(is_onaxis[idx[0]].item()):
                        continue

                    best_loss = None
                    best_dd = 0.0
                    best_shift = (0, 0)

                    target_bead = y_true[idx]  # [Z,H,W]

                    for dd in delta_candidates:
                        nfp_cand = NFPs[idx] + float(dd)
                        pred_cand = im_model(xyzps[idx], nfp_cand)  # [Z,H,W]

                        # one shift for the whole bead stack:
                        # use sum over z to estimate a single robust shift
                        a_ref = pred_cand.sum(dim=0)  # [H,W]
                        b_ref = target_bead.sum(dim=0)  # [H,W]

                        dy, dx = phasecorr_shift_int(a_ref, b_ref, max_shift_px=max_shift_px)

                        # sign ambiguity: test both directions on the summed image
                        b_ref_1 = torch.roll(b_ref, shifts=(dy, dx), dims=(0, 1))
                        b_ref_2 = torch.roll(b_ref, shifts=(-dy, -dx), dims=(0, 1))
                        if cc_score(a_ref, b_ref_2) > cc_score(a_ref, b_ref_1):
                            dy, dx = -dy, -dx

                        target_shifted = torch.roll(target_bead, shifts=(dy, dx), dims=(1, 2))

                        eps = 1e-12
                        pred_cand_n = pred_cand / (pred_cand.sum(dim=(1, 2), keepdim=True) + eps)
                        cand_loss = F.mse_loss(pred_cand_n, target_shifted).item()

                        if (best_loss is None) or (cand_loss < best_loss):
                            best_loss = cand_loss
                            best_dd = float(dd)
                            best_shift = (int(dy), int(dx))

                    # save best bead-wise alignment
                    nfp_offsets[idx] = best_dd
                    y_aligned[idx] = torch.roll(
                        target_bead,
                        shifts=best_shift,
                        dims=(1, 2)
                    )
                    ''' # replaced

            # forward again WITH grad, using the chosen per-bead fine defocus
            pred = im_model(xyzps, NFPs + nfp_offsets)

            eps = 1e-12
            pred_n = pred / (pred.sum(dim=(1, 2), keepdim=True) + eps)
            loss = F.mse_loss(pred_n, y_aligned)
            '''
            forceSmoothness = True
            if not forceSmoothness:
                loss = F.mse_loss(pred_n, y_aligned)
            else:
                # added force smoothness on 16/03/2026
                loss_data = F.mse_loss(pred_n, y_aligned)
                # aperture: only regularize inside the region you care about
                aperture = (im_model.circ_scaled > 0.5).to(im_model.phase_mask.dtype) if hasattr(im_model,"circ_scaled") else None
                lam = float(pr_dict.get("lambda_mask_smooth",5e-8))  # start small, e.g. 1e-4

                #loss_reg = phasor_tv_loss(im_model.phase_mask, aperture=aperture)
                loss_reg = phasor_laplacian_loss(im_model.phase_mask, aperture=aperture)
                #if epoch % 10 == 0:
                    #print("loss_data", float(loss_data), "loss_reg", float(loss_reg), "lam*reg", float(lam * loss_reg))
                loss = loss_data + lam * loss_reg
                #end
            '''
            # end 15/03/2026
        ''' removed on 15/03/2026 to add small defocus robustness in PR
        else:
            # added on 29/01/2026 - shift lateral for off axis cases
            pred = im_model(xyzps, NFPs)

            # --- translation-invariant loss for OFF-AXIS only ---
            y_aligned = y_true  # default: no change

            align_offaxis = True
            if align_offaxis:
                with torch.no_grad():
                    y_aligned = y_true.clone()

                    off_idx = torch.where(~is_onaxis)[0]
                    for i in off_idx.tolist():
                        a = pred[i].detach()  # [H,W]
                        b = y_true[i]  # [H,W]

                        dy, dx = phasecorr_shift_int(a, b, max_shift_px=10)

                        # handle sign ambiguity safely: try both directions, pick higher CC
                        b1 = torch.roll(b, shifts=(dy, dx), dims=(0, 1))
                        b2 = torch.roll(b, shifts=(-dy, -dx), dims=(0, 1))
                        if cc_score(a, b2) > cc_score(a, b1):
                            b1 = b2

                        y_aligned[i] = b1

            eps = 1e-12
            pred_n = pred / (pred.sum(dim=(1, 2), keepdim=True) + eps)  # unit-sum prediction
            loss = F.mse_loss(pred_n, y_aligned)
            #loss = F.mse_loss(pred, y_aligned)
            ''' # end 03/15/2026

        ''' 
        pred = im_model(xyzps, NFPs)
        with torch.no_grad():
            dy, dx = estimate_shift_phasecorr(pred, y_true)  # shift target to match pred
            y_aligned = torch.stack([torch.roll(y_true[i], shifts=(int(dy[i]), int(dx[i])), dims=(-2, -1))
                                     for i in range(y_true.shape[0])], dim=0)

        loss = F.mse_loss(pred, y_aligned)
        '''
        # keep some MSE to prevent "degenerate" solutions
        #loss = 0.2 * loss_mse + 0.8 * loss_ac
        #loss = 0.0 * loss_mse + 1.0 * loss_ac

        loss.backward()

        if epoch == 0:
            print("d_um:", im_model.d_um().detach().item())
            print("grad(d_raw):", None if im_model.d_raw.grad is None else im_model.d_raw.grad.detach().item())

        opt.step()
        Visualize_mask = True
        # visualization
        if Visualize_mask:
            if epoch % 10 == 0:
                with torch.no_grad():
                    mask = im_model.phase_mask.detach().cpu().numpy()

                    # wrap phase to [-pi, pi] for visualization
                    mask_wrapped = np.angle(np.exp(1j * mask))

                    plt.figure(figsize=(4, 4))
                    plt.imshow(mask, cmap="twilight")
                    plt.colorbar()
                    plt.title(f"Phase mask, epoch {epoch}")
                    plt.tight_layout()

                    path2save = 'phase_retrieval_with_displacement_iteration'
                    if not (os.path.isdir(path2save)):
                        os.mkdir(path2save)
                    plt.savefig(os.path.join(path2save, 'iteration_' + str(epoch) +  '.jpg'), bbox_inches='tight', dpi=300)
                    plt.clf()
                    # End visualization

        # keep sigma sane (optional but helps)
        with torch.no_grad():
            im_model.g_sigma.clamp_(min=1e-3, max=20.0)

        # monitor
        with torch.no_grad():
            pred2 = im_model(xyzps, NFPs)
            cc = calculate_cc(pred2.detach().cpu().numpy(), y_true.detach().cpu().numpy())
            ccs.append(cc)

            if (epoch % 10) == 0:
                d_now = float(im_model.d_um().detach().cpu().item())
                print(
                    f"[PR] epoch {epoch:4d} loss={float(loss.item()):.6g}  d={d_now:.2f} um  g_sigma={float(im_model.g_sigma.item()):.4f}")

    # end ori's edit from 26/01/2026 for improved pr with displacement

    ''' removed on 26/01/2026
    opt = torch.optim.Adam([im_model.phase_mask, im_model.g_sigma], lr=pr_dict['learning_rate'])

    # ----------------------------
    # Coordinate descent for d
    # ----------------------------
    d = float(param_dict.get("mask_offset_in_um", 0.0))
    d_step = float(pr_dict.get("d_step_um", 1000.0))
    d_update_every = int(pr_dict.get("d_update_every", 25))
    d_min, d_max = pr_dict.get("d_bounds_um", (0.0, 70000.0))

    ccs = []

    def loss_for_current_d():
        pred = im_model(xyzps, NFPs)
        return F.mse_loss(pred, y_true)

    for epoch in range(pr_dict['epochs']):
        opt.zero_grad()
        loss = loss_for_current_d()
        loss.backward()
        opt.step()

        # CC for monitoring
        with torch.no_grad():
            pred = im_model(xyzps, NFPs)
            cc = calculate_cc(pred.detach().cpu().numpy(), y_true.detach().cpu().numpy())
            ccs.append(cc)

        # --- update d by small 1D search ---
        if (epoch + 1) % d_update_every == 0:
            with torch.no_grad():
                candidates = np.clip(
                    d + np.array([-2, -1, 0, 1, 2], dtype=np.float32) * d_step,
                    d_min, d_max
                )

                best_d = d
                best_loss = float("inf")
                for dd in candidates:
                    im_model.mask_offset_in_um = float(dd)
                    L = float(loss_for_current_d().item())
                    if L < best_loss:
                        best_loss = L
                        best_d = float(dd)

                d = best_d
                im_model.mask_offset_in_um = d
                d_step *= 0.7  # refine step gradually
    '''
    # save final values back
    param_dict['mask_offset_in_um'] = float(im_model.d_um().detach().cpu().item())
    print(f"[PR] done. best d = {param_dict['mask_offset_in_um']:.2f} um")
    phase_mask = im_model.phase_mask.detach().cpu().numpy()
    g_sigma = float(im_model.g_sigma.detach().cpu().numpy())

    #print(f"[PR] done. best d = {d:.1f} um")

    # ----------------------------
    # SAVE OUTPUTS (after PR is done)
    # ----------------------------
    save_dir = pr_dict.get("save_dir", os.path.join(os.getcwd(), "phase_retrieval_outputs"))
    os.makedirs(save_dir, exist_ok=True)

    # save mask + scalar params
    np.save(os.path.join(save_dir, "phase_mask.npy"), phase_mask)
    with open(os.path.join(save_dir, "g_sigma_and_d.txt"), "w") as f:
        f.write(f"g_sigma = {g_sigma}\n")
        f.write(f"mask_offset_in_um (d) = {float(param_dict['mask_offset_in_um'])}\n")

    # helper: float stack -> uint16 for viewing
    def _to_u16(st):
        st = st.astype(np.float32)
        out = np.zeros_like(st, dtype=np.uint16)
        for i in range(st.shape[0]):
            mx = float(st[i].max())
            if mx > 0:
                out[i] = (np.clip(st[i] / mx, 0, 1) * 65535.0).astype(np.uint16)
        return out

    im_model.eval()
    cnt = -1
    with torch.no_grad():
        for name, zst, x_um, y_um in stacks:
            cnt+=1
            # build xyz for this bead (Z samples)
            #xyz_bead = np.stack([[x_um, y_um, float(nfps[zi]), 1.0] for zi in range(Z)], axis=0).astype(np.float32)
            xyz_bead = np.stack([[x_um, y_um, float(nfps[zi])*0, 1.0] for zi in range(Z)], axis=0).astype(np.float32)
            xyz_bead_t = torch.from_numpy(xyz_bead).to(device)
            NFPs_bead = torch.full((Z,), float(param_dict['NFP']), device=device)
            #NFPs_bead = torch.full((Z,), float(param_dict['NFP'])*0, device=device)

            #pred = im_model(xyz_bead_t, NFPs_bead).detach().cpu().numpy()  # [Z,H,W]
            pred = im_model(xyz_bead_t, torch.tensor(nfps).to(device)).detach().cpu().numpy()  # [Z,H,W]
            exp = zst / (np.sum(zst, axis=(1, 2), keepdims=True) + 1e-12)   # [Z,H,W] (same norm as training)

            exp_u16 = _to_u16(exp)
            sim_u16 = _to_u16(pred)

            io.imsave(os.path.join(save_dir, f"exp_stack_{cnt}_{name}.tif"), exp_u16, check_contrast=False)
            io.imsave(os.path.join(save_dir, f"sim_stack_{cnt}_{name}.tif"), sim_u16, check_contrast=False)

            # montage per z: left=exp, right=sim
            Z0, H0, W0 = exp_u16.shape
            montage = np.zeros((Z0, H0, 2 * W0), dtype=np.uint16)
            montage[:, :, :W0] = exp_u16
            montage[:, :, W0:] = sim_u16
            io.imsave(os.path.join(save_dir, f"montage_{cnt}_{name}.tif"), montage, check_contrast=False)

    # --- Final per-bead debug dump (central z only) ---
    # Put debug outputs inside the same phase_retrieval_outputs folder
    im_model.debug_bfp = True
    im_model.debug_every = 1
    im_model.debug_dir = os.path.join(save_dir, "per_bead_phase")
    im_model._debug_call_idx = 0

    # number of beads you used in PR (onaxis + offaxis)
    num_beads = len(stacks)
    im_model.debug_max_emitters = num_beads

    # Optional: give names to beads (so folders aren't emitter_000, emitter_001...)
    im_model.debug_names = [name for (name, _, _, _) in stacks]

    # Take ONLY the central z/NFP slice from each bead:
    zi = Z // 2
    idxs = [b * Z + zi for b in range(num_beads)]  # assumes your packing is bead-major then z

    xyz_mid = xyzps[idxs]

    #xyz_mid[:,0]  =  xyz_mid[:,0] * 100
    #xyz_mid[:,1]  =  xyz_mid[:,1] * 100

    NFPs_mid = NFPs[idxs]

    with torch.no_grad():
        _ = im_model(xyz_mid, NFPs_mid)  # triggers _maybe_save_debug once

    return phase_mask, g_sigma, ccs

    # end ori's edit from 26/01/2026 - phase retrival with d


    ''' removed on 26/01/2026
    def phase_retrieval(param_dict, pr_dict, fig_flag=True):
        # ori's edit from 26/01/2026 - phase retrival with d
        device = param_dict['device']
        # ori's edit
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
        print(f'device used (phase_retrieval): {device}')
        #end ori's edit
        file_path = pr_dict['zstack_file_path']
    
        nfps = pr_dict['nfps']
        r_bead = pr_dict['r_bead']
        epoch_num = pr_dict['epoch_num']
        loss_label = pr_dict['loss_label']
    
        # read the zstack and set the image size for the imaging model
        zstack = io.imread(file_path)  # axis0 -- z position
    
        corner_size = max(7, int(0.1*zstack.shape[1]))
    
        param_dict['H'], param_dict['W'] = zstack.shape[1], zstack.shape[2]
    
        # estimate Gaussian noise: mean and std
        patches = np.concatenate(
            (np.concatenate((zstack[:, :corner_size, :corner_size], zstack[:, :corner_size, -corner_size:]), axis=2),
             np.concatenate((zstack[:, -corner_size:, :corner_size], zstack[:, -corner_size:, -corner_size:]), axis=2)),
            axis=1)
        means = np.mean(patches, axis=(1, 2), keepdims=True)
        stds = np.std(patches, axis=(1, 2), keepdims=True)
        # filtering mask
        zstack = zstack - means
        mask = (zstack > stds)
        # erode and dimate the mask
        struct = ndimage.generate_binary_structure(2, 1)  # raius 1 or 2
        mask = [ndimage.binary_dilation(ndimage.binary_erosion(mask[i, :, :], struct), struct) for i in range(mask.shape[0])]
        # clean zstack
        zstack = zstack * np.array(mask)
        z_photons = np.sum(zstack, axis=(1, 2))
    
        # Nuke any cached phase mask so the model will use a fresh one for the current N
        param_dict.pop('phase_mask', None)
        im_model_bead = ImModelBead(param_dict)
        im_model_bead.circ = im_model_bead.circ_NA  # include SAF (use full NA support)
    
        print(f'BFP aperture in pixel unit: {int(np.round(im_model_bead.pn_pupil))}/{im_model_bead.N}.')
    
        im_model_bead.phase_mask.requires_grad_(True)
        im_model_bead.g_sigma.requires_grad_(True)
    
        num_zs = zstack.shape[0]
        xyzps = np.zeros((num_zs, 4))
        xyzps[:, 3] = z_photons
        xyzps = torch.tensor(xyzps, device=device)
        nfps_np = nfps.copy()
        nfps = torch.tensor(nfps, device=device).unsqueeze(1)
    
        y = torch.tensor(zstack, device=device)  # measurement
        optimizer = torch.optim.Adam([{'params': im_model_bead.phase_mask, 'lr': 0.1},
                                      {'params': im_model_bead.g_sigma, 'lr': 0.06}
                                      ])
        epoch_loss = []
        for i in range(100):
            fx = im_model_bead(xyzps, nfps)
    
            loss = torch.nn.functional.mse_loss(fx, y)  # mse
            # loss = torch.mean(fx-y*torch.log(fx))  # gauss log likelihood
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
    
        optimizer = torch.optim.Adam([{'params': im_model_bead.phase_mask, 'lr': 0.02},
                                      {'params': im_model_bead.g_sigma, 'lr': 0.01}
                                      ])
        for i in range(100):
            with torch.no_grad():
                fx = im_model_bead(xyzps, nfps)
                model_psfs = fx.detach().cpu().numpy()
                ccs = calculate_cc(zstack, model_psfs)
                ids = np.argsort(ccs)[:5]
    
            fx = im_model_bead(xyzps[ids], nfps[ids])
            loss = torch.nn.functional.mse_loss(fx, y[ids])
            # loss = torch.mean(fx-y[ids]*torch.log(fx))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
    
        with torch.no_grad():
            fx = im_model_bead(xyzps, nfps)
            model_psfs = fx.detach().cpu().numpy()
            ccs = calculate_cc(zstack, model_psfs)
    
        mask_rec = im_model_bead.phase_mask.detach().cpu().numpy()
        mask_rec_no_wrapping = im_model_bead.phase_mask.detach().cpu().numpy()
        mask_rec = np.angle(np.exp(1j * mask_rec))
        g_sigma = im_model_bead.g_sigma.detach().item()
        psfs_np = model_psfs
        phase_mask = mask_rec
        #phase_mask = mask_rec_no_wrapping
    
    
        if fig_flag:
            fig = plt.figure(1, figsize=(7, 4))
            gs = fig.add_gridspec(3, 7)
    
            ax = fig.add_subplot(gs[:2, :3])
            maskplot = ax.imshow(phase_mask)
            plt.colorbar(maskplot)
            ax.set_title('retrieved phase')
    
            ax = fig.add_subplot(gs[0, 3:])
            ids = (0, (zstack.shape[0]-1) // 4, (zstack.shape[0]-1) // 2, ((zstack.shape[0]-1)//4)*3, -1)
            im_demo = np.concatenate((zstack[ids[0]], zstack[ids[1]], zstack[ids[2]], zstack[ids[3]],
                                      zstack[ids[4]]), axis=1)
            ax.imshow(im_demo)
            ax.axis('off')
            ax.set_title('exp')
    
            ax = fig.add_subplot(gs[1, 3:])
            im_demo = np.concatenate((psfs_np[ids[0]], psfs_np[ids[1]], psfs_np[ids[2]], psfs_np[ids[3]],
                                      psfs_np[ids[4]]), axis=1)
            ax.imshow(im_demo)
            ax.axis('off')
            ax.set_title('model')
    
            ax = fig.add_subplot(gs[2, :])
            ax.plot(nfps_np, ccs)
            ax.set_xlabel('NFP [um]')
            ax.set_ylabel('CC')
            # ax.set_title('model accuracy')
    
            plt.savefig('phase_retrieval_results.jpg', bbox_inches='tight', dpi=300)
            plt.clf()
            # print(f'phase retrieval results: phase_retrieval_results.jpg')
    
        return phase_mask, g_sigma, ccs

    '''


def show_z_psf(param_dict):
    model = ImModel(param_dict)
    model.model_demo(np.linspace(param_dict['zrange'][0], param_dict['zrange'][1], 5))  # check PSFs


def background_removal(im_folder, num=100):
    save_folder = im_folder + '_br'  # where to save the images after background removal
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    im_files = sorted(os.listdir(im_folder))  # make sure the names are sortable
    n_ims = len(im_files)
    if n_ims > num:
        pointer = 0
        for i in range(n_ims//num):
            im_names = [im_files[pointer+j] for j in range(num)]
            im_stack = [io.imread(os.path.join(im_folder, im_files[pointer+j])) for j in range(num)]
            pointer += num
            im_stack = np.array(im_stack)
            im_stack = im_stack-np.min(im_stack, axis=0)

            for j in range(num):  # save
                io.imsave(os.path.join(save_folder, im_names[j]), im_stack[j], check_contrast=False)
                if j/100 == j//100 :
                    print("Saved file: " + os.path.join(im_names[j]))

        # remainder of n_ims/num
        im_stack = [io.imread(os.path.join(im_folder, im_files[-j])) for j in range(num)]
        im_stack = np.array(im_stack)
        im_min = np.min(im_stack, axis=0)
        for j in range(pointer, n_ims):
            im = io.imread(os.path.join(im_folder, im_files[j]))
            im = im-im_min
            io.imsave(os.path.join(save_folder, im_files[j]), im, check_contrast=False)

    else:
        im_stack = [io.imread(os.path.join(im_folder, im_files[j])) for j in range(n_ims)]
        im_stack = np.array(im_stack)
        im_stack = im_stack-np.min(im_stack, axis=0)
        for j in range(n_ims):
            io.imsave(os.path.join(save_folder, im_files[j]), im_stack[j], check_contrast=False)

    return save_folder

def mu_std_p(param_dict, noise_dict):

    im_br_folder = param_dict['im_br_folder']
    num = noise_dict['num_ims']
    # noise_roi = noise_dict['noise_roi']
    snr_roi = noise_dict['snr_roi']
    max_pv = noise_dict['max_pv']

    im_names = sorted(os.listdir(im_br_folder))
    im_names = im_names[-num:]  # at the end of the video, probably with sparse molecules. It's ok if num>len(im_names)
    ims = np.array([io.imread(os.path.join(im_br_folder, im_name)) for im_name in im_names])

    ims = ims[:, snr_roi[0]:snr_roi[2], snr_roi[1]:snr_roi[3]]

    max_map = np.max(ims, axis=0)
    mean_map = np.mean(ims, axis=0)

    r_idx, c_idx = np.unravel_index(np.argmin(mean_map), mean_map.shape)
    bg_pixel = ims[:, r_idx, c_idx]
    mu, std = np.mean(bg_pixel), np.std(bg_pixel)

    if max_pv == 0:
        exp_maxv = np.max(max_map)  # if max_pv is 0 in the GUI
    else:
        exp_maxv = max_pv  # detect max_pv in the selected ROI

    print(f'Detected MPV: {exp_maxv}.')

    model = ImModelBase(param_dict)

    photon_count = 1e4
    xyzps = np.array([[0, 0, (param_dict['zrange'][0]+param_dict['zrange'][1])/2, photon_count]])  # take the middle z
    xyzps = torch.from_numpy(xyzps).to(param_dict['device'])
    ims = model.get_psfs(xyzps).cpu().numpy()

    maxvs = np.max(ims, axis=(1, 2))
    mv = np.mean(maxvs) + mu  # model.get_psfs doesn't include noise yet
    p = photon_count/mv * exp_maxv

    return mu, std, p, exp_maxv


def training_data_func(param_dict):

    device = param_dict['device']
    # ori's edit
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
    print(f'device used (training_data_func): {device}')
    #end ori's edit

    # imaging model

    model = ImModelTraining(param_dict)
    #ori's edit:'
    tmp1 = param_dict['H']
    tmp2 = param_dict['W']
    tmp3 = param_dict['HH']
    tmp4 = param_dict['WW']

    #param_dict['H'] = 1400
    #param_dict['W'] = 1400
    ##param_dict['HH'] = 1400 #should keep? ori's edit 2025/12/06
    ##param_dict['WW'] = 1400
    #end ori's edit
    # sampling model
    sampling = Sampling(param_dict)
    sampling.show_volume()  # plot volume
    #ori's edit:'
    #param_dict['H'] = tmp1
    #param_dict['W'] = tmp2
    param_dict['HH'] = tmp3
    param_dict['WW'] = tmp4
    #end ori's edit
    # labels_dict for training
    labels_dict = {}
    # start
    td_folder = param_dict['td_folder']
    if os.path.exists(td_folder):  # delete the directory if it exists
        shutil.rmtree(td_folder)
    x_folder = os.path.join(td_folder, 'x')
    os.makedirs(x_folder)  # make the folder for training data
    # ori's edit:
    us = param_dict['us_factor']
    #Hc, Wc = param_dict['camera_size_px']  # e.g. (1400, 1400)
    Hc = param_dict['H']
    Wc = param_dict['W']
    num_tiles = 8 #8 #10/10*1  # your current setting
    #num_tiles = param_dict.get("num_tiles", 1)  # added on 14/12/2025

    H_new = Hc // num_tiles
    W_new = Wc // num_tiles

    # One volume_size for all samples in this dataset: the tile size
    labels_dict['volume_size'] = (param_dict['D'], int(H_new * us), int(W_new * us))
    labels_dict['us_factor'] = param_dict['us_factor']
    labels_dict['blob_r'] = sampling.blob_r
    labels_dict['blob_maxv'] = sampling.blob_maxv
    # ori's edit 12/09/2025 for tile dependence training:
    param_dict['camera_size_px'] = Hc, Wc

    labels_dict['tile_grid'] = (num_tiles, num_tiles)
    labels_dict['camera_size_px'] = param_dict['camera_size_px']


    #end ori's edit

    # 07/02/2026
    print("[TRAIN] camera_size_px in labels:", labels_dict.get("camera_size_px", None))
    print("[TRAIN] tile_grid:", labels_dict.get("tile_grid", None))
    print("[TRAIN] labels camera_size_px =", labels_dict.get("camera_size_px", None))

    # end 07/02/2026

    '''
    # labels_dict for training
    labels_dict = {}
    labels_dict['volume_size'] = (param_dict['D'], param_dict['HH'], param_dict['WW'])
    labels_dict['us_factor'] = param_dict['us_factor']
    labels_dict['blob_r'] = sampling.blob_r  # radius of each 3D blob representing an emitter in space
    labels_dict['blob_maxv'] = sampling.blob_maxv  # maximum value of blobs
    '''



    ntrain = param_dict['n_ims']
    for i in range(ntrain):
        xyzps, xyz_ids, blob3d = sampling.xyzp_batch()
        #ori's edit
        if i == 0:
            print("Train batch z range [µm]:", xyzps[:, 2].min(), xyzps[:, 2].max())
        #end ori's edit

        #im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.uint16)
        # Ori's edit
        #Hc, Wc = param_dict['camera_size_px']  # e.g. (1400, 1400)
        Hc = param_dict['H']
        Wc = param_dict['W']
        ps_xy = param_dict['ps_camera'] / param_dict['M']  # µm per pixel in image plane
        canvas = np.zeros((Hc, Wc), dtype=np.float32)
        #mask_offset_in_um = param_dict['mask_offset_in_um']
        # small-grid size (mask grid) from the model
        N = int(model.N)  # ≈203, odd
        r = N // 2

        for k in range(xyzps.shape[0]):
            #xyzp_t = np.array(xyzps[k],  copy=True)
            '''if k == 0:
                xyzps[k] = [0.0, 0.0, 0.8, 100000]
            if k == 1:
                xyzps[k] = [0.0, 0.11/2, 0.8, 100000]
            if k == 2:
                xyzps[k] = [0.0, 0.11/4, 0.8, 100000]
            if k == 3:
                xyzps[k] = [-0.11/2, 0.0, 0.8, 100000]
            if k>2:
                continue
            '''
            x_um, y_um, z_um, photons = xyzps[k]
            #print("x_um =", x_um, " y_um=", y_um)
            #x_um, y_um, z_um, photons = xyzps[k]
            # convert emitter center (µm) → canvas pixel center
            c = int(round(x_um / ps_xy + (Wc - 1) / 2))

            r0 = int(round(y_um / ps_xy + (Hc - 1) / 2))
            #if i == 0 and k < 3:
            #    print(f"[stitch] emitter{k}: x_um={x_um:.4f}, y_um={y_um:.4f} -> paste center (r0,c)={(r0, c)}")

            #patch = model.psf_patch_clean(xyzps[k])  # (N,N), float
            patch = model.psf_patch_clean(xyzps[k])  # (N,N), float

            # for debug
            #xyzps_debug = xyzps[0]
            #xyzps_debug = np.array([0.0,0.0,2.0,100000.0])
            #patch = model.psf_patch_clean(xyzps_debug)  # (N,N), float
            #xyzps_debug = np.array([1,0.0,2.0,100000.0])
            #patch += model.psf_patch_clean(xyzps_debug)  # (N,N), float
            #patch = model.psf_patch_clean(xyzps_debug[1])  # (N,N), float
            #end for debug

            #print("x_um =", x_um, " y_um=", y_um)

            #plt.imshow(patch, cmap="twilight")
            # plt.colorbar()
            #plt.title("x_um ="+ str(x_um) + " y_um="+ str(y_um))
            #plt.savefig("patch_"+str(k).zfill(3)+".png", dpi=200)

            # paste with clipping
            rr0 = max(0, r0 - r)
            rr1 = min(Hc, r0 + r + 1)
            cc0 = max(0, c - r)
            cc1 = min(Wc, c + r + 1)

            pr0 = r - (r0 - rr0)
            pr1 = r + 1 - (rr1 - r0)
            pc0 = r - (c - cc0)
            pc1 = r + 1 - (cc1 - c)

            if rr0 < rr1 and cc0 < cc1:
                canvas[rr0:rr1, cc0:cc1] += patch[pr0:N - pr1, pc0:N - pc1]
        # add noise once (keep your existing ranges)


        im = canvas
        #Background_for_shotnoise = 180  # gl: mean bg value before background removal for shot noise estimation
        #NoiseOffset = 22.0   #noise level (mean) after removal recommended 25.0

        # randomized once per CANVAS
        Background_for_shotnoise = float(np.random.uniform(140.0, 220.0))
        #NoiseOffset = float(np.random.uniform(20.0, 28.0))  # for microtubuls
        NoiseOffset = float(np.random.uniform(20.0, 30.0))  # for mitochondria
        #print("Background_for_shotnoise = " + str(Background_for_shotnoise) + "NoiseOffset = " + str(NoiseOffset))
        # ori's edit - added on 25/01/2026 to add readout noise
        read_baseline = 0.0  # noise average. keep 0
        read_std = 0  # ADU std (currently estimated by eye)
        # end ori's edit on 25/01/2026 to add readout noise

        if not param_dict['project_01']:
            im = np.random.poisson(im + Background_for_shotnoise) + NoiseOffset - Background_for_shotnoise # shot noise once # note the dark offset!
            im = np.abs(im)
            # readout baseline + spatial/read std (reuse your code)
            # ...

        numOfPatches = 8  # per axis. total number is numOfPatches x numOfPatches num_of_tiles / num_tiles
        #numOfPatches = 1 # per axis. total number is numOfPatches x numOfPatches num_of_tiles / num_tiles
        #numOfPatches = param_dict.get("num_tiles",1)
        if numOfPatches==1:
            im = np.clip(im, 0, 2 ** param_dict['bitdepth'] - 1).astype(np.uint16)
            if param_dict['project_01']:
                im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)

            x_name = str(i).zfill(5) + '.tif'
            io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
            labels_dict[x_name] = (xyz_ids, blob3d)
        #ori's edit
        else:
            #canvas2 = np.random.poisson(canvas + NoiseOffset)  # removed on 25/01/2026 to add readout noise
            #canvas2 = np.clip(canvas2, 0, 2 ** param_dict['bitdepth'] - 1).astype(np.uint16) # removed on 25/01/2026 to add readout noise

            # ori's edit on 25/01/2026 to add readout noise
            canvas2 = np.abs(np.random.poisson(canvas + Background_for_shotnoise) + NoiseOffset - Background_for_shotnoise).astype(np.float32)  # shot noise once # note the dark offset!
            #print("read_baseline = " + str(read_baseline) + "read_std = " + str(read_std))
            #canvas2 += read_baseline + np.random.randn(*canvas2.shape).astype(np.float32) * read_std
            canvas2 = np.clip(canvas2, 0, 2 ** param_dict['bitdepth'] - 1).astype(np.uint16)
            # end ori's edit on 25/01/2026 to add readout noise
            #canvas = im
            #H = param_dict['camera_size_px'][0]
            #W = param_dict['camera_size_px'][1]
            H = param_dict['H']
            W = param_dict['W']
            H_new = int(H/numOfPatches)
            W_new = int(W/numOfPatches)
            fov_inx = -1

            # (optional but useful later) remember your tiling
            param_dict['tile_grid'] = (numOfPatches, numOfPatches)
            param_dict['tile_size_px'] = (H_new, W_new)
            '''
            # added on 14 / 04 / 2026: save patches randomly
            num_crops_per_image = numOfPatches * numOfPatches
            for crop_idx in range(num_crops_per_image):
                # 1. Pick a random top-left corner that ensures the crop fits on the canvas
                y0 = np.random.randint(0, H - H_new + 1)
                x0 = np.random.randint(0, W - W_new + 1)
                y1 = y0 + H_new
                x1 = x0 + W_new

                # 2. Crop the image tile
                im = canvas2[y0:y1, x0:x1]

                # 3. Create the filename
                x_name = str(i).zfill(5) + '_FOV_' + str(crop_idx).zfill(5) + '.tif'

                # 4. Find which emitters fall inside this random crop
                mask = (
                        (xyz_ids[:, 0] >= x0) & (xyz_ids[:, 0] < x1) &
                        (xyz_ids[:, 1] >= y0) & (xyz_ids[:, 1] < y1)
                )

                # 5. Save the tile and its relative coordinates
                if np.any(mask) | 1:  # Keep your logic to save empty tiles too
                    xyz_ids_fov = xyz_ids[mask].copy()
                    blob3d_fov = blob3d[mask].copy()

                    # Rebase to tile-local indices
                    xyz_ids_fov[:, 0] -= x0
                    xyz_ids_fov[:, 1] -= y0

                    io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
                    labels_dict[x_name] = (xyz_ids_fov, blob3d_fov)
                    # end 14/04/2026
                '''
            for ii in range(0,numOfPatches):
                for jj in range(0,numOfPatches):
                    fov_inx += 1

                    H_range = range(ii * H_new, (ii + 1) * H_new)
                    W_range = range(jj * W_new, (jj + 1) * W_new)
                    # im = canvas[H_range, W_range]
                    # im = im.reshape(H_new, W_new)
                    im = canvas2[ii * H_new:(ii + 1) * H_new, jj * W_new:(jj + 1) * W_new]
                    # im = np.clip(im, 0, 2 ** param_dict['bitdepth'] - 1).astype(np.uint16)

                    x_name = str(i).zfill(5) + '_FOV_' + str(fov_inx).zfill(5) + '.tif'

                    # tile limits in pixel/voxel index space
                    y0 = ii * H_new
                    y1 = (ii + 1) * H_new
                    x0 = jj * W_new
                    x1 = (jj + 1) * W_new

                    # build one mask (x compares to W-range, y compares to H-range)
                    mask = (
                            (xyz_ids[:, 0] >= x0) & (xyz_ids[:, 0] < x1) &
                            (xyz_ids[:, 1] >= y0) & (xyz_ids[:, 1] < y1)
                    )

                    if np.any(mask) | 1 : # added on 24/12/2025 - the |1 to also save tiles without emitters (prevents false positives
                        xyz_ids_fov = xyz_ids[mask].copy()
                        blob3d_fov = blob3d[mask].copy()

                        # rebase to tile-local indices (columns=x => subtract x0; rows=y => subtract y0)
                        xyz_ids_fov[:, 0] -= x0
                        xyz_ids_fov[:, 1] -= y0

                        io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
                        # ori's edit
                        #print("Train batch z range [µm]:", xyz_ids_fov[:, 2].min(), xyz_ids_fov[:, 2].max())
                        #
                        labels_dict[x_name] = (xyz_ids_fov, blob3d_fov)



            ''' old! 
                    H_range = range(ii*H_new, (ii+1)*H_new)
                    W_range = range(jj * W_new,(jj + 1) * W_new)
                    #im = canvas[H_range, W_range]
                    #im = im.reshape(H_new, W_new)
                    im = canvas2[ii*H_new:(ii+1)*H_new, jj*W_new:(jj+1)*W_new]
                    #im = np.clip(im, 0, 2 ** param_dict['bitdepth'] - 1).astype(np.uint16)

                    x_name = str(i).zfill(5) + '_FOV_' + str(fov_inx).zfill(5) + '.tif'
                    blob3d_fov = blob3d[xyz_ids[:,0]>H_range[0] ]
                    xyz_ids_fov = xyz_ids[xyz_ids[:,0]>H_range[0] ]
                    blob3d_fov = blob3d_fov[xyz_ids_fov[:,0]<H_range[-1] ]
                    xyz_ids_fov = xyz_ids_fov[xyz_ids_fov[:,0]<H_range[-1] ]
                    blob3d_fov = blob3d_fov[xyz_ids_fov[:,1]>W_range[0] ]
                    xyz_ids_fov = xyz_ids_fov[xyz_ids_fov[:,1]>W_range[0] ]
                    blob3d_fov = blob3d_fov[xyz_ids_fov[:,1]<W_range[-1] ]
                    if np.shape(blob3d_fov)[0]>0:
                        xyz_ids_fov = xyz_ids_fov[xyz_ids_fov[:,1]<W_range[-1] ]
                        io.imsave(os.path.join(x_folder, x_name), im, check_contrast=False)
                        labels_dict[x_name] = (xyz_ids_fov, blob3d_fov)'''

        # end ori's edit



        if i % 100 == 0:#(ntrain // 10) == 0:
            print('Training image [%d / %d]' % (i + 1, ntrain))
    print('Training image [%d / %d]' % (ntrain, ntrain))

    y_file = os.path.join(td_folder, r'y.pickle')
    with open(y_file, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    param_file = os.path.join(td_folder, r'param.pickle')
    with open(param_file, 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('training images and data have been saved')

# added on 28/04/2026 - improve training images reading
def maybe_build_x_memmap(td_folder, force_rebuild=False):
    """
    Build or reuse a memmap cache for training_data/x/*.tif.
    Keeps TIFFs on disk for debugging, but training can read from one binary file.

    Returns:
        cache_info: dict with keys
            enabled, data_path, shape, dtype, ids
    """
    x_folder = os.path.join(td_folder, 'x')
    data_path = os.path.join(td_folder, 'x_memmap.dat')
    ids_path = os.path.join(td_folder, 'x_ids.npy')

    ids = sorted([f for f in os.listdir(x_folder) if f.lower().endswith('.tif')])
    if len(ids) == 0:
        raise RuntimeError(f'No TIFF files found in {x_folder}')

    # Use first TIFF to infer shape/dtype
    first_im = io.imread(os.path.join(x_folder, ids[0]))
    H, W = first_im.shape
    dtype = first_im.dtype

    rebuild = force_rebuild

    if (not os.path.exists(data_path)) or (not os.path.exists(ids_path)):
        rebuild = True
    else:
        try:
            cached_ids = np.load(ids_path, allow_pickle=True).tolist()
            if cached_ids != ids:
                rebuild = True
            else:
                expected_bytes = len(ids) * H * W * np.dtype(dtype).itemsize
                actual_bytes = os.path.getsize(data_path)
                if actual_bytes != expected_bytes:
                    rebuild = True
        except Exception:
            rebuild = True

    if rebuild:
        print(f'[memmap] building cache from TIFFs in {x_folder}')
        X = np.memmap(data_path, mode='w+', dtype=dtype, shape=(len(ids), H, W))
        for i, fname in enumerate(ids):
            if i % 1000 == 0:
                print(f'[memmap] packing [{i} / {len(ids)}]')
            im = io.imread(os.path.join(x_folder, fname))
            if im.shape != (H, W):
                raise ValueError(f'Image shape mismatch for {fname}: {im.shape} vs {(H, W)}')
            X[i] = im
        X.flush()
        np.save(ids_path, np.array(ids, dtype=object), allow_pickle=True)
        print(f'[memmap] cache saved: {data_path}')
    else:
        print(f'[memmap] reusing existing cache: {data_path}')

    cache_info = dict(
        enabled=True,
        data_path=data_path,
        shape=(len(ids), H, W),
        dtype=np.dtype(dtype).str,
        ids=ids,
    )
    return cache_info
# end 28/04/2026


def training_func(param_dict, training_dict):
    np.random.seed(66)
    torch.manual_seed(88)

    device = param_dict['device']
    # ori's edit
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
    print(f'device used (training_func): {device}')
    #end ori's edit
    torch.backends.cudnn.benchmark = True

    td_folder = param_dict['td_folder']
    path_save = param_dict['path_save']
    if not (os.path.isdir(path_save)):
        os.mkdir(path_save)

    batch_size = training_dict['batch_size']
    lr = training_dict['lr']
    num_epochs = training_dict['num_epochs']

    # ori's edit - to improve time 15/01/2026
    params_train = dict(
        batch_size=batch_size,
        #shuffle=True,  # removed on 07/02/2026
        num_workers=8,  # try 4, 8, 12 depending on CPU
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    params_validate = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


    # end ori's edit
    #params_train = {'batch_size': batch_size, 'shuffle': True}
    #params_validate = {'batch_size': batch_size, 'shuffle': True}

    # added on 28/04/2026 to change training data image reading
    x_folder = os.path.join(td_folder, 'x')

    # Build or reuse one binary cache for the TIFF tiles
    x_cache = maybe_build_x_memmap(td_folder, force_rebuild=False)

    # IMPORTANT: use the cache ID order, not raw os.listdir order
    x_list = list(x_cache['ids'])

    num_x = len(x_list)
    with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    partition = {'train': x_list[:int(num_x * 0.9)], 'validate': x_list[int(num_x * 0.9):]}
    train_ds = MyDataset(x_folder, partition['train'], labels, cache_info=x_cache)
    # end change training data image reading



    '''
    # removed on 28/04/2026 to change training data image reading
    x_folder = os.path.join(td_folder, 'x')
    x_list = os.listdir(x_folder)

    #num_tiles = 8
    #x_list = x_list[:param_dict['n_ims']*num_tiles*num_tiles]  # only 1000 training out of the 5000 generated
    #x_list = x_list[:1000]
    num_x = len(x_list)
    with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    partition = {'train': x_list[:int(num_x * 0.9)], 'validate': x_list[int(num_x * 0.9):]}
    train_ds = MyDataset(x_folder, partition['train'], labels)
    # end removed 28/04/2026 to change training data image reading
    '''
    # ori's edit to apply fov weightening 07/02/2027
    apply_fov_reweighting = True

    if apply_fov_reweighting:
        # --- FOV reweighting: sample edge tiles more than center tiles (relative to central bead) ---
        if param_dict.get("fov_reweight", True):
            num_py, num_px = param_dict.get("tile_grid", (1, 1))  # (numOfPatches, numOfPatches)
            tile_H, tile_W = param_dict.get("tile_size_px", (param_dict["H"], param_dict["W"]))
            r_cb, c_cb = param_dict.get("centralBeadCoordinates_pixel", [param_dict["H"] / 2, param_dict["W"] / 2])

            # strength: 0 -> no effect, larger -> more edge emphasis
            alpha = float(param_dict.get("fov_reweight_alpha", 4.0))  # 2.0
            power = float(param_dict.get("fov_reweight_power", 2.0))  # 1.0

            # max possible distance (to normalize)
            #H_full, W_full = param_dict["H"], param_dict["W"]
            #dmax = ((max(r_cb, H_full - 1 - r_cb)) ** 2 + (max(c_cb, W_full - 1 - c_cb)) ** 2) ** 0.5 + 1e-12

            weights = []
            for fname in partition["train"]:
                # fname example: "00012_FOV_00037.tif"
                m = re.search(r"_FOV_(\d+)\.tif$", fname)
                if m is None:
                    # fallback: if no FOV index, don't reweight
                    weights.append(1.0)
                    continue

                fov_inx = int(m.group(1))
                ii = fov_inx // num_px  # row tile index
                jj = fov_inx % num_px  # col tile index

                # tile center in FULL-camera pixel coordinates
                cy = ii * tile_H + (tile_H - 1) / 2.0
                cx = jj * tile_W + (tile_W - 1) / 2.0

                # distance from CENTRAL BEAD (not image center)
                #d = ((cy - r_cb) ** 2 + (cx - c_cb) ** 2) ** 0.5
                #d_normalized = float(d / dmax)  # [0..1]

                # weight increases with distance
                #w = 1.0 + alpha * (d_normalized ** power)

                R = float(param_dict.get("fov_reweight_radius_px", 500.0))  # your ring radius in pixels. this is where the probability function will saturate
                R = max(R, 1.0)

                d = ((cy - r_cb) ** 2 + (cx - c_cb) ** 2) ** 0.5

                # cap distance at R (ring cap)
                d_eff = min(d, R)
                d_normalized = float(d_eff / R)  # [0..1]

                w = 1.0 + alpha * (d_normalized ** power)  # in [1 .. 1+alpha]

                weights.append(w)

            # ---- FORCE weights for specific tile indices (FOV indices) ----
            #force_tiles = set(param_dict.get("fov_force_tiles", [0,1,2,4,5,6,7,8, 9,10, 16,17,18, 24,25,26, 56 , 23,31,39,47,55 ,59,60,61,62,63]))  # e.g. [1,2,3,9,10,11] for mes2
            force_tiles = set(param_dict.get("fov_force_tiles", [0,1,2,4,5,6,7,8, 9,10, 13,14,15,16,17,18, 21,22, 24,25,26, 56,57, 40, 41,49,48, 23,31,39,47,55 ,61,62,63]))  # e.g. [1,2,3,9,10,11] for mes2
            force_value = float(param_dict.get("fov_force_value", 1*3))  # 0.0 = never sample, 1.0 = baseline, 1e6 = almost always

            # weights list is aligned with partition["train"] order
            for k, fname in enumerate(partition["train"]):
                m = re.search(r"_FOV_(\d+)\.tif$", fname)
                if m is None:
                    continue
                fov_inx = int(m.group(1))
                if fov_inx in force_tiles:
                    weights[k] = force_value
                    #print('skipping frame ' + str(fov_inx))

            # ---- end weights for specific tile indices (FOV indices) ----

            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

            # IMPORTANT: when sampler is used, shuffle must be False
            train_dl = DataLoader(train_ds, sampler=sampler, shuffle=False, **params_train)
        else:
            train_dl = DataLoader(train_ds, shuffle=True, **params_train)


    else:
        train_dl = DataLoader(train_ds, **params_train)

    # end ori's edit to apply fov weightening 07/02/2027
    #train_dl = DataLoader(train_ds, **params_train)  # removed on 07/02/2027.

    validate_ds = MyDataset(x_folder, partition['validate'], labels, cache_info=x_cache) # added on 28/04/2026 to change training data image reading
    # validate_ds = MyDataset(x_folder, partition['validate'], labels)     # replaced on 28/04/2026 to change training data image reading

    validate_dl = DataLoader(validate_ds, **params_validate)

    D, us_factor, maxv = labels['volume_size'][0], labels['us_factor'], labels['blob_maxv']
    # added on 28/04/2026 to enable training resume
    ## resume-capable loading
    resume_file = training_dict.get('resume_net_file', None)
    if resume_file == 'None':
        resume_file = None

    model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)

    optimizer = Adam(list(model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=1,
        verbose=True,
        min_lr=1e-6
    )

    start_epoch = 0
    best_metric = None
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    resume_checkpoint = None

    if resume_file is not None:
        ckpt_path = os.path.join(path_save, resume_file)
        resume_checkpoint = torch.load(ckpt_path, map_location=device)

        state_dict = resume_checkpoint.get('model_state_dict', resume_checkpoint.get('state_dict'))
        model.load_state_dict(state_dict)

        if resume_checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])

        if resume_checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])

        start_epoch = int(resume_checkpoint.get('epoch', 0))
        best_metric = resume_checkpoint.get('best_metric', None)
        epochs_without_improvement = int(resume_checkpoint.get('epochs_without_improvement', 0))
        history = resume_checkpoint.get('fit_history', history)

        # --- restore RNG states safely ---
        rng_state = resume_checkpoint.get('torch_rng_state', None)
        if rng_state is not None:
            try:
                if isinstance(rng_state, torch.Tensor):
                    rng_state = rng_state.detach().cpu()
                    if rng_state.dtype != torch.uint8:
                        rng_state = rng_state.to(torch.uint8)
                    torch.set_rng_state(rng_state)
                elif isinstance(rng_state, np.ndarray):
                    torch.set_rng_state(torch.from_numpy(rng_state.astype(np.uint8)))
                elif isinstance(rng_state, (list, tuple)):
                    torch.set_rng_state(torch.tensor(rng_state, dtype=torch.uint8))
                else:
                    print(f"[resume] skipping torch RNG restore: unsupported type {type(rng_state)}")
            except Exception as e:
                print(f"[resume] skipping torch RNG restore: {e}")

        np_state = resume_checkpoint.get('numpy_rng_state', None)
        if np_state is not None:
            try:
                np.random.set_state(np_state)
            except Exception as e:
                print(f"[resume] skipping NumPy RNG restore: {e}")

        cuda_state = resume_checkpoint.get('cuda_rng_state_all', None)
        if torch.cuda.is_available() and cuda_state is not None:
            try:
                # support both list-of-tensors and list-of-arrays/lists
                fixed_cuda_state = []
                for st in cuda_state:
                    if isinstance(st, torch.Tensor):
                        st = st.detach().cpu()
                        if st.dtype != torch.uint8:
                            st = st.to(torch.uint8)
                    elif isinstance(st, np.ndarray):
                        st = torch.from_numpy(st.astype(np.uint8))
                    elif isinstance(st, (list, tuple)):
                        st = torch.tensor(st, dtype=torch.uint8)
                    else:
                        raise TypeError(f"unsupported CUDA RNG state type: {type(st)}")
                    fixed_cuda_state.append(st)
                torch.cuda.set_rng_state_all(fixed_cuda_state)
            except Exception as e:
                print(f"[resume] skipping CUDA RNG restore: {e}")

        print(f'[resume] loaded full training state: {ckpt_path}')
        print(f'[resume] continuing from epoch {start_epoch}')
    else:
        print('[resume] starting from scratch')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')
    # end

    '''
    #update: removed on 28/04/2026 to enable training resume
    ## adding resume functionality 19/04/2026 (old version):
    resume_file = training_dict.get('resume_net_file', None)

    if resume_file is not None:
        ckpt_path = os.path.join(path_save, resume_file)
        checkpoint = torch.load(ckpt_path, map_location=device)

        model = checkpoint['net']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

        print(f'[resume] loaded checkpoint: {ckpt_path}')
    else:
        model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)
        print('[resume] starting from scratch')
    #end resume functionality

    #model = Net(D=D, us_factor=us_factor, maxv=maxv).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'# of trainable parameters: {n_params}')

    optimizer = Adam(list(model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True,
                                  min_lr=1e-6)  # verbose True
    '''
    #added on 19/04/2026
    tv_z_weight = 5e0 *0 # first try 1e2?
    if param_dict['us_factor'] == 1:
        my_loss_func = KDE_loss3D(sigma=1.0, device=device, tv_z_weight=tv_z_weight)
    else:
        my_loss_func = KDE_loss3D(
            sigma=0.5 * (param_dict['us_factor'] / 2),
            device=device,
            tv_z_weight=tv_z_weight
        )
    # end 19/04/2026
    ''' removed on 19/04/2026
    if param_dict['us_factor']==1:
        # my_loss_func = torch.nn.MSELoss()
        my_loss_func = KDE_loss3D(sigma=1.0, device=device)
    else:
        my_loss_func = KDE_loss3D(sigma=0.5*(param_dict['us_factor']/2), device=device)  # 0.5-2, 1.0-4
    '''
    # added on 28/04/2026 to enable training resume
    trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

    if resume_checkpoint is not None:
        best_file_path = resume_checkpoint.get('file_name', None)
        last_file_path = resume_checkpoint.get('last_file_name', None)

        if best_file_path is None:
            time_now = datetime.today().strftime('%m-%d_%H-%M')
            net_file = 'net_' + time_now + '.pt'
            best_file_path = os.path.join(path_save, net_file)
        else:
            net_file = os.path.basename(best_file_path)

        if last_file_path is None:
            if net_file.startswith('net_'):
                last_net_file = 'last_' + net_file
            else:
                last_net_file = 'last_net_' + datetime.today().strftime('%m-%d_%H-%M') + '.pt'
            last_file_path = os.path.join(path_save, last_net_file)
        else:
            last_net_file = os.path.basename(last_file_path)
    else:
        time_now = datetime.today().strftime('%m-%d_%H-%M')
        net_file = 'net_' + time_now + '.pt'
        last_net_file = 'last_net_' + time_now + '.pt'
        best_file_path = os.path.join(path_save, net_file)
        last_file_path = os.path.join(path_save, last_net_file)

    checkpoints = dict(
        file_name=best_file_path,
        last_file_name=last_file_path,
        net=Net(D=D, us_factor=us_factor, maxv=maxv),
        state_dict=None,
        note='resume-capable checkpoint'
    )

    t0 = time.time()
    fit_results = trainer.fit(
        train_dl,
        validate_dl,
        num_epochs=num_epochs,
        checkpoints=checkpoints,
        early_stopping=4,
        start_epoch=start_epoch,
        history=history,
        best_metric=best_metric,
        epochs_without_improvement=epochs_without_improvement
    )

    fit_stamp = datetime.today().strftime('%m-%d_%H-%M')
    fit_file = 'fit_' + fit_stamp + '.pickle'
    with open(os.path.join(path_save, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()
    print(f'training results in {net_file}, {last_net_file} and {fit_file}')
    print(f'finished training in {t1 - t0}s.')

    param_dict['net_file'] = net_file
    param_dict['last_net_file'] = last_net_file
    param_dict['fit_file'] = fit_file

    return net_file, fit_file
    # end
    '''
    # removed on 28/04/2026 to enable training resume
    trainer = TorchTrainer(model, my_loss_func, optimizer, lr_scheduler=scheduler, device=device)

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    net_file = 'net_' + time_now + '.pt'
    checkpoints = dict(file_name=os.path.join(path_save, net_file),
                       net=Net(D=D, us_factor=us_factor, maxv=maxv),
                       state_dict=None,
                       note=' '
                       )

    t0 = time.time()
    fit_results = trainer.fit(train_dl, validate_dl, num_epochs=num_epochs, checkpoints=checkpoints, early_stopping=4)
    fit_file = 'fit_' + time_now + '.pickle'
    with open(os.path.join(path_save, fit_file), 'wb') as handle:
        pickle.dump(fit_results, handle)

    t1 = time.time()
    print(f'training results in {net_file} and {fit_file}')
    print(f'finished training in {t1 - t0}s.')

    return net_file, fit_file

    '''
def inference_func1(param_dict, test_idx, fig_flag=True):  # simulation and try one exp image

    np.random.seed(11)
    torch.manual_seed(11)

    device = param_dict['device']
    # ori's edit
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
    print(f'device used (interference_func1): {device}')
    #end ori's edit
    try:
        path_save = param_dict['path_save']

        net_file = param_dict['net_file']
        fit_file = param_dict['fit_file']
        with open(os.path.join(path_save, fit_file), 'rb') as handle:
            fit_result = pickle.load(handle)

    except:
        path_save = "training_results"  # added on 1/1/2026
        net_file = "net_02-20_08-24.pt"
        fit_file = "fit_02-20_08-24.pickle"
        with open(os.path.join(path_save, fit_file), 'rb') as handle:
            fit_result = pickle.load(handle)

    # training performance
    #with open(os.path.join(path_save, fit_file), 'rb') as handle:
        #fit_result = pickle.load(handle)
    train_loss = fit_result.train_loss
    test_loss = fit_result.test_loss

    if fig_flag:
        num_epochs = len(train_loss)
        plt.figure(figsize=(6, 3))
        plt.plot(np.arange(num_epochs), train_loss)
        plt.plot(np.arange(num_epochs), test_loss)
        plt.title('training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.savefig('loss_curves.jpg', bbox_inches='tight', dpi=300)
        plt.clf()

        print('Training loss curves: loss_curves.jpg')

    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # simulated data
    model = ImModelTraining(param_dict)
    sampling = Sampling(param_dict)
    volume2xyz = Volume2XYZ(param_dict)
    # added on 23/04/2026
    # =========================
    # test on one ACTUAL training tile
    # =========================
    volume2xyz = Volume2XYZ(param_dict)

    td_folder = param_dict['td_folder']
    x_folder = os.path.join(td_folder, 'x')

    with open(os.path.join(td_folder, 'y.pickle'), 'rb') as handle:
        labels = pickle.load(handle)

    x_list = sorted([f for f in os.listdir(x_folder) if f.lower().endswith('.tif')])
    if len(x_list) == 0:
        raise RuntimeError(f'No training tiles found in {x_folder}')

    # choose one tile by index
    sim_idx = int(test_idx) % len(x_list)
    sim_name = x_list[sim_idx]

    # load the exact tile image used in training
    im = io.imread(os.path.join(x_folder, sim_name)).astype(np.float32)

    # build network input exactly like training/inference
    H_sim, W_sim = im.shape

    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min() + 1e-12)).astype(np.float32)

    if param_dict.get('use_xy_maps', True):
        tile_rows, tile_cols = param_dict.get('tile_grid', (8, 8))
        tile_h, tile_w = param_dict.get('tile_size_px', (H_sim, W_sim))

        import re
        m = re.search(r'_FOV_(\d+)', sim_name)
        if m is None:
            raise RuntimeError(f'Could not parse FOV index from filename: {sim_name}')
        fov_idx = int(m.group(1))

        tile_row = fov_idx // tile_cols
        tile_col = fov_idx % tile_cols

        yy_local, xx_local = np.mgrid[0:H_sim, 0:W_sim]
        yy_global = yy_local + tile_row * tile_h
        xx_global = xx_local + tile_col * tile_w

        Xmap_sim = (2.0 * xx_global / (param_dict['WW'] - 1) - 1.0).astype(np.float32)
        Ymap_sim = (2.0 * yy_global / (param_dict['HH'] - 1) - 1.0).astype(np.float32)

        net_input = torch.from_numpy(np.stack([im, Xmap_sim, Ymap_sim], axis=0)[np.newaxis]).to(device)
    else:
        net_input = torch.from_numpy(im[np.newaxis, np.newaxis]).to(device)

    with torch.no_grad():
        net.eval()
        vol = net(net_input)

    xyz_rec, conf_rec = volume2xyz(vol)

    # reconstruct GT volume from labels to get GT localizations in tile coordinates
    xyz_ids, blob3d = labels[sim_name][0], labels[sim_name][1]
    xyz_ids = np.asarray(xyz_ids)

    # convert integer voxel anchors to approximate physical tile-local GT positions
    # this is coarse GT (voxel-anchor based), but good enough for sanity-check plots
    D = labels['volume_size'][0]
    HH = labels['volume_size'][1]
    WW = labels['volume_size'][2]

    x_gt = (xyz_ids[:, 0] - ((WW - 1) / 2)) * param_dict['vs_xy']
    y_gt = (xyz_ids[:, 1] - ((HH - 1) / 2)) * param_dict['vs_xy']
    z_gt = (xyz_ids[:, 2] + 0.5) * param_dict['vs_z'] + param_dict['zrange'][0]
    xyz_gt = np.c_[x_gt, y_gt, z_gt]

    if xyz_rec is not None and len(xyz_rec) > 0:
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)
        jaccard_index = np.round(jaccard_index, decimals=2)
        RMSE_xy = None if RMSE_xy is None else np.round(RMSE_xy * 1000, decimals=2)
        RMSE_z = None if RMSE_z is None else np.round(RMSE_z * 1000, decimals=2)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], c='b', marker='o', label='GT', depthshade=False)
        ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='Rec', depthshade=False)
        ax.set_xlabel('X [um]')
        ax.set_ylabel('Y [um]')
        ax.set_zlabel('Z [um]')
        if RMSE_xy is not None:
            plt.title(f'Found {xyz_rec.shape[0]} / {xyz_gt.shape[0]}, j_idx: {jaccard_index}, r_xy: {RMSE_xy} nm, r_z: {RMSE_z} nm')
        else:
            plt.title(f'Found {xyz_rec.shape[0]} emitters out of {xyz_gt.shape[0]}')
        plt.legend()
        plt.savefig('sim_loc_gt_rec.jpg', dpi=300)
        plt.clf()

        # reconstruct image from recovered localizations for visual sanity check
        param_dict_local = param_dict.copy()
        param_dict_local['H'] = H_sim
        param_dict_local['W'] = W_sim
        model = ImModelTraining(param_dict_local)

        nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
        psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()
        im_rec = np.sum(psfs_rec, axis=0)
        im_rec = (im_rec - im_rec.min()) / (im_rec.max() - im_rec.min() + 1e-12)

        im_show = (im - im.min()) / (im.max() - im.min() + 1e-12)

        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im_show, cmap='gray')
        plt.title('im')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(im_rec, cmap='gray')
        plt.title('im_rec')
        plt.axis('off')
        # added on 23/04/2026
        # stronger, clearer overlay
        im_base = (im - im.min()) / (im.max() - im.min() + 1e-12)
        im_rec_n = (im_rec - im_rec.min()) / (im_rec.max() - im_rec.min() + 1e-12)

        # optional gamma to make weak detections more visible
        gamma = 0.6
        im_rec_vis = im_rec_n ** gamma

        # suppress very weak reconstructed background
        thr = 0.15
        alpha = np.clip((im_rec_vis - thr) / (1 - thr + 1e-12), 0, 1)

        # grayscale base image as RGB
        im_overlay = np.stack([im_base, im_base, im_base], axis=-1)

        # magenta overlay: add rec to R and B
        overlay_color = np.zeros_like(im_overlay)
        overlay_color[:, :, 0] = im_rec_vis  # R
        overlay_color[:, :, 2] = im_rec_vis  # B

        # alpha blend
        alpha_scale = 2
        a = (alpha_scale * alpha)[..., None]
        im_overlay = (1 - a) * im_overlay + a * overlay_color

        plt.subplot(1, 3, 3)
        plt.imshow(np.clip(im_overlay, 0, 1))
        plt.title('overlay')
        plt.axis('off')
        # end
        ''' replaced on 23/04/2026
        mask = np.max(psfs_rec, axis=0)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-12)
        mask = 1 - mask
        transparency = 0.2 + mask * 0.8
        im_overlay = np.stack((im_show, im_show, im_show, transparency), axis=-1)
        im_overlay[:, :, 1] = im_overlay[:, :, 1] * mask

        plt.subplot(1, 3, 3)
        plt.imshow(im_overlay)
        plt.title('overlay')
        plt.axis('off')
        '''
        plt.savefig('sim_im_gt_rec.jpg', bbox_inches='tight', dpi=300)
        plt.clf()

        print(f'Network inference on training tile: {sim_name}')
    # end

    # removed on 23/04/2026:
    # simulation
    ''' # removed on 23/04/2026
    xyzps, _, _ = sampling.xyzp_batch()
    # ori's edit:
    #im = model(torch.from_numpy(xyzps%(1400/8)).to(device)).cpu().numpy().astype(np.float32)
    ##im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
    #end ori's edit
    im = model(torch.from_numpy(xyzps).to(device)).cpu().numpy().astype(np.float32)
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min()))
    # ori's edit 09/12/2025 tile depend training
    # --- NEW: build coord channels (treat this as whole camera) ---
    H_sim, W_sim = im.shape
    yy_sim, xx_sim = np.mgrid[0:H_sim, 0:W_sim]
    Xmap_sim = (xx_sim / (W_sim - 1) * 2 - 1).astype(np.float32)
    Ymap_sim = (yy_sim / (H_sim - 1) * 2 - 1).astype(np.float32)
    im_sim_3 = np.stack([im, Xmap_sim, Ymap_sim], axis=0)  # (3,H,W)

    with torch.no_grad():
        net.eval()
        vol = net(torch.from_numpy(im_sim_3[np.newaxis]).to(device))

    xyz_rec, conf_rec = volume2xyz(vol)
    #with torch.no_grad():
    #    net.eval()
    #    # ori's edit (not sure) 12/07/2025
    #    #XX = im[0,:,:]
    #    #vol = net(torch.from_numpy(XX[np.newaxis, np.newaxis, :, :]).to(device))
    #    # end ori's edit
    #    vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
    #xyz_rec, conf_rec = volume2xyz(vol)
    #
    # ori's edit
    # after: xyz_rec, conf_rec = volume2xyz(vol)
    #if xyz_rec is None or (hasattr(xyz_rec, "shape") and xyz_rec.shape[0] == 0):
    #    print("No emitters detected in the simulated test image (xyz_rec=None). "
    #          "Consider lowering `threshold` or using tiled inference. Skipping the simulated overlay.")
    #else:
    #    nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
    #    psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()
    #    # ... keep existing overlay code ...
    #end ori's edit
    #
    if xyz_rec is not None:
        xyz_gt = xyzps[:, :-1]
        #jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.1)
        #jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 0.11)  # ori's edit
        jaccard_index, RMSE_xy, RMSE_z, _ = calc_jaccard_rmse(xyz_gt, xyz_rec, 30)  # ori's edit
        jaccard_index, RMSE_xy, RMSE_z = np.round(jaccard_index, decimals=2), np.round(RMSE_xy*1000, decimals=2), np.round(RMSE_z*1000, decimals=2)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2], c='b', marker='o', label='GT', depthshade=False)
        ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='Rec', depthshade=False)
        ax.set_xlabel('X [um]')
        ax.set_ylabel('Y [um]')
        ax.set_zlabel('Z [um]')
        if RMSE_xy is not None:
            plt.title(f'Found {xyz_rec.shape[0]} / {xyz_gt.shape[0]}, j_idx: {jaccard_index}, r_xy: {RMSE_xy} nm, r_z: {RMSE_z} nm')
        else:
            plt.title(f'Found {xyz_rec.shape[0]} emitters out of {xyz_gt.shape[0]}')
        plt.legend()
        plt.savefig('sim_loc_gt_rec.jpg', dpi=300)
        plt.clf()

        nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
        psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy()
        im_rec = np.sum(psfs_rec, axis=0)
        im_rec = (im_rec-im_rec.min())/(im_rec.max()-im_rec.min())
        im = (im-im.min())/(im.max()-im.min())

        fig = plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(im, cmap='gray')
        plt.title('im')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(im_rec, cmap='gray')
        plt.title('im_rec')
        plt.axis('off')

        mask = np.max(psfs_rec, axis=0)
        mask = (mask-mask.min())/(mask.max()-mask.min())
        mask = 1-mask
        transparency = 0.2+mask*0.8
        im_overlay = np.stack((im, im, im, transparency), axis=-1)
        im_overlay[:, :, 1] = im_overlay[:, :, 1] * mask
        plt.subplot(1, 3, 3)
        plt.imshow(im_overlay)
        plt.title('overlay')
        plt.axis('off')

        plt.savefig('sim_im_gt_rec.jpg', bbox_inches='tight', dpi=300)
        plt.clf()

        print('Network inference on simulated an image: sim_im_gt_rec.jpg')
        # end removed on 23/04/2026
        '''

    exp_imgs_path = param_dict['im_br_folder']
    img_names = sorted(os.listdir(exp_imgs_path))

    im = io.imread(os.path.join(exp_imgs_path, img_names[test_idx])).astype(np.float32)  # read the test image
    if param_dict['project_01']:
        im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
    #Ori's edit 09/12/2025 tile dependence training
    # --- NEW: coord channels over full camera FOV ---
    H_exp, W_exp = im.shape
    yy_exp, xx_exp = np.mgrid[0:H_exp, 0:W_exp]
    # use camera_size_px if present (so normalization is identical to training)
    Hc, Wc = param_dict.get('camera_size_px', (H_exp, W_exp))
    Xmap_exp = (xx_exp / (Wc - 1) * 2 - 1).astype(np.float32)
    Ymap_exp = (yy_exp / (Hc - 1) * 2 - 1).astype(np.float32)
    im_exp_3 = np.stack([im, Xmap_exp, Ymap_exp], axis=0)  # (3,H,W)
    # end
    with torch.no_grad():
        net.eval()
        #vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
        #vol = net(torch.from_numpy(im_exp_3[np.newaxis, np.newaxis, :, :]).to(device))
        vol = net(torch.from_numpy(im_exp_3[np.newaxis]).to(device))  # (1,3,H,W)

    xyz_rec, conf_rec = volume2xyz(vol)


    H, W = im.shape
    param_dict['H'], param_dict['W'] = H, W
    model = ImModelTraining(param_dict)

    if H > param_dict['phase_mask'].shape[0] or W > param_dict['phase_mask'].shape[1]:
        sf = max(H // param_dict['phase_mask'].shape[0]+1, W // param_dict['phase_mask'].shape[1]+1)
        param_dict['ps_BFP'] /= sf
        phase_mask = param_dict['phase_mask']

        HW = np.floor(param_dict['f_4f'] * param_dict['lamda'] / (param_dict['ps_camera'] * param_dict['ps_BFP']))  # simulation size
        HW = int(HW + 1 - (HW % 2))  # make it odd

        phase_mask = interpolate(torch.tensor(phase_mask).unsqueeze(0).unsqueeze(1), size=(HW, HW))
        param_dict['phase_mask'] = phase_mask[0, 0].numpy()
        model = ImModelTraining(param_dict)

    nphotons_rec = 1e4 * np.ones(xyz_rec.shape[0])
    psfs_rec = model.get_psfs(torch.from_numpy(np.c_[xyz_rec, nphotons_rec]).to(device)).cpu().numpy() #originally
    #psfs_rec = model.psf_patch_clean(np.c_[xyz_rec, nphotons_rec]).cpu().numpy()

    im_rec = np.sum(psfs_rec, axis=0)
    im_rec = (im_rec - im_rec.min()) / (im_rec.max() - im_rec.min())

    im = (im - im.min()) / (im.max() - im.min())

    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title('im')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(im_rec, cmap='gray')
    plt.title(f'im_rec, found {xyz_rec.shape[0]} emitters')
    plt.axis('off')

    mask = np.max(psfs_rec, axis=0)
    #mask = im_rec
    #MEAN = mask.mean()
    #STD = mask[mask>0].std()
    MIN = mask.min()*2
    MAX = mask.max()*1
    #mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = (mask - MIN) / (MAX - MIN)
    #mask = 1 - mask
    mask[mask<0] = 0

    transparency = 1 + mask * 1


    im_overlay = np.stack((im, im, im, transparency), axis=-1)

    # mask = mask<0.1
    im_overlay[:, :, 1] = im_overlay[:, :, 1]/2 + mask/2
    plt.subplot(1, 3, 3)
    plt.imshow(im_overlay)
    plt.title('overlay')
    plt.axis('off')


    plt.savefig('exp_im_gt_rec.jpg', bbox_inches='tight', dpi=300)

    print('Network inference on a test experimental image: exp_im_gt_rec.jpg')


def inference_func2(param_dict):
    # ori's edit at 14/12/2025 for overlapping
    device = param_dict['device']
    # ori's edit (keep if you had it)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f'device used (inference_func2): {device}')

    try:
        path_save = param_dict['path_save']
        net_file = param_dict['net_file']

    except:
        path_save = "training_results"  # added on 1/1/2026
        net_file = "net_02-20_08-24.pt"
        fit_file = "fit_02-20_08-24.pickle"

    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # localization converter (unchanged)
    volume2xyz = Volume2XYZ(param_dict)

    exp_imgs_path = param_dict['im_br_folder']
    img_names = sorted(os.listdir(exp_imgs_path))
    num_imgs = len(img_names)

    # --- patch & overlap parameters ---
    #patch_H = int(param_dict['H'])  # network input size in y  # removed on 24/12/2025 - bug
    #patch_W = int(param_dict['W'])  # network input size in x  # removed on 24/12/2025 - bug


    ''' removed on 24/02/2026
    patch_H, patch_W = param_dict.get("tile_size_px", (param_dict["H"], param_dict["W"]))  # added on 24/12/2025 - bug
    # fraction of the patch (per dimension) we trust in the center, e.g. 0.9
    center_fraction = float(param_dict.get('center_fraction', 0.9))
    center_fraction = max(0.0, min(center_fraction, 1.0))  # clamp to [0,1]

    # central "valid" half-sizes in pixels (per patch)
    valid_half_H = 0.5 * center_fraction * patch_H
    valid_half_W = 0.5 * center_fraction * patch_W
    
    # stride between patch starts → gives overlap
    # with this choice, central regions of consecutive patches just touch
    stride_H = max(1, int(patch_H * center_fraction))
    stride_W = max(1, int(patch_W * center_fraction))
    '''
    # added on 24/02/2026
    # --- patch & overlap parameters ---
    patch_H, patch_W = param_dict.get("tile_size_px", (param_dict["H"], param_dict["W"]))

    center_fraction = float(param_dict.get('center_fraction', 0.7))  # you said 0.7
    center_fraction = max(0.0, min(center_fraction, 1.0))

    # IMPORTANT: use ROUND (not floor) so stride is consistent
    stride_H = int(round(patch_H * center_fraction))
    stride_W = int(round(patch_W * center_fraction))
    stride_H = max(1, min(stride_H, patch_H))
    stride_W = max(1, min(stride_W, patch_W))

    # Integer margins for the trusted center region
    m_top = (patch_H - stride_H) // 2
    m_bot = (patch_H - stride_H) - m_top
    m_left = (patch_W - stride_W) // 2
    m_right = (patch_W - stride_W) - m_left
    #

    vs_xy = param_dict['vs_xy']  # μm per camera pixel (or μm per us-pixel)
    us = param_dict['us_factor']

    tall_start = time.time()
    # results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])  # removed on 17/01/2026 to save as chunks instead
    #  17/01/2026 to save as chunks
    header = ['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]']

    chunk_size_frames = int(param_dict.get('save_chunk_frames', 5000))  #10_000 default 10k
    chunk_id = 0
    rows_buffer = []  # list of rows (each row is length-5)
    # end  17/01/2026 to save as chunks
    with torch.no_grad():
        net.eval()

        for im_ind, im_name in enumerate(img_names):
            print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))
            tfrm_start = time.time()

            # --- load full experimental image ---
            im_full = io.imread(os.path.join(exp_imgs_path, im_name)).astype(np.float32)

            if param_dict.get('project_01', False):
                im_min, im_max = im_full.min(), im_full.max()
                if im_max > im_min:
                    im_full = (im_full - im_min) / (im_max - im_min)
                im_full = im_full.astype(np.float32)

            H_full, W_full = im_full.shape
            #ch_full, cw_full = H_full / 2.0, W_full / 2.0  # global image center in pixel coords
            ch_full, cw_full = (H_full - 1) / 2.0, (W_full - 1) / 2.0  # from 23/02/2026 to fix tiles
            # --- compute sliding-window start indices with coverage up to the borders ---
            y_starts = list(range(0, max(1, H_full - patch_H + 1), stride_H))
            if y_starts[-1] != H_full - patch_H:
                y_starts.append(H_full - patch_H)

            x_starts = list(range(0, max(1, W_full - patch_W + 1), stride_W))
            if x_starts[-1] != W_full - patch_W:
                x_starts.append(W_full - patch_W)

            all_xyz_nm = []
            all_conf = []

            for y0 in y_starts:
                for x0 in x_starts:
                    # crop patch

                    # center of this patch in GLOBAL pixel coordinates
                    ''' removed on 24/02/2026
                    patch = im_full[y0:y0 + patch_H, x0:x0 + patch_W]
                    cy = y0 + (patch_H - 1) / 2.0
                    cx = x0 + (patch_W - 1) / 2.0

                    # define central valid region for this patch in GLOBAL pixel coords
                    x_min_valid = cx - valid_half_W
                    x_max_valid = cx + valid_half_W
                    y_min_valid = cy - valid_half_H
                    y_max_valid = cy + valid_half_H
                    '''
                    # added on 24/02/2026
                    # valid region in GLOBAL pixel coordinates (integers)
                    x_min_valid = x0 + m_left
                    x_max_valid = x0 + patch_W - m_right
                    y_min_valid = y0 + m_top
                    y_max_valid = y0 + patch_H - m_bot

                    # Expand at image borders so we don't drop true borders
                    if x0 == 0:
                        x_min_valid = 0
                    if x0 == (W_full - patch_W):
                        x_max_valid = W_full
                    if y0 == 0:
                        y_min_valid = 0
                    if y0 == (H_full - patch_H):
                        y_max_valid = H_full

                    cy = y0 + (patch_H - 1) / 2.0
                    cx = x0 + (patch_W - 1) / 2.0

                    #end
                    # pass through network
                    # ori's edit added on 14/12/2025 for overlapping fov
                    # crop patch
                    patch = im_full[y0:y0 + patch_H, x0:x0 + patch_W]

                    # --- NEW: build coord channels for this patch (global FOV coordinates) ---
                    H_patch, W_patch = patch.shape
                    yy_local, xx_local = np.mgrid[0:H_patch, 0:W_patch]

                    # global pixel coordinates of this patch
                    yy_global = yy_local + y0
                    xx_global = xx_local + x0

                    # use the original camera size used during training, if available
                    Hc, Wc = param_dict.get("camera_size_px", (H_full, W_full))

                    # added on 07/02/2026 for debug
                    if im_ind == 0 and y0 == y_starts[0] and x0 == x_starts[0]:
                        print(f"[INFER2] H_full,W_full={H_full},{W_full}  Hc,Wc={Hc},{Wc}")
                        print("[INFER2] param camera_size_px =", param_dict.get("camera_size_px", None))

                    # end 07/02/2026

                    Xmap_patch = (xx_global / (Wc - 1) * 2 - 1).astype(np.float32)
                    Ymap_patch = (yy_global / (Hc - 1) * 2 - 1).astype(np.float32)
                    '''
                    # added on 07/02/2026
                    # --- DEBUG  ---
                    if im_ind == 0 and ((y0 == y_starts[0] and x0 == x_starts[0]) or
                                        (y0 == y_starts[0] and x0 == x_starts[-1]) or
                                        (y0 == y_starts[-1] and x0 == x_starts[0]) or
                                        (y0 == y_starts[-1] and x0 == x_starts[-1])):
                        xm, xM = float(Xmap_patch.min()), float(Xmap_patch.max())
                        ym, yM = float(Ymap_patch.min()), float(Ymap_patch.max())
                        xmean, ymean = float(Xmap_patch.mean()), float(Ymap_patch.mean())
                        print(f"[INFER2] frame={im_ind} tile y0={y0} x0={x0}  "
                              f"X[{xm:+.3f},{xM:+.3f}] mean={xmean:+.3f}  "
                              f"Y[{ym:+.3f},{yM:+.3f}] mean={ymean:+.3f}")
                    # --- END DEBUG ---

                    # end 07/02/2026

                    # added on 07/02/2026
                    # --- DEBUG: coord ranges (prints for first few z slices only) ---
                    if im_ind < 3:  # or just once
                        xm, xM = float(Xmap_patch.min()), float(Xmap_patch.max())
                        ym, yM = float(Ymap_patch.min()), float(Ymap_patch.max())
                    #    print(f"[INFER] zz={im_ind} Xmap[{xm:+.3f},{xM:+.3f}]  Ymap[{ym:+.3f},{yM:+.3f}]")
                    # end 07/02/2026
                    '''
                    patch_3 = np.stack([patch, Xmap_patch, Ymap_patch], axis=0).astype(np.float32)  # (3,H,W)

                    # pass through network with 3 channels
                    inp = torch.from_numpy(patch_3[np.newaxis]).to(device)  # (1,3,H,W)
                    vol_patch = net(inp)
                    # end ori's edit

                    # debug 19/04/2026
                    DeBug = True
                    if DeBug:
                        debug_tiles = {
                            (y_starts[0], x_starts[0]),
                            (y_starts[0], x_starts[-1]),
                            (y_starts[-1], x_starts[0]),
                            (y_starts[-1], x_starts[-1]),
                            (y_starts[len(y_starts) // 2], x_starts[len(x_starts) // 2]),
                        }

                        if im_ind == 0 and (y0, x0) in debug_tiles:
                            z_profile = vol_patch[0].sum(dim=(1, 2)).detach().cpu().numpy()
                            tag = f"y{y0}_x{x0}"

                            np.savetxt(f"debug_z_profile_{tag}.csv", z_profile, delimiter=",")

                            plt.figure(figsize=(6, 3))
                            plt.plot(z_profile)
                            plt.xlabel("z channel")
                            plt.ylabel("sum over x,y")
                            plt.title(f"Raw z-profile {tag}")
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(f"debug_z_profile_{tag}.png", dpi=200)
                            plt.close()

                            print(f"[DEBUG] saved raw z-profile {tag}")

                        if im_ind == 0 and y0 == y_starts[0] and x0 == x_starts[0]:
                            vol_np = vol_patch[0].detach().cpu().numpy()  # [D,H,W]

                            xz = vol_np.max(axis=1)  # [D,W]
                            yz = vol_np.max(axis=2)  # [D,H]

                            plt.figure(figsize=(8, 3))
                            plt.imshow(xz, aspect='auto', origin='lower', cmap='gray')
                            plt.xlabel("x")
                            plt.ylabel("z channel")
                            plt.title("Raw network volume: XZ max projection")
                            plt.tight_layout()
                            plt.savefig("debug_raw_vol_xz_first_tile.png", dpi=200)
                            plt.close()

                            plt.figure(figsize=(8, 3))
                            plt.imshow(yz, aspect='auto', origin='lower', cmap='gray')
                            plt.xlabel("y")
                            plt.ylabel("z channel")
                            plt.title("Raw network volume: YZ max projection")
                            plt.tight_layout()
                            plt.savefig("debug_raw_vol_yz_first_tile.png", dpi=200)
                            plt.close()

                            print("[DEBUG] saved raw XZ/YZ projections")

                    # end debug 19/04/2026
                    # comment due to ori's edit:
                    #inp = torch.from_numpy(patch[np.newaxis, np.newaxis, :, :]).to(device)
                    #vol_patch = net(inp)

                    # convert to xyz (μm, relative to patch center, same as before)
                    xyz_patch, conf_patch = volume2xyz(vol_patch)
                    if xyz_patch is None or xyz_patch.shape[0] == 0:
                        continue

                    # ensure numpy
                    if isinstance(xyz_patch, torch.Tensor):
                        xyz_patch = xyz_patch.detach().cpu().numpy()
                    if isinstance(conf_patch, torch.Tensor):
                        conf_patch = conf_patch.detach().cpu().numpy()

                    # local → global μm (same formula you already used for tiles)
                    # xyz_patch[:,0/1/2] assumed to be μm offset from patch center
                    x_um_global = xyz_patch[:, 0] + (cx - cw_full) * vs_xy * us
                    y_um_global = xyz_patch[:, 1] + (cy - ch_full) * vs_xy * us
                    z_um_global = xyz_patch[:, 2]

                    # compute global pixel positions for gating
                    # invert x_um_global = (x_pix_global - cw_full) * vs_xy * us
                    x_pix_global = x_um_global / (vs_xy * us) + cw_full
                    y_pix_global = y_um_global / (vs_xy * us) + ch_full

                    # keep only emitters inside the central valid region of this patch
                    '''mask_valid = (
                            (x_pix_global >= x_min_valid) & (x_pix_global < x_max_valid) &
                            (y_pix_global >= y_min_valid) & (y_pix_global < y_max_valid)
                    )'''
                    # added on 24/02/2026
                    eps = 1e-6
                    mask_valid = (
                            (x_pix_global >= (x_min_valid - eps)) & (x_pix_global < (x_max_valid + eps)) &
                            (y_pix_global >= (y_min_valid - eps)) & (y_pix_global < (y_max_valid + eps))
                    )
                    #end
                    if not np.any(mask_valid):
                        continue

                    x_um_global = x_um_global[mask_valid]
                    y_um_global = y_um_global[mask_valid]
                    z_um_global = z_um_global[mask_valid]
                    conf_keep = conf_patch[mask_valid]

                    # convert to nm for saving
                    x_nm_global = x_um_global * 1000.0
                    y_nm_global = y_um_global * 1000.0
                    z_nm_global = z_um_global * 1000.0

                    all_xyz_nm.append(np.c_[x_nm_global, y_nm_global, z_nm_global])
                    all_conf.append(conf_keep)

            # combine all patches for this frame
            if len(all_xyz_nm) == 0:
                nemitters = 0
            else:
                xyz_rec_nm = np.vstack(all_xyz_nm)
                conf_rec = np.concatenate(all_conf)
                nemitters = xyz_rec_nm.shape[0]

                frm_idx = (im_ind + 1) * np.ones(nemitters)

                #frame_results = np.column_stack((frm_idx, xyz_rec_nm, conf_rec)) #removed on 17/01/2026 to save as chunks instead
                #results = np.vstack((results, frame_results))  #removed on 17/01/2026 to save as chunks instead
                #  17/01/2026 to save as chunks
                frame_results = np.column_stack((frm_idx, xyz_rec_nm, conf_rec))
                # buffer as python rows (fast append, no big copies)
                rows_buffer.extend(frame_results.tolist())

                # end  17/01/2026 to save as chunks
            tfrm_end = time.time() - tfrm_start
            print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))
            # added on 17/01/2026 to save chunks
            # flush every chunk_size_frames frames
            if (im_ind + 1) % chunk_size_frames == 0:
                time_now = datetime.today().strftime('%m-%d_%H-%M')
                chunk_name = os.path.join(os.getcwd(), f'localizations_{time_now}_chunk_{chunk_id:04d}.csv')

                with open(chunk_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows_buffer)

                print(f'[chunk saved] {chunk_name}  rows={len(rows_buffer)}')
                rows_buffer.clear()
                chunk_id += 1
            # end "save chunks"
    # to save chunks on 17/01/2025
    # flush remainder
    if len(rows_buffer) > 0:
        time_now = datetime.today().strftime('%m-%d_%H-%M')
        chunk_name = os.path.join(os.getcwd(), f'localizations_{time_now}_chunk_{chunk_id:04d}.csv')

        with open(chunk_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows_buffer)

        print(f'[chunk saved] {chunk_name}  rows={len(rows_buffer)}')

    # end save chunks
    ''' removed on 17/01/2026 to save as chunks
    tall_end = time.time() - tall_start
    print('=' * 50)
    print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        tall_end // 3600,
        np.floor((tall_end / 3600 - tall_end // 3600) * 60),
        tall_end % 60))
    print('=' * 50)

    # write the results to a csv file named "localizations_*.csv"
    row_list = results.tolist()
    time_now = datetime.today().strftime('%m-%d_%H-%M')
    file_name = os.path.join(os.getcwd(), 'localizations_' + time_now + '.csv')

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

    print(f'Localization list: {file_name}')
    return file_name
    #end ori's edit
    '''



    ''' removed at 14/12/2025 for overlapping
    
    device = param_dict['device']
    # ori's edit
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # GPU device cuda:0, cuda:1, cuda:2 or cuda:3
    print(f'device used (interference_func2): {device}')
    #end ori's edit

    path_save = param_dict['path_save']
    net_file = param_dict['net_file']
    # ori's edit
    #print("Using net_file:", net_file)
    # end ori's edit
    checkpoint = torch.load(os.path.join(path_save, net_file), map_location=device)
    net = checkpoint['net']
    net.load_state_dict(checkpoint['state_dict'])

    # simulated data
    volume2xyz = Volume2XYZ(param_dict)

    exp_imgs_path = param_dict['im_br_folder']
    img_names = sorted(os.listdir(exp_imgs_path))
    num_imgs = len(img_names)

    

    # ori's edit 09/12/2025 tile dependence training
    # we'll reuse full frame size and camera size for coord normalization
    first_im = io.imread(os.path.join(exp_imgs_path, img_names[0])).astype(np.float32)
    H_full, W_full = first_im.shape
    Hc, Wc = param_dict.get('camera_size_px', (H_full, W_full))
    # end

    tall_start = time.time()
    results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])
    with torch.no_grad():
        net.eval()
        for im_ind, im_name in enumerate(img_names):
            # print current image number
            print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))

            tfrm_start = time.time()

            im = io.imread(os.path.join(exp_imgs_path, im_name)).astype(np.float32)
            if param_dict['project_01']:
                im = ((im - im.min()) / (im.max() - im.min())).astype(np.float32)
            #vol = net(torch.from_numpy(im[np.newaxis, np.newaxis, :, :]).to(device))
            #xyz_rec, conf_rec = volume2xyz(vol)
            # ~~~~~~~ ori's edit 03/12/2025 # ~~~~~~~~~~~~~ # instead of vol=net(...) and xyz_rec, conf_rec = volume2xyz(vol)
            #num_tiles = 8
            num_tiles = param_dict.get("num_tiles",1)
            H, W = im.shape
            H_new = H // num_tiles
            W_new = W // num_tiles
            vs_xy = param_dict['vs_xy']
            us = param_dict['us_factor']

            all_xyz = []
            all_conf = []

            for ii in range(num_tiles):
                for jj in range(num_tiles):
                    y0 = ii * H_new
                    y1 = (ii + 1) * H_new
                    x0 = jj * W_new
                    x1 = (jj + 1) * W_new

                    tile = im[y0:y1, x0:x1]

                    # ori's edit 09/12/2025 tile dependence training
                    # --- NEW: build coord channels for this tile, using global FOV coords ---
                    yy_local, xx_local = np.mgrid[0:H_new, 0:W_new]
                    yy_global = yy_local + y0
                    xx_global = xx_local + x0

                    Xmap_tile = (xx_global / (Wc - 1) * 2 - 1).astype(np.float32)
                    Ymap_tile = (yy_global / (Hc - 1) * 2 - 1).astype(np.float32)

                    tile_3 = np.stack([tile, Xmap_tile, Ymap_tile], axis=0)  # (3,H_new,W_new)

                    vol_tile = net(torch.from_numpy(tile_3[np.newaxis]).to(device))
                    xyz_tile, conf_tile = volume2xyz(vol_tile)
                    # end
                    #vol_tile = net(torch.from_numpy(tile[np.newaxis, np.newaxis, :, :]).to(device))
                    #xyz_tile, conf_tile = volume2xyz(vol_tile)

                    if xyz_tile is None:
                        continue

                    # convert tile-local µm coords to global pixels offset
                    # (same convention as original code, but with tile center)
                    ch_tile, cw_tile = H_new / 2, W_new / 2

                    # replaced on 09/12/2025 to fix rectangular artifacts:
                    # first do the original conversion for the tile:
                    ''' ''' 
                    #xnm_tile = (xyz_tile[:, 0] + cw_tile * vs_xy * us) * 1000
                    #ynm_tile = (xyz_tile[:, 1] + ch_tile * vs_xy * us) * 1000

                    ## now shift by the tile center offset (in pixels * vs_xy * us)
                    #dx_pix = (x0 + cw_tile) - (W / 2)
                    #dy_pix = (y0 + ch_tile) - (H / 2)
                    #xnm_global = xnm_tile + dx_pix * vs_xy * us * 1000
                    #ynm_global = ynm_tile + dy_pix * vs_xy * us * 1000
                    ''' '''
                    x_center_pix = x0 + W_new / 2
                    y_center_pix = y0 + H_new / 2

                    xnm_global = (xyz_tile[:, 0] + x_center_pix * vs_xy * us) * 1000
                    ynm_global = (xyz_tile[:, 1] + y_center_pix * vs_xy * us) * 1000
                    # end 09/12/2025
                    znm_global = xyz_tile[:, 2] * 1000

                    all_xyz.append(np.c_[xnm_global, ynm_global, znm_global])
                    #all_conf.append(conf_tile.cpu().numpy())
                    #if error, try:
                    all_conf.append(conf_tile)


            if len(all_xyz) == 0:
                xyz_rec = None
                conf_rec = None
            else:
                xyz_rec = np.vstack(all_xyz)
                conf_rec = np.concatenate(all_conf)
            # ~~~~~~~~~~~~~~~~End or's edit ~~~~~~~~~~~~~~~
            tfrm_end = time.time() - tfrm_start

            # if this is the first image, get the dimensions and the relevant center for plotting
            H, W = im.shape
            ch, cw = H / 2, W / 2

            # if prediction is empty then set number fo found emitters to 0
            # otherwise generate the frame column and append results for saving
            if xyz_rec is None:
                nemitters = 0
            else:
                nemitters = xyz_rec.shape[0]
                frm_rec = (im_ind + 1) * np.ones(nemitters)
                # removed by ori 2025/12/03
                #xnm = (xyz_rec[:, 0] + cw * param_dict['vs_xy']*param_dict['us_factor']) * 1000
                #ynm = (xyz_rec[:, 1] + ch * param_dict['vs_xy']*param_dict['us_factor']) * 1000
                #znm = (xyz_rec[:, 2]) * 1000  # make sure they are above 0
                #xyz_save = np.c_[xnm, ynm, znm]
                xyz_save = xyz_rec # ori's edit 2025/12/03
                results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

            print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))

    # print the time it took for the entire analysis
    tall_end = time.time() - tall_start
    print('=' * 50)
    print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
    print('=' * 50)

    # write the results to a csv file named "localizations.csv" under the exp img folder
    row_list = results.tolist()

    time_now = datetime.today().strftime('%m-%d_%H-%M')
    file_name = os.path.join(os.getcwd(), 'localizations_' + time_now + '.csv')
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

    print(f'Localization list: {file_name}')

    return file_name
    '''





