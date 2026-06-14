import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
from DS3Dplus.ds3d_utils import asm_propagate

class ImModel_pr(torch.nn.Module):
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
        self.d_min_um = params["d_min_um"]
        self.d_max_um = params["d_max_um"]
        #d_init = float(params.get("mask_offset_in_um", 0.0))
        if "mask_offset_in_um" in params:
            if params["mask_offset_in_um"]!=0:  # bandage! to fix!
                d_init = float(params["mask_offset_in_um"])
            else:
                d_init = 0.5 * (self.d_min_um + self.d_max_um)  # midrange default
        else:
            d_init = 0.5 * (self.d_min_um + self.d_max_um)  # midrange default
        self.mask_offset_in_um = float(params.get("mask_offset_in_um", 0.0))

        self.centralBeadCoordinates_pixel = list(params['centralBeadCoordinates_pixel'])
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

        self.d_raw = torch.nn.Parameter(torch.tensor(d_raw_init, device=device, dtype=torch.float32))
        # end improved pr 27/01/2026
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

        k_immersion = 2 * math.pi * n_immersion / lamda  # [1/um]
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

        k_sample = 2 * math.pi * n_sample / lamda
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

        pn_circ = np.floor(np.sqrt(np.sum(circ) / math.pi) * 2)
        pn_circ = int(pn_circ + 1 - (pn_circ % 2))
        Xgrid = 2 * math.pi * xi * M / (lamda * f_4f)
        Ygrid = 2 * math.pi * eta * M / (lamda * f_4f)
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
        blur_kernel = 1 / (2 * math.pi * sigma ** 2) * (torch.exp(-0.5 * (self.g_xx ** 2 + self.g_yy ** 2) / sigma ** 2))
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
