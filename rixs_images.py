import os
import time
import h5py
import fabio
import re

import numpy as np
import matplotlib.pyplot as plt

# from scipy.signal import fftconvolve
from scipy.signal import medfilt2d, correlate
from scipy.ndimage import median_filter
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
# from scipy import optimize
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from cmcrameri import cm
from abc import ABC, abstractmethod
from nexusformat.nexus import *
from static_functions import _determine_polarization, _parallel_median_filter



class RIXS_Image(ABC):

    def __init__(self):
        """
        Initialize the shared state common to all RIXS image sources.

        Subclasses call this via `super().__init__()` before loading any
        file-specific data.

        Sets
        ----
        self.imgs_processed : None
            Placeholder for the processed image stack, populated later by
            `single_photon_counting` or `_remove_bkg_and_filter`.
        self.normalization_factor : None
            Placeholder for the normalization factor, populated later by
            `_get_normalization_factor`.
        """
        self.imgs_processed = None
        self.normalization_factor = None

    @abstractmethod
    def _get_raw_data(self):
        """
        Load raw detector images for this run and store them on the instance.

        Concrete subclasses must implement this to read from their specific
        file format (EDF, HDF5, NeXus, ...) and populate `self.raw_data` as a
        3D numpy array of shape (n_images, height, width), even when only a
        single image is present.
        """
        pass

    @abstractmethod
    def _get_run_number(self):
        """
        Determine the run number for this image source and store it on the
        instance as `self.run_number`.
        """
        pass

    @abstractmethod
    def _get_normalization_factor(self):
        """
        Determine the beam-intensity normalization factor for this image
        source and store it on the instance as `self.normalization_factor`.
        """
        pass

    @abstractmethod
    def _get_attributes(self):
        """
        Read experimental metadata (motor positions, polarization, energy,
        etc.) for this image source and store it on the instance as
        `self.attributes`, a dict keyed by attribute name.
        """
        pass

    @abstractmethod
    def _get_energy(self):
        """
        Determine the incident photon energy for this image source and
        store it on the instance as `self.energy`.
        """
        pass

    def plot(self):
        """
        Placeholder for a generic plotting entry point.

        Not implemented and not currently overridden by any subclass in this
        file — calling it does nothing. Consider removing if it is not
        planned for future use, or implementing it if it is.
        """
        pass

    def process_imgs(self,
                     use_spc,
                     spc_parameters={},
                     no_spc_parameters={},
                     plot_generation=False
                     ):
        """
        Dispatch image processing to either the single-photon-counting
        pipeline or the background-removal/filtering pipeline.

        Parameters
        ----------
        use_spc : bool
            If True, process images with `single_photon_counting`. If False,
            process images with `_remove_bkg_and_filter`.
        spc_parameters : dict, optional
            Keyword arguments forwarded to `single_photon_counting` when
            `use_spc` is True. Default {}.
        no_spc_parameters : dict, optional
            Keyword arguments forwarded to `_remove_bkg_and_filter` when
            `use_spc` is False. Default {}.
        plot_generation : bool, optional
            If True, call `plot_generation` with the same parameters after
            processing. Default False.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray or float)
            imgs_processed : the processed image stack.
            normalization_factor : the associated normalization factor(s).
        """
        
        if use_spc:
            self.single_photon_counting(**spc_parameters)
            if plot_generation:
                self.plot_generation(use_spc, **spc_parameters)
        else:
            self._remove_bkg_and_filter(**no_spc_parameters)
            if plot_generation:
                self.plot_generation(use_spc, **no_spc_parameters)

        return self.imgs_processed, self.normalization_factor

    def plot_generation(self, use_spc):
        """
        Placeholder for diagnostic-plot generation after processing.

        Base implementation does nothing; `DLS_Image` overrides this with an
        actual multi-panel diagnostic plot. Other subclasses inherit this
        no-op, so calling `process_imgs(..., plot_generation=True)` on them
        has no visible effect.

        Parameters
        ----------
        use_spc : bool
            Whether single-photon-counting (True) or background subtraction
            (False) was used, so an override can choose what to plot.
        """
        pass

    @abstractmethod
    def _remove_bkg_and_filter(self,
                               **kwargs):
        """
        Process raw images via background subtraction and spike filtering,
        as an alternative to single-photon counting.

        Concrete subclasses must implement this to remove detector
        background/dark current and filter spikes from `self.raw_data`,
        populating `self.imgs_processed`.

        Parameters
        ----------
        **kwargs : dict
            Subclass-specific processing parameters.
        """
        pass

    def single_photon_counting(
            self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=0.55,
            vertical_shift = 0,
            subtract_background_from_img=False,
            subtract_background_from_corner=False,
            bkg=0,
            dark_img=None,
            plot_raw_image=False):
        """
        Process raw detector images into a photon-counted 2D histogram using the
        single-photon-counting (centroiding) technique.

        Crops the raw data to the given ROI, optionally subtracts a flat or
        image/corner-based background, locates photon hits with `_centroid`, and
        bins the resulting sub-pixel positions into a grid with `_bin`. The
        per-photon positions are cached in `self.res` on first call, so calling
        this again with different binning parameters does not redo the
        centroiding step.

        Parameters
        ----------
        curve_a : float
            Linear (first-order) curvature correction coefficient.
        curve_b : float, optional
            Quadratic (second-order) curvature correction coefficient. Default 0.
        roi_x : tuple of int, optional
            (start, end) region of interest along the x-axis. Default (0, 2048).
        roi_y : tuple of int, optional
            (start, end) region of interest along the y-axis. Default (0, 2048).
        roi_x_for_dark : tuple of int, optional
            (start, end) region along x used to scale the dark image when
            `subtract_background_from_img` is True. Default (1600, 1800).
        roi_y_for_dark : tuple of int, optional
            (start, end) region along y used to scale the dark image when
            `subtract_background_from_img` is True. Default (250, 1800).
        subdivide_bins_factor_x : float, optional
            Number of sub-pixel bins per pixel along x. Default 1.
        subdivide_bins_factor_y : float, optional
            Number of sub-pixel bins per pixel along y. Default 2.7.
        factor_ADC : float, optional
            Factor to convert ADU to electrons. Default 0.55.
        vertical_shift : float, optional
            Vertical shift (in pixels) subtracted from the y-coordinates of photon
            positions, typically from cross-correlation between images. Leave at
            0 if cross-correlation has not been performed yet. Default 0.
        subtract_background_from_img : bool, optional
            If True, scale and subtract `dark_img` from each raw image before
            centroiding. Default False.
        subtract_background_from_corner : bool, optional
            If True, estimate a flat background from a fixed detector corner and
            subtract it (used for TPS data). Default False.
        bkg : float, optional
            Flat background value subtracted when neither
            `subtract_background_from_img` nor `subtract_background_from_corner`
            is set. Default 0.
        dark_img : numpy.ndarray, optional
            Pre-processed dark image; required if `subtract_background_from_img`
            is True. Default None.
        plot_raw_image : bool, optional
            If True, call `plot_raw_image()` after processing. Default False.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray or float)
            imgs_processed : stack of 2D photon-count histograms, one per input
            image.
            normalization_factor : normalization factor(s) for the images
            (unchanged by this method).
        """
        # Make sure we have raw data to process
        if self.raw_data is None:
            raise ValueError("No raw data available for processing")
        
        if self.raw_data.ndim < 3:
                self.raw_data = np.expand_dims(self.raw_data, axis=0)

        # Crop the image according to roi_x and roi_y
        cropped_data = np.array([img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] for img in self.raw_data])
            
        # Use the centroid function to process the image
        if self.res is None:
            
            if subtract_background_from_corner:
                # Used for TPS
                bkg = self.raw_data[:,50:-50,:70:100].mean()
                dark_img_cropped = 0
                mean_factors_dark = np.zeros(self.raw_data.shape[0])

            elif subtract_background_from_img:
                # Used for TPS, possibly
                if dark_img is None:
                    raise ValueError("Dark image must be provided")
                bkg = 0
                mean_factors_dark = [np.mean(raw[roi_y_for_dark[0]:roi_y_for_dark[1], roi_x_for_dark[0]:roi_x_for_dark[1]]) - 
                                     np.mean(dark_img[roi_y_for_dark[0]:roi_y_for_dark[1], roi_x_for_dark[0]:roi_x_for_dark[1]]) for raw in self.raw_data]
                dark_img_cropped = dark_img[roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] 
            else:
                #used for ESRF
                dark_img_cropped = 0
                mean_factors_dark = np.zeros(self.raw_data.shape[0])

            self.raw_data = cropped_data

            for raw, norm_factor_dark in zip(self.raw_data, mean_factors_dark):
                res_now, _ = self._centroid(
                    img=raw - dark_img_cropped + norm_factor_dark,
                    energy=self.attributes["energy"],
                    bkg_mean=bkg,
                    factor_ADC=factor_ADC,
                    avoid_double=False,
                    curve_a=curve_a, 
                    curve_b=curve_b,
                )
                self.res = res_now if self.res is None else self.res + res_now              

        #bin into a grid
        self.imgs_processed, _ = self._bin(self.res, 
                            image_size_h=self.raw_data.shape[-1], 
                            image_size_v=self.raw_data.shape[-2], 
                            subdivide_bins_factor_x=subdivide_bins_factor_x, 
                            subdivide_bins_factor_y=subdivide_bins_factor_y,
                            vertical_shift = vertical_shift)


        # Plot the raw image if requested
        if plot_raw_image:
            self.plot_raw_image()

        return self.imgs_processed, self.normalization_factor
    

    def plot_raw_image(self, roi_y=(0,2048), roi_x=(0,2048)):
        """
        Display a diagnostic view of the raw detector image: a 2D image
        panel plus its vertical and horizontal average profiles.

        If `self.raw_data` holds a stack of images (3D array), they are
        summed over the stack before display.

        Parameters
        ----------
        roi_y : tuple of int, optional
            (start, end) region of interest along the y-axis. Default (0, 2048).
        roi_x : tuple of int, optional
            (start, end) region of interest along the x-axis. Default (0, 2048).

        Returns
        -------
        tuple of (matplotlib.figure.Figure, tuple of matplotlib.axes.Axes)
            fig : the created figure.
            (ax1, ax2, ax3) : the vertical-average, image, and
            horizontal-average axes, respectively.
        """
        fig = plt.figure(figsize=(5, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 2])

        # Plot the raw image
        ax2 = fig.add_subplot(gs[1, 1])
        img = self.raw_data[:,roi_y[0]:roi_y[1],roi_x[0]:roi_x[1]].sum(axis=0) \
            if self.raw_data.ndim == 3 else self.raw_data[roi_y[0]:roi_y[1],roi_x[0]:roi_x[1]]
        vmin = img.mean() - 3*img.std()
        vmax = img.mean() + 3*img.std()
        im = ax2.imshow(img, cmap=cm.lapaz, vmin=vmin, vmax=vmax)
        # fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('Raw Image')
        ax2.set_aspect('auto')

        # Plot the vertical average
        ax1 = fig.add_subplot(gs[0, 1])
        vertical_avg = img.mean(axis=0)
        ax1.plot(vertical_avg)
        ax1.set_title('Vertical Average')
        ax1.set_ylim([vertical_avg.min()*0.9, vertical_avg.max()*1.1])
        # ax1.set_ylim([0, len(vertical_avg)])
        # ax1.invert_yaxis()

        # Plot the horizontal average
        ax3 = fig.add_subplot(gs[1, 0])
        horizontal_avg = img.mean(axis=1)
        ax3.plot(horizontal_avg, np.arange(len(horizontal_avg)))
        ax3.set_title('Horizontal Average')
        ax3.set_xlim([horizontal_avg.min(), horizontal_avg.max()])
        ax3.set_ylim([0, len(horizontal_avg)])
        ax3.invert_yaxis()
        ax3.invert_xaxis()

        plt.tight_layout()
        plt.show()

        return fig, (ax1, ax2, ax3)
    
    @staticmethod
    def _centroid(
        img,
        energy,
        bkg_mean=0,
        factor_ADC=1.2,
        factor_for_ghost_clouds = 0,
        avoid_double=False,
        curve_a=0,
        curve_b=0,
    ):
        """
        Identify single-photon events in a 2D detector image via centroiding.

        Candidate pixels are selected using intensity thresholds derived from
        the photon energy. A 3x3 patch around each candidate is used to compute
        a weighted sub-pixel centroid, and each event is classified as a single
        or double photon hit based on total patch intensity.

        Parameters
        ----------
        img : numpy.ndarray
            2D detector image (raw counts).
        energy : float
            Incident photon energy in eV, used to derive intensity thresholds.
        bkg_mean : float, optional
            Flat background subtracted from `img` before centroiding. Default 0.
        factor_ADC : float, optional
            ADU-to-electron conversion factor. Default 1.2 (ESRF value; TPS
            typically uses ~0.55).
        factor_for_ghost_clouds : float, optional
            Lower bound on the low-intensity threshold, used to discard "ghost
            cloud" artifacts specific to TPS data. Use 0 for ESRF, ~200 for TPS.
            Default 0.
        avoid_double : bool, optional
            If True, disables double-event detection (no event is ever
            classified as a double). Default False.
        curve_a : float, optional
            Linear curvature correction applied to the y-centroid. Default 0.
        curve_b : float, optional
            Quadratic curvature correction applied to the y-centroid. Default 0.

        Returns
        -------
        tuple of (list, list)
            res_shifted : list of (y, x) tuples with the sub-pixel positions of
            all detected photons. Double events are appended twice, matching
            the convention expected by `_bin`.
            double_shifted : list of (y, x) tuples with the positions of
            double-photon events only.
        """

        SpotLOW = max(0.4 * energy / 3.6 / factor_ADC, factor_for_ghost_clouds)  # Multiplication factor * ADU/photon
        SpotHIGH = 1.5 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon
        low_th_px = 0.2 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon
        high_th_px = 1 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon

        if avoid_double == True:
            SpotHIGH = 100000
            print(
                "The double events are not taken into account, the double event threshold is set to "
            )
            print(SpotHIGH)

        img = img - bkg_mean

        gs = 2
        h = gs // 2  # = 1
        cp = np.argwhere(
            (img[h:-h, h:-h] > low_th_px)
            * (img[h:-h, h:-h] < high_th_px)
        ) + np.array([h, h])

        if len(cp) == 0:
            return [], []

        # --- vectorized patch extraction ---
        # offsets: [-1, 0, 1]
        offsets = np.arange(-h, h + 1)
        dy, dx = np.meshgrid(offsets, offsets, indexing='ij')   # (3, 3)

        patch_y = cp[:, 0, None, None] + dy[None]               # (N, 3, 3)
        patch_x = cp[:, 1, None, None] + dx[None]               # (N, 3, 3)

        spots = img[patch_y, patch_x]                           # (N, 3, 3)
        spots_clipped = np.maximum(spots, 0)                    # clip negatives

        # --- local maximum check ---
        # center pixel must be >= every neighbor in the clipped patch
        center_vals = img[cp[:, 0], cp[:, 1]]                   # (N,)
        is_local_max = (spots_clipped > center_vals[:, None, None]).sum(axis=(1, 2)) == 0  # (N,)

        # --- weighted centroid ---
        # x-centroid: weight each x-column by the column sum (sum over rows)
        # y-centroid: weight each y-row   by the row   sum (sum over cols)
        x_weights = spots_clipped.sum(axis=1)                   # (N, 3)  sum over rows
        y_weights = spots_clipped.sum(axis=2)                   # (N, 3)  sum over cols

        x_coords = cp[:, 1, None] + offsets[None]              # (N, 3)
        y_coords = cp[:, 0, None] + offsets[None]              # (N, 3)

        total_weight = spots_clipped.sum(axis=(1, 2))           # (N,)
        safe_weight = np.where(total_weight > 0, total_weight, 1)

        mx = (x_coords * x_weights).sum(axis=1) / safe_weight  # (N,)
        my = (y_coords * y_weights).sum(axis=1) / safe_weight  # (N,)
        my = my - (curve_a + curve_b * mx) * mx                # curvature correction

        # --- event classification ---
        spot_sums = total_weight
        single_mask = is_local_max & (spot_sums > SpotLOW) & (spot_sums <= SpotHIGH)
        double_mask = is_local_max & (spot_sums > SpotHIGH)

        my_single = my[single_mask]
        mx_single = mx[single_mask]
        my_double = my[double_mask]
        mx_double = mx[double_mask]

        # doubles are appended twice to res (same behaviour as the original loop)
        res_my = np.concatenate([my_single, my_double, my_double])
        res_mx = np.concatenate([mx_single, mx_double, mx_double])

        res_shifted    = list(zip(res_my    + 0.5, res_mx    + 0.5))
        double_shifted = list(zip(my_double + 0.5, mx_double + 0.5))

        return res_shifted, double_shifted

    @staticmethod
    def _centroid_parallel(
        imgs,
        energy,
        bkg_mean=0,
        factor_ADC=1.2,
        factor_for_ghost_clouds=0,
        avoid_double=False,
        curve_a=0,
        curve_b=0,
    ):
        """
        Vectorized single photon counting for a stack of images.
        Processes all images simultaneously without an explicit Python loop.
        Patches are always 2D: candidate pixels are compared only to neighbours
        within the same image.

        Parameters
        ----------
        imgs : ndarray, shape (N_img, H, W)
            Stack of 2D detector images.
        energy : float
            Energy of the incident photons in eV.
        bkg_mean : float, optional
            Flat background level to subtract.
        factor_ADC : float, optional
            ADU-to-electron conversion factor.
        factor_for_ghost_clouds : float, optional
            Lower bound for SpotLOW (use > 0 to discard ghost clouds, e.g. 200 for TPS).
        avoid_double : bool, optional
            If True, SpotHIGH is set to infinity (double events are kept as singles).
        curve_a, curve_b : float, optional
            Linear/quadratic curvature correction coefficients.

        Returns
        -------
        res_list : list of ndarray, each shape (M_i, 2)
            One array per input image.  Each row is (y + 0.5, x + 0.5) of a
            detected photon (doubles counted twice, consistent with _centroid).
        double_list : list of ndarray, each shape (D_i, 2)
            Double-event positions for each image.
        """
        SpotLOW  = max(0.4 * energy / 3.6 / factor_ADC, factor_for_ghost_clouds)
        SpotHIGH = 1.5 * energy / 3.6 / factor_ADC
        low_th_px  = 0.2 * energy / 3.6 / factor_ADC
        high_th_px = 1.0 * energy / 3.6 / factor_ADC

        if avoid_double:
            SpotHIGH = 1e9

        imgs = imgs - bkg_mean          # (N_img, H, W)  — does not modify original
        n_imgs = imgs.shape[0]
        h = 1                           # half patch-size → 3×3 patches

        # ------------------------------------------------------------------ #
        # 1. Candidate pixels                                                 #
        # ------------------------------------------------------------------ #
        # mask shape: (N_img, H-2, W-2)
        inner = imgs[:, h:-h, h:-h]
        mask  = (inner > low_th_px) & (inner < high_th_px)

        # cp: (N_cand, 3)  columns → [img_idx, y, x]
        cp = np.argwhere(mask) + np.array([0, h, h])

        empty_res    = [np.empty((0, 2)) for _ in range(n_imgs)]
        empty_double = [np.empty((0, 2)) for _ in range(n_imgs)]

        if len(cp) == 0:
            return empty_res, empty_double

        img_idx = cp[:, 0]              # (N,)
        cy      = cp[:, 1]              # (N,)
        cx      = cp[:, 2]              # (N,)

        # ------------------------------------------------------------------ #
        # 2. Vectorized 3×3 patch extraction                                 #
        # ------------------------------------------------------------------ #
        offsets = np.arange(-h, h + 1)                          # [-1, 0, 1]
        dy, dx  = np.meshgrid(offsets, offsets, indexing='ij')  # (3, 3)

        patch_y = cy[:, None, None]      + dy[None]             # (N, 3, 3)
        patch_x = cx[:, None, None]      + dx[None]             # (N, 3, 3)
        patch_i = img_idx[:, None, None] * np.ones((1, 3, 3), dtype=np.intp)  # (N, 3, 3)

        spots         = imgs[patch_i, patch_y, patch_x]         # (N, 3, 3)
        spots_clipped = np.maximum(spots, 0)

        # ------------------------------------------------------------------ #
        # 3. Local-maximum check (within the same image's patch)             #
        # ------------------------------------------------------------------ #
        center_vals = imgs[img_idx, cy, cx]                      # (N,)
        is_local_max = (
            (spots_clipped > center_vals[:, None, None]).sum(axis=(1, 2)) == 0
        )                                                        # (N,)

        # ------------------------------------------------------------------ #
        # 4. Weighted centroid                                                #
        # ------------------------------------------------------------------ #
        x_weights    = spots_clipped.sum(axis=1)                 # (N, 3)  sum over rows
        y_weights    = spots_clipped.sum(axis=2)                 # (N, 3)  sum over cols
        x_coords     = cx[:, None] + offsets[None]              # (N, 3)
        y_coords     = cy[:, None] + offsets[None]              # (N, 3)
        total_weight = spots_clipped.sum(axis=(1, 2))            # (N,)
        safe_weight  = np.where(total_weight > 0, total_weight, 1)

        mx = (x_coords * x_weights).sum(axis=1) / safe_weight   # (N,)
        my = (y_coords * y_weights).sum(axis=1) / safe_weight   # (N,)
        my = my - (curve_a + curve_b * mx) * mx                 # curvature correction

        # ------------------------------------------------------------------ #
        # 5. Event classification                                             #
        # ------------------------------------------------------------------ #
        single_mask = is_local_max & (total_weight > SpotLOW) & (total_weight <= SpotHIGH)
        double_mask = is_local_max & (total_weight > SpotHIGH)

        # ------------------------------------------------------------------ #
        # 6. Build per-image output lists                                     #
        # ------------------------------------------------------------------ #
        res_list    = []
        double_list = []

        for i in range(n_imgs):
            in_img = img_idx == i

            s_mask = in_img & single_mask
            d_mask = in_img & double_mask

            my_s, mx_s = my[s_mask], mx[s_mask]
            my_d, mx_d = my[d_mask], mx[d_mask]

            # doubles appended twice (same convention as _centroid)
            res_my = np.concatenate([my_s, my_d, my_d])
            res_mx = np.concatenate([mx_s, mx_d, mx_d])

            res_list.append(
                np.column_stack([res_my + 0.5, res_mx + 0.5]) if len(res_my) > 0
                else np.empty((0, 2))
            )
            double_list.append(
                np.column_stack([my_d + 0.5, mx_d + 0.5]) if len(my_d) > 0
                else np.empty((0, 2))
            )

        return res_list, double_list

    @staticmethod
    def _centroid_old(
        img,
        energy,
        bkg_mean=0,
        factor_ADC=1.2,
        factor_for_ghost_clouds = 0,
        avoid_double=False,
        curve_a=0,
        curve_b=0,
    ):
        """
        Reference (non-vectorized) implementation of `_centroid`, kept for
        validation purposes. Loops over candidate pixels one at a time instead
        of using the vectorized patch extraction in `_centroid`; produces the
        same results, just slower. See `_centroid` for the full parameter and
        return documentation, which applies identically here.

        Parameters
        ----------
        img : numpy.ndarray
            2D detector image (raw counts).
        energy : float
            Incident photon energy in eV, used to derive intensity thresholds.
        bkg_mean : float, optional
            Flat background subtracted from `img` before centroiding. Default 0.
        factor_ADC : float, optional
            ADU-to-electron conversion factor. Default 1.2 (ESRF value; TPS
            typically uses ~0.55).
        factor_for_ghost_clouds : float, optional
            Lower bound on the low-intensity threshold, used to discard "ghost
            cloud" artifacts specific to TPS data. Use 0 for ESRF, ~200 for TPS.
            Default 0.
        avoid_double : bool, optional
            If True, disables double-event detection (no event is ever
            classified as a double). Default False.
        curve_a : float, optional
            Linear curvature correction applied to the y-centroid. Default 0.
        curve_b : float, optional
            Quadratic curvature correction applied to the y-centroid. Default 0.

        Returns
        -------
        tuple of (list, list)
            res_shifted : list of (y, x) tuples with the sub-pixel positions of
            all detected photons. Double events are appended twice, matching
            the convention expected by `_bin`.
            double_shifted : list of (y, x) tuples with the positions of
            double-photon events only.
        """

        SpotLOW = max(0.4 * energy / 3.6 / factor_ADC, factor_for_ghost_clouds)  # Multiplication factor * ADU/photon
        SpotHIGH = 1.5 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon
        low_th_px = 0.2 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon
        high_th_px = 1 * energy / 3.6 / factor_ADC  # Multiplication factor * ADU/photon

        if avoid_double == True:
            SpotHIGH = 100000
            print(
                "The double events are not taken into account, the double event threshold is set to "
            )
            print(SpotHIGH)


        img = img - bkg_mean

        gs = 2
        cp = np.argwhere(
            (img[gs // 2 : -gs // 2, gs // 2 : -gs // 2] > low_th_px)
            * (img[gs // 2 : -gs // 2, gs // 2 : -gs // 2] < high_th_px)
        ) + np.array([gs // 2, gs // 2])


        res = []
        double = []

        for cy, cx in cp:
            spot = img[cy - gs // 2 : cy + gs // 2 + 1, cx - gs // 2 : cx + gs // 2 + 1]
            spot[spot < 0] = 0
            if (spot > img[cy, cx]).sum() == 0:
                mx = np.average(
                    np.arange(cx - gs // 2, cx + gs // 2 + 1), weights=spot.sum(axis=0)
                )
                my = np.average(
                    np.arange(cy - gs // 2, cy + gs // 2 + 1), weights=spot.sum(axis=1)
                )
                my -= (curve_a + curve_b * mx) * mx
                if (spot.sum() > SpotLOW) * (spot.sum() <= SpotHIGH):
                    res.append((my, mx))
                elif spot.sum() > SpotHIGH:
                    res.append((my, mx))
                    res.append((my, mx))
                    double.append((my, mx))

        res_shifted = [(my + 0.5, mx + 0.5) for (my, mx) in res]
        double_shifted = [(my + 0.5, mx + 0.5) for (my, mx) in double]
        
        return res_shifted, double_shifted
    
    @staticmethod
    def _bin(
        p_pos_list, 
        image_size_h, 
        image_size_v, 
        subdivide_bins_factor_x, 
        subdivide_bins_factor_y,
        vertical_shift = 0):
        """
        Bin a list of sub-pixel photon positions into a 2D histogram.

        Parameters
        ----------
        p_pos_list : list of tuple
            (y, x) sub-pixel photon coordinates, as returned by `_centroid`.
        image_size_h : int
            Width of the original detector image (x-extent / number of
            columns), in pixels.
        image_size_v : int
            Height of the original detector image (y-extent / number of
            rows), in pixels.
        subdivide_bins_factor_x : float
            Number of sub-pixel bins per pixel along the x-axis.
        subdivide_bins_factor_y : float
            Number of sub-pixel bins per pixel along the y-axis.
        vertical_shift : float, optional
            Vertical shift (in pixels) subtracted from the y-coordinates of
            the photon positions before binning, typically derived from
            cross-correlation analysis. Leave at 0 if cross-correlation has
            not been performed. Default 0.

        Returns
        -------
        tuple of (numpy.ndarray, int)
            hist_p : 2D histogram of photon counts, dtype float32, shape
            (image_size_v * subdivide_bins_factor_y, image_size_h * subdivide_bins_factor_x).
            photon_count : total number of photon positions binned.
        """
        # Check if p_pos_list is empty
        if not p_pos_list:
            # If empty, set hist_p to an array of zeros with shape (image_size_v, image_size_h)
            hist_p = np.zeros(
                (
                    int(image_size_v * subdivide_bins_factor_y),
                    int(image_size_h * subdivide_bins_factor_x),
                )
            )
            photon_count = 0
        else:
            # Convert p_pos_list to a NumPy array
            p_pos_array = np.array(p_pos_list)
            p_pos_array[:, 0] -= vertical_shift/subdivide_bins_factor_y

            # Define bin edges
            x_edges_pht = list(np.arange(0, image_size_h, 1 / subdivide_bins_factor_x)) + [
                image_size_h
            ]
            y_edges_pht = list(np.arange(0, image_size_v, 1 / subdivide_bins_factor_y)) + [
                image_size_v
            ]

            # Create histogram
            hist_p, _, _ = np.histogram2d(
                p_pos_array[:, 0], p_pos_array[:, 1], bins=(y_edges_pht, x_edges_pht)
            )
            photon_count = len(p_pos_array)

        return hist_p.astype(np.float32), photon_count


class H5_ESRF_Image(RIXS_Image):
    def __init__(self, file_path):
        """
        Initialize an HDF5 image object from a file path.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 image file
        """

        super().__init__()
        self.file_path = file_path
        



class EDF_Image(RIXS_Image):
    def __init__(self, file_path):
        """
        Initialize an EDF image object from a file path.

        Parameters
        ----------
        file_path : str
            Path to the EDF image file
        """

        super().__init__()
        self.file_path = file_path
        
        self.attributes = {}

        self.raw_data = self._get_raw_data()
        self.n_images = self.raw_data.shape[0]
        self._get_run_number()
        self._get_image_number()
        self._get_absolute_image_number()
        self._get_normalization_factor()
        self._get_attributes()

    def _get_raw_data(self):
        """
        Load a raw ESRF detector image from an EDF file.

        Reads the image via `fabio`, flips it vertically to match the
        expected orientation, and reorganizes the file header into
        `self.header`. Adds a leading axis if the loaded image is 2D.

        Returns
        -------
        numpy.ndarray
            The loaded raw image, shape (n_images, height, width).
        """
        file = fabio.open(self.file_path)
        self.raw_data = np.flipud(file.data)
        self.header = self._reorganize_header(file.header)

        if self.raw_data.ndim == 2:
            self.raw_data = np.expand_dims(self.raw_data, axis=0)

        return self.raw_data


    def _get_run_number(self):
        """
        Get the run number from the header
        """
        self.run_number = int(self.header["scan_no"])
        return self.run_number
    
    def _get_image_number(self):
        """
        Get the image number from the header
        """
        self.image_number = int(self.header["point_no"])
        return self.image_number
    
    def _get_absolute_image_number(self):
        """
        Get the absolute image number from the header
        """
        # self.absolute_image_number = int(self.header["run"])
        self.absolute_image_number = int(self.file_path[-8:-4])
        return self.absolute_image_number

    def _get_normalization_factor(self):
        """
        Get the normalization factor from the header
        """

        self.normalization_factor = self.header['mir'] / 1E6

        return self.normalization_factor
    
    def _get_energy(self):
        """
        Get the energy from the header
        """
        if "energy" not in self.attributes:
            raise ValueError("No energy found in the header")
        else:
            self.energy = self.attributes["energy"]
        
        return self.energy
    
    def _get_attributes(self):
        """
        Get the attributes from the header and map them using metadata_name_mapping.
        """

        metadata_name_mapping = {
            "th": "th",
            "chi": "chi",
            "phi": "phi",
            "tth": "rtth",
            "energy": "energy",
            "x": "xsam",
            "y": "ysam",
            "z": "zsam",
            "T": "tstage",
        }
        self.attributes = {}

        # Map the attributes using metadata_name_mapping
        for key, mapped_key in metadata_name_mapping.items():
            if key in self.header:
                self.attributes[key] = self.header[mapped_key]

        # Retrieve polarization motor values if present
        # Try to get polarization motor values from either HU88AP/HU88CP or HU70AP/HU70CP
        if "HU88AP" in self.header:
            ap = float(self.header["HU88AP"])
            polarization = _determine_polarization(ap, ap)
        elif "HU70AP" in self.header:
            ap = float(self.header["HU70AP"])
            polarization = _determine_polarization(ap, ap)
        else:
            print("Warning: Polarization motor values not found. Polarization could not be determined.")
            polarization = "Unknown"

        self.attributes["polarization"] = polarization
        
        return self.attributes
    
    def single_photon_counting(self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=1.2,
            vertical_shift = 0,
            subtract_background_from_img=False,
            subtract_background_from_corner=False,
            dark_img=None,
            plot_raw_image=False):
        """
        ESRF-specific wrapper around `RIXS_Image.single_photon_counting`.

        Forces `factor_ADC=1.2`, a fixed flat background of 300.52, and
        disables `subtract_background_from_img`/`subtract_background_from_corner`,
        since ESRF images use a flat background rather than a dark-image
        subtraction. Emits a warning if `subtract_background_from_img` is
        True or if a non-default `factor_ADC` was requested.

        Parameters
        ----------
        See `RIXS_Image.single_photon_counting` for the full parameter list.
        ROI, binning, and photon-counting parameters behave identically;
        only the background-handling defaults differ for ESRF data.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray or float)
            Same as `RIXS_Image.single_photon_counting`.
        """
        
        if subtract_background_from_img:
            print("Warning: ESRF images do NOT need a dark image. Not using it. Setting flat background to 300.52...")

        if factor_ADC != 1.2:
            print("Warning: factor_ADC set 1.2, standard for ESRF.")
        
        return super().single_photon_counting(curve_a=curve_a, 
                curve_b=curve_b,
                roi_x=roi_x,
                roi_y=roi_y,
                subdivide_bins_factor_x=subdivide_bins_factor_x, 
                subdivide_bins_factor_y=subdivide_bins_factor_y,
                factor_ADC=1.2,
                vertical_shift=vertical_shift,
                subtract_background_from_img=False,
                subtract_background_from_corner=False,
                bkg=300.52,
                plot_raw_image=plot_raw_image)
    
    @staticmethod
    def _reorganize_header(header):
        """
        Flatten an EDF header into a single dict of scalar/string values.

        The EDF header stores motor and counter names/positions as
        space-separated strings under 'counter_mne'/'motor_mne' (names) and
        'counter_pos'/'motor_pos' (values). This unzips those parallel lists
        and merges them into the header dict as individual
        `{motor_or_counter_name: float_value}` entries, alongside all other
        header fields (kept as strings).

        Parameters
        ----------
        header : dict
            Raw EDF header as returned by `fabio.open(...).header`.

        Returns
        -------
        dict
            Flattened header: string values for ordinary fields, plus one
            float entry per named motor/counter.
        """
        # Keys that need special handling (splitting into lists)
        keys_to_split = ['counter_mne', 'motor_mne']
        values_to_split = ['counter_pos', 'motor_pos']
        # Extract all header items as strings, except for the keys that need splitting
        header_dict = {k: str(v) for k, v in header.items() if k not in keys_to_split and k not in values_to_split}

        # Add the split keys separately as lists
        names = []
        for key in keys_to_split:
            if key in header:
                names += str(header[key]).split()

        values = []
        for key in values_to_split:
            if key in header:
                values += [float(x) for x in header[key].split()]

        for name, value in zip(names, values):
            header_dict[name] = value

        return header_dict
    

class TPS_Image(RIXS_Image):
    def __init__(self, file_path, file_path_background = None):
        """
        Load a TPS detector image and its associated metadata.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the raw detector image(s).
        file_path_background : str, optional
            Path to a background/dark HDF5 file. Stored on
            `self.file_path_background` but not otherwise used within this
            class. Default None.
        """
        super().__init__()
        self.file_path = file_path
        self.file_path_background = file_path_background if file_path_background is not None else None

        self._get_raw_data()
        self.n_images = self.raw_data.shape[0]
        self._get_attributes()
        self._get_normalization_factor()
        self._get_run_number()
        self._get_energy()

    def _get_raw_data(self):
        """
        Load raw detector images from the TPS HDF5 file.

        Reads the "data" dataset, adds a leading axis if only a single 2D
        image is present, and rotates each image 90 degrees clockwise to
        match the expected orientation. Result is stored on `self.raw_data`.

        Raises
        ------
        FileNotFoundError
            If `self.file_path` does not exist.
        ValueError
            If the loaded data is not 2D or 3D.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")

        with h5py.File(self.file_path, "r") as file:
            self.raw_data = file["data"][:].astype(np.float32)
            if self.raw_data.ndim < 3:
                self.raw_data = np.expand_dims(self.raw_data, axis=0)

            self.raw_data = np.array([np.rot90(raw, k=-1) for raw in self.raw_data])
            if self.raw_data.ndim != 3:
                raise ValueError("The data must be 2D or 3D")
            print(f"Found {self.raw_data.shape[0]} images in the file {os.path.basename(self.file_path)}")


    def _get_run_number(self):
        """
        Get the run number from the image file and store it on the instance.

        Sets `self.run_number` from the "f" attribute. Does not return a
        value (implicitly returns None) — read `self.run_number` afterwards.

        Raises
        ------
        ValueError
            If no "f" attribute is present in `self.attributes`.
        """
        if "f" not in self.attributes:
            raise ValueError("No run number found in the image file")
        
        self.run_number = self.attributes["f"]
                

    def _get_attributes(self):
        """
        Parse the image file's header attributes and store them on the instance.

        Reads the 'header' HDF5 attribute (a colon/comma-separated string of
        "name value" pairs) and populates `self.attributes` as a dict of
        floats keyed by attribute name. Does not return a value (implicitly
        returns None) — read `self.attributes` afterwards.

        Raises
        ------
        ValueError
            If `self.file_path` is None.
        """
        if self.file_path is not None:
            with h5py.File(self.file_path, "r") as file:
                header = file.attrs['header']

            self.attributes = {}
            for item in header.split(":")[1].split(","):
                name, value = item.strip().split(" ")
                self.attributes[name] = float(value)

        else:
            raise ValueError("No file loaded")

    def _get_energy(self):
        """
        Get the incident photon energy from the image file and store it on
        the instance.

        Sets `self.energy` from the "agm" attribute. Does not return a value
        (implicitly returns None) — read `self.energy` afterwards.

        Raises
        ------
        ValueError
            If no "agm" attribute is present in `self.attributes`.
        """
        if "agm" not in self.attributes:
            raise ValueError("No energy found in the image file")
        else:
            self.energy = self.attributes["agm"]


    def _get_normalization_factor(self):
        """
        Retrieve the normalization factor for the TPS image file, giving
        precedence to a pre-computed value if one is stored in the file.

        If the "normalization_factor" HDF5 attribute is present, it is used
        directly. Otherwise, the factor is derived from the "Iph" attribute
        (scaled by 1e9). Either way, the result is stored on
        `self.normalization_factor` and also returned.

        Returns
        -------
        float
            The normalization factor.

        Raises
        ------
        ValueError
            If neither "normalization_factor" nor "Iph" can be found.
        """
        with h5py.File(self.file_path, "r") as file:
            if "normalization_factor" in file.attrs:
                self.normalization_factor = file.attrs["normalization_factor"]
                return self.normalization_factor

        if "Iph" not in self.attributes:
            raise ValueError("No normalization factor found in the image file")
        else:
            self.normalization_factor = self.attributes["Iph"]*1E9
        return self.normalization_factor

    def single_photon_counting(
            self, 
            curve_a, curve_b=0,
            roi_x=(0,2048),roi_y=(0,2048),
            roi_x_for_dark=(1600,1800), roi_y_for_dark=(250,1800),
            subdivide_bins_factor_x=1, subdivide_bins_factor_y=2.7,
            factor_ADC=0.55,
            vertical_shift = 0,
            subtract_background_from_img=False,
            subtract_background_from_corner=False,
            dark_img=None,
            plot_raw_image=False):
        """
        TPS-specific wrapper around `RIXS_Image.single_photon_counting`.

        Warns if `factor_ADC` differs from the expected TPS value (0.55) or
        if no background subtraction method
        (`subtract_background_from_img`/`subtract_background_from_corner`)
        is enabled, since TPS data requires background subtraction. Does
        not otherwise change behavior — all parameters are passed straight
        through to the base implementation.

        Parameters
        ----------
        See `RIXS_Image.single_photon_counting` for the full parameter list.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray or float)
            Same as `RIXS_Image.single_photon_counting`.
        """
        
        if factor_ADC != 0.55:
            print("Warning: factor_ADC should be 0.55 for TPS data.")

        if not subtract_background_from_img and not subtract_background_from_corner:
            print("Warning: No background subtraction is applied. This should be done for TPS data.")
        
        return  super().single_photon_counting( 
                        curve_a=curve_a,
                        curve_b=curve_b,
                        roi_x=roi_x,
                        roi_y=roi_y,
                        roi_x_for_dark=roi_x_for_dark,
                        roi_y_for_dark=roi_y_for_dark,
                        subdivide_bins_factor_x=subdivide_bins_factor_x, 
                        subdivide_bins_factor_y=subdivide_bins_factor_y,
                        factor_ADC=factor_ADC,
                        vertical_shift=vertical_shift,
                        subtract_background_from_img=subtract_background_from_img,
                        subtract_background_from_corner=subtract_background_from_corner,
                        dark_img=dark_img,
                        plot_raw_image=plot_raw_image)
    

class DLS_Image(RIXS_Image):
    def __init__(self, file_path):
        """
        Load a Diamond Light Source (i21) RIXS image and its metadata from
        a NeXus file.

        Parameters
        ----------
        file_path : str
            Path to the .nxs file. The run number is extracted from this
            path (expects an "i21-<digits>" pattern).
        """

        super().__init__()
        self.file_path = file_path

        start_time = time.time()
        print(f"Loading DLS images from {self.file_path}...")
        self._get_run_number()
        self.raw_data = self._get_raw_data()
        self.n_images = self.raw_data.shape[0]
        self._get_normalization_factor()
        self._get_attributes()
        elapsed_time = time.time() - start_time
        print(f"DLS images and attributes loaded successfully.\nElapsed time: {elapsed_time:.2f} seconds.")

    def _get_run_number(self):
        """
        Extracts the run number from the image file.
        """
        match = re.search(r'i21-(\d+)', self.file_path)
        if match:
            self.run_number = int(match.group(1))
        else:
            raise ValueError(f"Could not extract run number from file path: {self.file_path}")
        return self.run_number

    def _get_raw_data(self):
        """
        Load raw detector images from a NeXus file.

        Reads the Andor detector data from the 'entry' group of the NeXus
        file, falling back to 'entry1' if the first path is not found. A
        leading axis is added if only a single 2D image is present. The
        returned array is assigned to `self.raw_data` by the caller
        (`__init__`) — this method does not set any instance attributes
        itself.

        Returns
        -------
        numpy.ndarray, dtype float32
            Stack of raw detector images, shape (n_images, height, width).

        Raises
        ------
        ValueError
            If the detector data cannot be found under either 'entry' or
            'entry1'.
        """

        with nxload(self.file_path,mode='r') as f:
            try:
                raw_data = f.entry['andor']['data'].nxvalue 
            except:
                try:
                    raw_data = f.entry1['andor']['data'].nxvalue
                except:
                    raise ValueError("Could not retrieve raw images from NeXus file. Both 'entry' and 'entry1' paths failed.")

            if raw_data.ndim == 2:
                raw_data = np.expand_dims(raw_data, axis=0)

        print(f"Retrieved {raw_data.shape[0]} images from NeXus file.")

        return raw_data.astype(np.float32)

    def _get_attributes(self):
        """
        Get the attributes from the .nxs file.
        """
        source_nexus = ["entry/", "entry1/"]
        metadata_paths = {
            "th": "instrument/manipulator/th",
            "chi": "instrument/manipulator/chi",
            "phi": "instrument/manipulator/phi",
            "tth": "instrument/spectrometer/armtth",
            'H': "/Q/H",
            'K': "/Q/K",
            'L': "/Q/L",
            "energy": "instrument/pgm/energy",
            "x": "instrument/manipulator/x",
            "y": "instrument/manipulator/y",
            "z": "instrument/manipulator/z",
            "T": "instrument/lakeshore336/sample",
            "polarization": "instrument/id/polarisation",
            "count_time": "instrument/m4c1/count_time",
        }
        self.attributes = {}

        # Map the attributes using metadata_name_mapping
        with nxload(self.file_path,mode='r') as f:
            for key, mapped_key in metadata_paths.items():
                try:
                    self.attributes[key] = f[source_nexus[0] + mapped_key].nxvalue
                except (KeyError, AttributeError, Exception):
                    try:
                        self.attributes[key] = f[source_nexus[1] + mapped_key].nxvalue
                    except (KeyError, AttributeError, Exception):
                        print(f"Warning: Could not retrieve {mapped_key} from NeXus file")
                        self.attributes[key] = np.nan

            #Retrieve the counting time
            try: 
                self.data_count_time = f.entry['instrument']['andor']['count_time'].nxvalue
            except:
                try:
                    self.data_count_time = f.entry1['instrument']['andor']['count_time'].nxvalue
                except:
                    print("WARNING: Could not retrieve count time from NeXus file")
                    self.data_count_time = 120
        
        _ = self._get_energy()
        
        return self.attributes
    
    def _get_energy(self):
        """
        Get the incident photon energy from the already-loaded attributes.

        Returns
        -------
        float
            Incident photon energy in eV (`self.attributes["energy"]`).
        """
        self.energy = self.attributes["energy"]
        return self.energy

    def _get_normalization_factor(self):
        """
        Retrieve the per-image normalization factor (m4c1 monitor counts)
        from the NeXus file.

        Tries the 'entry' group first, falling back to 'entry1'. If neither
        is found, prints a warning and defaults to 1 for every image. A
        scalar result is broadcast to one value per image.

        Returns
        -------
        numpy.ndarray
            Normalization factor(s), shape (n_images,).
        """
        with nxload(self.file_path,mode='r') as f:  
            try:
                self.normalization_factor = f.entry['instrument']['m4c1']['m4c1'].nxvalue
            except:
                try:
                    self.normalization_factor = f.entry1['instrument']['m4c1']['m4c1'].nxvalue
                except:
                    print("WARNING: Could not retrieve normalization factor from NeXus file")
                    self.normalization_factor = 1
        if isinstance(self.normalization_factor, float):
            self.normalization_factor = [self.normalization_factor]*self.n_images

        self.normalization_factor = np.array(self.normalization_factor)

        return self.normalization_factor

    def _remove_bkg_and_filter(self, **kwargs):
        """
        Removes background and filters from spikes the raw image data using specified parameters.
        Background is fitted to filtered raw images with a linear model (translated and scaled).
        Parameters
        ----------
        **kwargs : dict
            Dictionary containing parameters for background removal and filtering.
            Expected keys include:
            - file_path_dark : str or list of str
                Path(s) to the dark image file(s)
            - dark_from_processed_file : bool
                If True, use dark image from processed file
            - hdf5_path_to_dark : str or None
                Path within HDF5 file to the dark image
            - dark_median_filter_kernel_size : list of int
                Kernel size for median filtering the dark image
            - dark_smoothing_parameters : list of int
                Parameters for smoothing the dark image
            - filtertype : str
                Type of filter to use ('gaussian', 'median', etc.)
            - mean_before_spike_removal_dark : bool
                If True, compute mean before spike removal in dark image
            - index_start_fit_bkg : int
                Index to start fitting the background
            - curve_a : float
                Linear coefficient for curvature correction
            - curve_b : float
                Quadratic coefficient for curvature correction
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray or float)
            imgs_processed : the background-subtracted, filtered, and
            curvature-corrected image stack.
            normalization_factor : normalization factor(s) for the images
            (unchanged by this method).
        """
        self._get_dark_image(file_path_dark=kwargs.get('file_path_dark'),
                             dark_from_processed_file=kwargs.get('dark_from_processed_file', False),
                             hdf5_path_to_dark=kwargs.get('hdf5_path_to_dark', None),
                             dark_median_filter_kernel_size=kwargs.get('dark_median_filter_kernel_size', [5,15]),
                             dark_smoothing_parameters=kwargs.get('dark_smoothing_parameters', [3,15]),
                             filtertype=kwargs.get('filtertype', 'gaussian'),
                             mean_before_spike_removal_dark=kwargs.get('mean_before_spike_removal_dark', True))

        self._filter_img(kwargs.get('img_median_filter_kernel', [5,3,5]), 
                         kwargs.get('spikes_threshold', 1.4))

        self._fit_bkg_sklearn(index_start_fit_bkg=kwargs.get('index_start_fit_bkg', 1192))
        self._subtract_dark()
        self._correct_curvature(curve_a=kwargs.get('curve_a', 0), curve_b=kwargs.get('curve_b', 0))

        #flip the direction of image
        self.imgs_processed = np.array([np.flipud(img) for img in self.imgs_processed])

        return self.imgs_processed, self.normalization_factor


    def _get_dark_image(self,
                        file_path_dark,
                        dark_from_processed_file=False,
                        hdf5_path_to_dark=None,
                        dark_median_filter_kernel_size=[5,15], 
                        dark_smoothing_parameters=[3,15], filtertype='gaussian', 
                        mean_before_spike_removal_dark=True):
        """
        Obtain the dark image to use for background subtraction, either by
        loading an already-processed dark image or by loading raw dark runs
        and filtering/smoothing them.

        Result is stored on `self.dark_img`; does not return a value.

        Parameters
        ----------
        file_path_dark : str or list of str
            Path(s) to the dark-image file(s). If `dark_from_processed_file`
            is True, a single HDF5 path with a pre-processed dark image; 
            otherwise one or more NeXus files with raw dark-image runs.
        dark_from_processed_file : bool, optional
            If True, load an already-processed dark image via
            `_get_processed_dark_img` instead of processing raw runs.
            Default False.
        hdf5_path_to_dark : str, optional
            Path within the HDF5 file to the pre-processed dark image, used
            only when `dark_from_processed_file` is True. Default None.
        dark_median_filter_kernel_size : list of int, optional
            Median filter kernel size passed to
            `_filter_and_smooth_dark_image`. Default [5, 15].
        dark_smoothing_parameters : list of int, optional
            Smoothing filter parameters passed to
            `_filter_and_smooth_dark_image`. Default [3, 15].
        filtertype : str, optional
            Smoothing filter type passed to `_filter_and_smooth_dark_image`.
            Default 'gaussian'.
        mean_before_spike_removal_dark : bool, optional
            Passed to `_filter_and_smooth_dark_image`. Default True.
        """
        
        
        if dark_from_processed_file:
            self.dark_img = self._get_processed_dark_img(file_path_dark,
                                                        hdf5_path_to_dark)
        else:
            file_path_dark = file_path_dark if isinstance(file_path_dark, list) else [file_path_dark]
            dark_img_raw, _ = self._get_dark_img_from_nxs(file_path_dark) #loading the dark images
            self._filter_and_smooth_dark_image(dark_img_raw,
                                                kernel_size = dark_median_filter_kernel_size, 
                                                filter_parameter = dark_smoothing_parameters, 
                                                filtertype=filtertype,
                                                mean_before_spike_removal_dark=mean_before_spike_removal_dark)

        # #fit the dark image to the raw_data
        # self._fit_bkg_sklearn(index_start_fit_bkg=index_start_fit_bkg)
        # self._subtract_dark()
        # print("Dark image processed successfully.")

        

    @staticmethod
    def _get_processed_dark_img(dark_hdf5_filename,
                               path_to_dark=None):
        """
        Load an already-processed dark image directly from an HDF5 file,
        with no further filtering or smoothing applied.

        Parameters
        ----------
        dark_hdf5_filename : str
            Path to the HDF5 file containing the processed dark image.
        path_to_dark : str, optional
            Path within the HDF5 file to the dark image dataset. Defaults
            to "dark_no_spikes_filtered" if not given.

        Returns
        -------
        numpy.ndarray
            The processed dark image, as stored in the file.
        """

        print(f"Using dark image from hdf file {dark_hdf5_filename}. \nNo processing done. \n")
        #get dark from a hdf file
        with h5py.File(dark_hdf5_filename, "r") as f:
            if path_to_dark is None:
                dark_img = f["dark_no_spikes_filtered"][:, :]
            else:
                dark_img = f[path_to_dark][:, :]

        return np.array(dark_img)

    @staticmethod
    def _get_dark_img_from_nxs(file_path_dark): #backround
        """
        Load and concatenate raw dark images from one or more NeXus files.

        Parameters
        ----------
        file_path_dark : list of str
            Paths to the NeXus files containing dark-image runs to load.

        Returns
        -------
        tuple of (numpy.ndarray, float)
            dark_img_raw : concatenated stack of raw dark images (not yet
            averaged), dtype float.
            dark_count_time : total accumulated counting time (seconds)
            across all loaded dark-image runs.
        """
        print(f"Retrieving dark images.")
        start_time = time.perf_counter()

        dark_img_raw = np.empty((0,2048,2048)) #initializing the array
        dark_count_time = 0
        for dark_path in file_path_dark:
            print(f"Retrieving dark image from file {dark_path}")
            with nxload(dark_path,mode='r') as f: #loading background
                try:
                    dark_img_raw = np.concatenate((dark_img_raw, f.entry['andor']['data'].nxvalue))
                except:
                    dark_img_raw = np.concatenate((dark_img_raw, f.entry1['andor']['data'].nxvalue))

                try:
                    dark_count_time += f.entry['instrument']['m4c1']['count_time'].nxvalue #count time
                except:
                    try:
                        dark_count_time += f.entry1['instrument']['m4c1']['count_time'].nxvalue
                    except:
                        dark_count_time += 180

        dark_img_raw = dark_img_raw.astype(float)
        #self.dark_img = self.dark_img.mean(axis=0)

        print(f"Total number of dark images retrieved: {dark_img_raw.shape[0]}")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

        return dark_img_raw, dark_count_time

    def _filter_and_smooth_dark_image(self, 
                                      dark_img_raw, 
                                      kernel_size=[5,15], 
                                      filter_parameter=[3,15], filtertype='gaussian', 
                                      mean_before_spike_removal_dark=True):
        """
        Filters spikes from the dark image and applies smoothing filters.

        Result is stored on `self.dark_img`; the method does not return a
        value.

        Parameters
        ----------
        dark_img_raw : numpy.ndarray
            Stack of raw dark images, shape (n_images, height, width), as
            returned by `_get_dark_img_from_nxs`.
        kernel_size : list of int, default=[5, 15]
            Dimensions for median filter kernel. If 2D [vertical_pixels, horizontal_pixels],
            will be converted to 3D [1, vertical_pixels, horizontal_pixels]
        filter_parameter : list of int, default=[3, 15]
            Parameters for additional filtering:
            - For gaussian filter: sigma values [vertical, horizontal]
            - For FFT filter: cutoff frequencies [vertical, horizontal]
            - For butterworth filter: cutoff frequencies [vertical, horizontal]
        filtertype : str, default='gaussian'
            Type of filter to apply after median filtering
            - 'gaussian': Gaussian smoothing with sigma values
            - 'fft': Fourier transform based filter with cutoff frequencies
            - 'butterworth': Butterworth filter with cutoff frequencie
        mean_before_spike_removal_dark : bool, default=True
            If True, averages dark images before spike removal
            If False, removes spikes from each image independently, then averages
        """
        start_time = time.perf_counter()
        # Initialize arrays for normalization parameters
        aa = np.zeros(dark_img_raw.shape[0])
        bb = np.zeros(dark_img_raw.shape[0])
        dark_norm = np.zeros_like(dark_img_raw)

        # Ensure kernel_size is 3D
        if len(kernel_size) == 2:
            kernel_size = [1] + kernel_size

        if mean_before_spike_removal_dark:
            # Average dark images first, then remove spikes
            print("Averaging dark images before spike removal.")
            dark_img_raw = dark_img_raw.mean(axis=0)
            if kernel_size[1]+kernel_size[2] > 1.0:
                print(f"Removing spikes from dark image with median filter: kernel_parameter = {kernel_size[1]}x{kernel_size[2]}.")
                aa = dark_img_raw[1600:1700,:].mean()
                bb = dark_img_raw[100:200,:].mean()-dark_img_raw[1600:1700,:].mean()
                dark_norm = (dark_img_raw-aa)/bb
                dark_med = medfilt2d(dark_norm, kernel_size[1:]) #applying the median filter
                spikes_dark = dark_norm - dark_med #getting the spikes
                dark_filt = np.where(spikes_dark>0.4, dark_med, dark_norm) #filtering the spikes
                dark_filt = dark_filt * bb + aa
            else:
                print("No spike removal for dark image.")
                dark_filt = dark_img_raw.copy()
        else:
            # Remove spikes from each image independently, then average
            print("Removing spikes from each dark image independently.")
            if sum(kernel_size) > 1.0:
                print(f"Removing spikes from dark images with median filter: kernel_parameter = {kernel_size[0]}x{kernel_size[1]}x{kernel_size[2]}.")
                dark_filt = np.zeros_like(dark_img_raw)

                for i in range(dark_img_raw.shape[0]):
                    # Normalize each image independently
                    aa[i] = dark_img_raw[i,1600:1700,:].mean()
                    bb[i] = dark_img_raw[i,100:200,:].mean()-dark_img_raw[i,1600:1700,:].mean()
                    dark_norm[i] = (dark_img_raw[i]-aa[i])/bb[i]

                # Apply 3D median filter to the normalized image
                dark_med = median_filter(dark_norm, size=kernel_size) 
                spikes_dark = dark_norm - dark_med
                dark_filt_temp = np.where(spikes_dark>0.4, dark_med, dark_norm)
                    
                # Rescale back
                for i in range(dark_filt.shape[0]):
                    dark_filt[i] = dark_filt_temp[i] * bb[i] + aa[i]
            else:
                print("No spike removal for dark image.")
                dark_filt = dark_img_raw.copy()
            
            # Take mean after spike removal
            dark_filt = dark_filt.mean(axis=0)

        # Apply smoothing
        if filtertype == 'fft':
            dark_filt = self._apply_fft_filter_and_plot(dark_filt, cutoff_frequency = filter_parameter)
        elif filtertype=='gaussian':
            if filter_parameter[0]+filter_parameter[1]>1.5:
                print(f"Filtering dark image with {filtertype} filter.")
                dark_filt = gaussian_filter(dark_filt, filter_parameter)
            else:
                print("No filtering of image.")
                dark_filt = dark_filt.copy()          
        elif filtertype=='butterworth': 
            dark_filt = self._apply_butterworth_filter_and_plot(dark_filt, cutoff_frequency = filter_parameter, order=2)

        self.dark_img = np.squeeze(dark_filt.mean(axis=0)) if self.dark_img.ndim==3 else dark_filt

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")


    @staticmethod
    def _apply_fft_filter_and_plot(image, cutoff_frequency=[70,70], plot_flag=True):
        """
        Apply a 2D FFT-based low-pass filter to remove high-frequency components from an image.

        Parameters
        ----------
        image : ndarray
            2D input image to be filtered
        cutoff_frequency : list of float, default=[70,70]
            Cutoff frequencies [vertical, horizontal] that define the elliptical mask
            in frequency space. Higher values allow more high frequencies to pass.
        plot_flag : bool, default=True
            If True, displays plots of the original image, FFT magnitude spectrum
            with cutoff boundary, and filtered result.

        Returns
        -------
        ndarray
            The filtered image after applying the low-pass filter
        """
        # Compute the 2D FFT of the image and shift the zero frequency component to the center
        f_transform = fft2(image)
        f_transform = fftshift(f_transform)
        
        # Compute the magnitude spectrum of the FFT
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)
        
        # Get the dimensions of the image
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create the cutoff mask
        mask = np.zeros((rows, cols), np.float32)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt(((i - crow)/cutoff_frequency[0])**2 + ((j - ccol)/cutoff_frequency[1])**2)
                if distance <= 1:
                    mask[i, j] = 1.0

        # Apply the mask to the FFT of the image
        f_transform_filtered = f_transform * mask
        
        # Transform back to the spatial domain
        f_transform_filtered = ifftshift(f_transform_filtered)
        image_filtered = np.abs(ifft2(f_transform_filtered))

        if plot_flag:
            # # Plot the original image, the FFT magnitude spectrum with cutoff, and the filtered image
            # fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # # Original Image
            # im0 = axs[0].imshow(image, cmap='gray', vmin=300, vmax=350)
            # axs[0].set_title("Original Image")
            # axs[0].axis('off')
            
            # # FFT Magnitude Spectrum with Cutoff
            # im1 = axs[1].imshow(magnitude_spectrum, cmap='gray')
            # axs[1].set_title("FFT with Cutoff Frequency")
            # axs[1].set_xlabel("Frequency X")
            # axs[1].set_ylabel("Frequency Y")
            
            # # Draw the cutoff circle
            # theta = np.linspace(0, 2 * np.pi, 100)
            # x = cutoff_frequency[1] * np.cos(theta) + ccol
            # y = cutoff_frequency[0] * np.sin(theta) + crow
            # axs[1].plot(x, y, color='red', linestyle='--', linewidth=2)
            
            # # Filtered Image
            # im2 = axs[2].imshow(image_filtered, cmap='gray')
            # axs[2].set_title("Filtered Image")
            # axs[2].axis('off')

            # # Add colorbar to the plots
            # fig.colorbar(im0, ax=axs[0], orientation='vertical')
            # fig.colorbar(im1, ax=axs[1], orientation='vertical')
            # fig.colorbar(im2, ax=axs[2], orientation='vertical')

            plt.figure()
            plt.plot(magnitude_spectrum[:,ccol])

            plt.show()
            
        return image_filtered

    @staticmethod
    def _apply_butterworth_filter_and_plot(image, cutoff_frequency=50, order=2, plot_flag=True):
        """
        Apply a 2D Butterworth low-pass filter to an image in frequency space.

        Parameters
        ----------
        image : numpy.ndarray
            2D input image to be filtered.
        cutoff_frequency : list of float, default=50
            [vertical, horizontal] cutoff frequencies defining the elliptical
            Butterworth mask in frequency space. Note: the default value of
            50 is a plain int and will raise a `TypeError` if used as-is,
            since the code indexes it as `cutoff_frequency[0]`/`[1]` — always
            pass a 2-element list/tuple in practice.
        order : int, optional
            Order of the Butterworth filter; higher values give a sharper
            transition at the cutoff. Default 2.
        plot_flag : bool, optional
            If True, displays plots of the original image, the FFT magnitude
            spectrum with the cutoff ellipse, and the filtered result.
            Default True.

        Returns
        -------
        numpy.ndarray
            The filtered image after applying the low-pass filter.
        """
        # Compute the 2D FFT of the image and shift the zero frequency component to the center
        f_transform = fft2(image)
        f_transform = fftshift(f_transform)
        
        # Compute the magnitude spectrum of the FFT
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)
        
        # Get the dimensions of the image
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        # Create a grid of frequencies
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(u, v, sparse=False, indexing='ij')

        # Modify the distance calculation to account for different cutoff frequencies
        D = np.sqrt((U / cutoff_frequency[0])**2 + (V / cutoff_frequency[1])**2)

        # Create the Butterworth filter
        mask = 1 / (1 + D**(2 * order))

        # Apply the mask to the FFT of the image
        f_transform_filtered = f_transform * mask
        
        # Transform back to the spatial domain
        f_transform_filtered = ifftshift(f_transform_filtered)
        image_filtered = np.abs(ifft2(f_transform_filtered))

        if plot_flag:
            # Plot the original image, the FFT magnitude spectrum with cutoff, and the filtered image
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original Image
            im0 = axs[0].imshow(image, cmap='gray', vmin=300, vmax=350)
            axs[0].set_title("Original Image")
            axs[0].axis('off')
            
            # FFT Magnitude Spectrum with Cutoff
            im1 = axs[1].imshow(magnitude_spectrum, cmap='gray')
            axs[1].set_title("FFT with Cutoff Frequency")
            axs[1].set_xlabel("Frequency X")
            axs[1].set_ylabel("Frequency Y")
            
            # Draw the cutoff circle
            theta = np.linspace(0, 2 * np.pi, 100)
            x = cutoff_frequency[1] * np.cos(theta) + ccol
            y = cutoff_frequency[0] * np.sin(theta) + crow
            axs[1].plot(x, y, color='red', linestyle='--', linewidth=2)
            
            # Filtered Image
            im2 = axs[2].imshow(image_filtered, cmap='gray')
            axs[2].set_title("Filtered Image")
            axs[2].axis('off')

            # Add colorbar to the plots
            fig.colorbar(im0, ax=axs[0], orientation='vertical')
            fig.colorbar(im1, ax=axs[1], orientation='vertical')
            fig.colorbar(im2, ax=axs[2], orientation='vertical')
            
            plt.show()

        return image_filtered

    def _fit_bkg_sklearn(self,
                        index_start_fit_bkg=1192):
        """
        Fits a linear background model to the detector images using scikit-learn.
        
        This method:
        1. Takes a portion of the dark image and detector images above a threshold index
        2. Fits a linear model (y = ax + b) to each detector image using the dark image as predictor
        3. Stores the fitted parameters (a,b) in self.dark_poly for later background subtraction
        
        The background fitting is done using scikit-learn's LinearRegression to find the optimal
        scaling (a) and offset (b) parameters that relate the dark image to each detector image.
        
        The fit is performed on a subset of the image data starting from index_start_fit_bkg
        to avoid including the signal region in the background fit.
        """

        start_time = time.perf_counter()

        chunk_bkg = self.dark_img[index_start_fit_bkg:-100, :].reshape(-1,1)

        model = LinearRegression(copy_X=True)

        self.dark_poly = np.zeros((2,self.raw_data.shape[0])) #initializing the array
        for num_image in range(0,self.raw_data.shape[0]):
            # Flatten the matrices
            img_flat = self.raw_data[num_image,index_start_fit_bkg:-100,:].reshape(-1,1)

            # Fit the model
            model.fit(chunk_bkg, img_flat)

            # Extract the parameters a and b
            self.dark_poly[0,num_image] = model.coef_[0]
            self.dark_poly[1,num_image] = model.intercept_

        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        print(f"Fitting completed. Scale factor: {self.dark_poly[0]}, Offset: {self.dark_poly[1]}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _fit_bkg(self):
        """
        Fit a linear background model (a * dark + b) per image using
        `scipy.optimize.minimize`, as an alternative to `_fit_bkg_sklearn`.

        Not currently called anywhere in this class, and not functional as
        written: it references `self.index_start_fit_bkg`,
        `self.dark_img_filtered`, `self.imgs_pure`, and `self.imgs`, none of
        which are set anywhere in `DLS_Image` — calling this will raise an
        `AttributeError`. Looks like a leftover alternative implementation
        to `_fit_bkg_sklearn` (see also `_filter_img_old`, `_centroid_old`
        for similar legacy methods, though unlike this one those are still
        internally consistent). Update the referenced attribute names or
        remove this method.
        """

        print(f"Fitting background to spectrum.")
        start_time = time.perf_counter()

        #fit the background
        index_start_fit = self.index_start_fit_bkg 

        # Extract the last 400 rows of K
        chunk_bkg = self.dark_img_filtered[index_start_fit:-100, :]

        for num_image in range(0,self.imgs_pure.shape[0]):
            # calculate chunk of the image
            chunk_img = self.imgs[num_image,index_start_fit:-100,:]

            # Define the objective function
            def objective(params):
                a, b = params
                difference = a * chunk_bkg + b - chunk_img

                return np.sum(difference**2)
            
            # Initial guess for the parameters
            initial_guess = [1, 0]

            # Perform the optimization
            result = minimize(objective, initial_guess)

            # Get the optimal values of a and b
            a_opt, b_opt = result.x
            self.dark_poly[0,num_image] = a_opt
            self.dark_poly[1,num_image] = b_opt

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")


    def _filter_img(self, img_median_filter_kernel=[5, 3, 5], spikes_threshold=1.4, n_jobs=None):
        """
        Removes spikes from detector images using a 3D median filter.

        Same behavior as before, but the median filter is parallelized by
        splitting the 3D stack along the vertical spatial axis (axis 1).
        Each worker filters a slab plus a halo (overlap) of `kernel` rows on
        each side so the result is bit-for-bit identical to filtering the whole
        array at once; the halo is cropped off before reassembly.

        Parameters
        ----------
        img_median_filter_kernel : list of int, default=[5,3,5]
            [frames, vertical_pixels, horizontal_pixels] kernel dimensions.
            The frames dimension is capped at min(5, number of images).
        spikes_threshold : float, default=1.4
            Pixels with (original - median)/count_time > threshold are spikes.
        n_jobs : int or None, default=None
            Number of worker processes. None -> os.cpu_count().
        """
        start_time = time.perf_counter()

        if self.data_count_time is None:
            raise Exception("Provide data counting time first.")

        if img_median_filter_kernel[0] + img_median_filter_kernel[1] + img_median_filter_kernel[2] > 1.0:
            k0 = min(self.raw_data.shape[0], 5)
            k1 = img_median_filter_kernel[1]
            k2 = img_median_filter_kernel[2]
            size = (k0, k1, k2)

            print(f"Removing spikes from images: kernel size {k0}x{k1}x{k2}, "
                  f"spikes_threshold={spikes_threshold}.")
            print("Using parallel median_filter of scipy.ndimage (split along axis 1).")

            img_corr_med = _parallel_median_filter(self.raw_data, size, n_jobs=n_jobs)

            spikes = (self.raw_data - img_corr_med) / np.asarray(self.data_count_time).reshape(-1, 1, 1)
            self.imgs_processed = np.where(spikes > spikes_threshold, img_corr_med, self.raw_data)

            print("Found these spikes per image: ", end="")
            for spike_2d in spikes:
                count = np.sum(spike_2d > spikes_threshold)
                print(f"{count}, ", end="")
            print("\n", end="")
        else:
            print("No spike removal.")
            self.imgs_processed = self.raw_data.copy()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")



    def _filter_img_old(self, img_median_filter_kernel=[5,3,5], spikes_threshold=1.4):
        """
        Removes spikes from detector images using a 3D median filter.

        The function applies a median filter to detect and remove anomalous high-intensity pixels (spikes)
        from the detector images. It compares the original images with the median-filtered version to
        identify spikes above a threshold, then replaces those pixels with the median-filtered values.

        Parameters
        ----------
        img_median_filter_kernel : list of int, default=[5,3,5]
            Dimensions of the 3D median filter kernel:
            [frames, vertical_pixels, horizontal_pixels]
            The frames dimension is capped at min(5, number of images)
        
        spikes_threshold : float, default=1.4 
            Threshold for spike detection in counts/second.
            Pixels with (original - median)/count_time > threshold are classified as spikes.
        """
        start_time = time.perf_counter()

        if self.data_count_time is None:
            raise Exception("Provide data counting time first.")
        
        if img_median_filter_kernel[0]+img_median_filter_kernel[1]+img_median_filter_kernel[2]>1.0:
            print(f"Removing spikes from images: kernel size {min(img_median_filter_kernel[0], self.raw_data.shape[0])}x{img_median_filter_kernel[1]}x{img_median_filter_kernel[2]}, spikes_threshold={spikes_threshold}.")

            print("Using median_filter of scipy.ndimage.")
            img_corr_med = median_filter(self.raw_data, size=(min(self.raw_data.shape[0], 5), 
                                                               img_median_filter_kernel[1], img_median_filter_kernel[2])) #applying the median filter
            spikes = (self.raw_data - img_corr_med) / np.asarray(self.data_count_time).reshape(-1, 1, 1) #getting the spikes
            self.imgs_processed = np.where(spikes>spikes_threshold, img_corr_med ,self.raw_data) #filtering the spikes

            print("Found these spikes per image: ", end="")
            for spike_2d in spikes:
                count = np.sum(spike_2d>spikes_threshold)
                print(f"{count}, ", end="")
            print("\n", end="")
        else:
            print(f"No spike removal.")
            self.imgs_processed = self.raw_data.copy()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _subtract_dark(self):
        """
        Subtracts the scaled and offset dark image from each raw image.

        The dark image subtraction follows the formula:
            imgs_processed = imgs - (alpha * dark_img + beta)

        where:
        - alpha (dark_poly[0]): Scaling factor for the dark image
        - beta (dark_poly[1]): Offset/background term
        - dark_img: Filtered dark image
        - imgs: Raw detector images
        - imgs_processed: Background-subtracted images

        The alpha and beta parameters are determined separately for each image
        in the sequence using the fit_bkg() method.

        The subtraction is performed in-place, modifying the imgs_processed attribute.
        """
        
        print(f"Subtracting background from images.")
        start_time = time.perf_counter()

        for num_image in range(0,self.imgs_processed.shape[0]):
            self.imgs_processed[num_image] = self.imgs_processed[num_image] - (self.dark_poly[0,num_image]*self.dark_img + self.dark_poly[1,num_image])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def _correct_curvature(self,
                           curve_a,
                           curve_b=0):
        """
        Corrects the curvature of detector images using a linear/quadratic
        slope correction, in place on `self.imgs_processed`.

        The curvature correction is performed by remapping each pixel
        position using:
            y_new = y - curve_a * x - curve_b * x**2
        where:
        - x, y are the original pixel coordinates
        - y_new is the corrected y-coordinate

        The correction is applied to each image in `self.imgs_processed`
        using 2D histogram binning to remap the intensity values to the
        corrected coordinates. Does not return a value.

        Parameters
        ----------
        curve_a : float
            Linear curvature correction parameter (slope).
        curve_b : float, optional
            Quadratic curvature correction parameter. Default 0.
        """

        print(f"Correcting curvature of image.")
        start_time = time.perf_counter()
        
        for num_image in range(0,self.imgs_processed.shape[0]):

            xdim,ydim=self.imgs_processed[num_image,:,:].shape
            x=np.arange(xdim+1)
            y=np.arange(ydim+1)
            xx,yy=np.meshgrid(x[:-1]+0.5,y[:-1]+0.5)
            #xxn=xx-curv[0]*yy-curv[1]*yy**2
            xxn=xx-curve_a*yy-curve_b*yy**2 #correcting the curvature
            #yyn=yy-curv*xx

            self.imgs_processed[num_image,:,:] = np.histogramdd((xxn.flatten(),yy.flatten()),bins=[y,x],weights=self.imgs_processed[num_image,:,:].T.flatten())[0]

            #im_corr = np.histogramdd((xx.flatten(),yyn.flatten()),bins=[y,x],weights=im.T.flatten())[0]#.T
            #print(ret.shape)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")


    def plot_generation(self, use_spc=True, **kwargs):
        """
        Display diagnostic plots for the background-subtraction/filtering
        pipeline: the mean raw image with the fitted curvature/background
        overlaid, the dark-image scaling fit, and a sample of processed
        RIXS spectra.

        Only produces output when `use_spc` is False — since this pipeline
        (`_remove_bkg_and_filter`) is the alternative to single-photon
        counting. Note the default is `use_spc=True`, so calling this with
        no arguments is a silent no-op; pass `use_spc=False` explicitly to
        see the plots.

        Parameters
        ----------
        use_spc : bool, optional
            Whether single-photon-counting was used. Only `False` produces
            a plot. Default True.
        **kwargs : dict
            Same processing parameters passed to `_remove_bkg_and_filter`
            (`curve_a`, `curve_b`, `index_start_fit_bkg`), used here to
            annotate the overlay on the mean raw image consistently with
            how the data was actually processed.
        """
        if not use_spc:
            # Figure 1: Dark image profiles
            plt.figure(figsize=(10,3))
            
            # Panel 1: Dark image profiles
            plt.subplot(131)
            plt.imshow(self.raw_data.mean(axis=0),
                    vmin=self.raw_data[:,:,:].mean()-1*self.raw_data[:,:,:].std(),
                        vmax=self.raw_data[:,:,:].mean()+3*self.raw_data[:,:,:].std(),cmap = cm.lipari)
            # plt.imshow(self.imgs_processed.mean(axis=0),
            #         vmin=self.imgs_processed[:,:,:].mean()-1*self.imgs_processed[:,:,:].std(),
            #             vmax=self.imgs_processed[:,:,:].mean()+3*self.imgs_processed[:,:,:].std(),cmap = 'gray')
            mean_img = self.imgs_processed.mean(axis=0)
            # max_coos = tuple(np.unravel_index(np.nanargmax(mean_img), mean_img.shape))
            ny, nx = self.imgs_processed.shape[1], self.imgs_processed.shape[2]
            x = np.arange(nx)
            curve_a = kwargs.get('curve_a', 0)
            curve_b = kwargs.get('curve_b', 0)
            y = curve_a * (x) + curve_b * (x)**2 - curve_a * (nx) - curve_b * (nx)**2 + kwargs.get('index_start_fit_bkg', 1192) - 100
            y = np.clip(y, 0, ny - 1)
            ax = plt.gca()
            plt.plot(x, y, color='r', linestyle='--', linewidth=0.8)
            index_start = int(kwargs.get('index_start_fit_bkg', 1192))
            ax.axhline(index_start, color='darkgreen', linestyle='--', linewidth=1)
            plt.title("Mean raw image")
            plt.colorbar(label='Counts')
            plt.xlabel("X (pixel)")
            plt.ylabel("Y (pixel)")

            # Panel 2: Mean spectrum near elastic line
            plt.subplot(132)
            plt.plot(self.raw_data[0,:,:].mean(axis=1),color="k", label='Image profile')
            plt.plot(self.dark_img.mean(axis=1), color='darkgreen', label='Dark profile (filtered)')
            plt.plot(self.dark_poly[0,0]*self.dark_img.mean(axis=1) + self.dark_poly[1,0],color="r", label='Dark profile (scaled)')
            plt.xlabel("Pixel ")
            plt.ylabel("Counts")
            plt.legend()
            plt.title(f"Scaling parameters: A={self.dark_poly[0,0]:.4f}, B={self.dark_poly[1,0]:.4f}")

            # Plot up to 5 spectra (vertical profiles) from imgs_processed with different colors
            n_images_total = self.imgs_processed.shape[0] 
            n_to_plot = min(5, n_images_total)

            # pick indices evenly spaced through the available images
            if n_images_total == 1:
                indices = [0]
            else:
                indices = np.linspace(0, n_images_total - 1, n_to_plot).astype(int).tolist()

            colors = cm.managua(np.linspace(0, 1, max(5, n_to_plot)))
            plt.subplot(133)
            for i, idx in enumerate(indices):
                img = self.imgs_processed[idx]
                spectrum = img.mean(axis=1)  # vertical profile (mean over X)
                plt.plot(spectrum, color=colors[i], label=f"Image {idx}")

            plt.xlabel("Pixel ")
            plt.ylabel("Counts")
            plt.title("Selected RIXS spectra")
            plt.legend()
            plt.tight_layout()
            plt.show()


    def get_dark_img_from_nxs(self): #backround
        """
        Loads and processes dark images from NeXus files.

        This method:
        1. Loads dark images from specified run numbers (self.dark_img_run)
        2. Converts images to float type
        3. Averages multiple dark images if present
        4. Accumulates total counting time

        The processed data is stored in:
        - self.dark_img: Averaged dark image array
        - self.count_time: Total counting time for dark images

        Raises
        ------
        Exception
            If dark image files cannot be found or loaded
        """
        print(f"Retrieving dark images from run(s) {self.dark_img_run}.")
        start_time = time.perf_counter()

        for dark_img_run_loop in self.dark_img_run.tolist():

            filename = self.find(dark_img_run_loop,self.directory_dark)
            print(f"Retrieving dark image from file {filename}")

            with nxload(filename,mode='r') as f: #loading background
                try:
                    self.dark_img = np.concatenate((self.dark_img, f.entry['andor']['data'].nxvalue))
                except:
                    self.dark_img = np.concatenate((self.dark_img, f.entry1['andor']['data'].nxvalue))

                try:
                    self.dark_count_time += f.entry['instrument']['m4c1']['count_time'].nxvalue #count time
                except:
                    try:
                        self.dark_count_time += f.entry1['instrument']['m4c1']['count_time'].nxvalue
                    except:
                        self.dark_count_time += 180

        self.dark_img = self.dark_img.astype(float)
        #self.dark_img = self.dark_img.mean(axis=0)

        print(f"Total number of dark images retrieved: {self.dark_img.shape[0]}")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    def get_dawn_spectrum(self, normalized=True):
        """
        Retrieves the spectrum data processed by DAWN from a processed NeXus file.
        
        Parameters
        ----------
        normalized : bool, default=True
            If True, returns the normalized spectrum data
            If False, returns the raw spectrum data
            
        Returns
        -------
        tuple
            (energyLoss, spectrum) where:
            - energyLoss: array of energy loss values
            - spectrum: array of spectrum intensity values
            
        Notes
        -----
        The data is retrieved from different paths in the NeXus file structure:
        - Normalized data: '.../normalized_correlated_spectrum_0'
        - Raw data: '.../correlated_spectrum_0'
        """
        filename = self.find(self.runNB,self.directory, processed=True)
        with nxload(filename,mode='r') as f:
            if normalized:
                try:
                    energyLoss = f.processed.summary['1-Combined RIXS image reduction']['normalized_correlated_spectrum_0']['Energy loss'].nxvalue
                    spectrum = f.processed.summary['1-Combined RIXS image reduction']['normalized_correlated_spectrum_0'].data.nxvalue
                except:
                    energyLoss = f.processed.summary['1-RIXS image reduction']['normalized_correlated_spectrum_0']['Energy loss'].nxvalue
                    spectrum = f.processed.summary['1-RIXS image reduction']['normalized_correlated_spectrum_0'].data.nxvalue
            else:
                try:
                    energyLoss = f.processed.summary['1-Combined RIXS image reduction']['correlated_spectrum_0']['Energy loss'].nxvalue
                    spectrum = f.processed.summary['1-Combined RIXS image reduction']['correlated_spectrum_0'].data.nxvalue       
                except:
                    energyLoss = f.processed.summary['1-RIXS image reduction']['correlated_spectrum_0']['Energy loss'].nxvalue
                    spectrum = f.processed.summary['1-RIXS image reduction']['correlated_spectrum_0'].data.nxvalue
                    
        return energyLoss, spectrum