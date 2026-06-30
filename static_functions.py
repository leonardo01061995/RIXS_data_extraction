import matplotlib.pyplot as plt
import os
import time
import numpy as np
import xarray as xr
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
from cmcrameri import cm
from IPython.display import display
import pandas as pd

def calculate_shift(spectra, 
                   aligning_range=None,
                    fit_shifts=False,
                    smooth_shifts=False, 
                    correlation_batch_size=1, 
                    poly_order=1):
    """
    Corrects energy shifts between spectra by aligning them to the first spectrum.

    The alignment is performed by:
    1. Computing cross-correlation between each spectrum and the first spectrum
    2. Finding the optimal shift that maximizes correlation
    3. Applying the shift correction using interpolation
    4. Storing the shift values for each spectrum

    Parameters
    ----------
    aligning_range : tuple
        A tuple specifying the pixel row range for alignment (start, stop)
    """
    print(f"Calculating energy shifts of {spectra.shape[0]} spectra. Shifting spectra by: ")
    start_time = time.perf_counter()

    if aligning_range is not None:
        pixel_row_start, pixel_row_stop = aligning_range

    # Initialize arrays
    shifts = np.zeros(spectra.shape[0])
    real_shifts = shifts.copy()
    corrected_spectra = np.zeros_like(spectra)
    
    # Pre-average the spectra to have a more reliable cross-correlation
    if correlation_batch_size > 1:
        spec_avg = average_images_in_batches(spectra, np.min((correlation_batch_size, spectra.shape[0])))
    else:
        spec_avg = spectra.copy()

    # Get reference spectrum once (avoid recomputing for each iteration)
    ref_spectrum = spec_avg[0, :]

    real_shifts = np.zeros(spec_avg.shape[0])
    # Calculate the shift for the remaining spectra
    for num_spectrum in range(1, spec_avg.shape[0]):
        # Get current spectrum
        curr_spectrum = spec_avg[num_spectrum, :]
        
        # Calculate shift
        real_shifts[num_spectrum] = correlate_spectra(
            curr_spectrum, ref_spectrum, pixel_row_start, pixel_row_stop)

    index_aux = np.arange(0, spectra.shape[0], correlation_batch_size)[:spectra.shape[0] // correlation_batch_size]
    if fit_shifts:
        coeffs = np.polyfit(index_aux, real_shifts, deg=poly_order)
        shifts = np.polyval(coeffs, range(0, spectra.shape[0]))
    elif smooth_shifts:
        # Extrapolate by specifying left and right values
        shifts = np.interp(range(0, spectra.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1])          
        sigma = correlation_batch_size   # Adjust sigma based on correlation_batch_size
        shifts = gaussian_filter1d(shifts, sigma=sigma)
    else:
        shifts = np.interp(range(0, spectra.shape[0]), index_aux, real_shifts, left=real_shifts[0], right=real_shifts[-1])

    # Round all the shifts to the nearest integer
    # shifts = np.round(shifts)

    # Save the real shifts somewhere for plotting
    for i in range(0, real_shifts.shape[0]):
        real_shifts[i * correlation_batch_size:(i + 1) * correlation_batch_size] = real_shifts[i]


    print("")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    return shifts, real_shifts


def calculate_shift_new(spectra,
                       aligning_range=None,
                       fit_shifts=False,
                       smooth_shifts=False,
                       interp_shifts=True,
                       correlation_batch_size=50,
                       poly_order=2):
    """
    Improved shift calculation using sequential (neighbor-to-neighbor)
    cross-correlation with sub-pixel parabolic refinement.

    Accepts the same parameters as ``calculate_shift`` so it can be used
    as a drop-in replacement inside ``process_images``.

    Key improvement over ``calculate_shift``: instead of correlating every
    batch against the *first* one (which breaks down when the total drift is
    large), each batch is correlated against its immediate predecessor and
    the absolute shifts are obtained by cumulative summation.  This keeps
    the relative displacement between adjacent batches small and the
    cross-correlation well-conditioned.  The actual correlation is also
    performed by ``correlate_spectra_robust``, which adds true sub-pixel 
    parabolic refinement, and zero-means the spectra before correlating.
    The second round of correlation against a preliminary mean spectrum further
    improves the accuracy of the shift estimates.

    Parameters
    ----------
    pixel_row_start : int
        Start of the spectral window used for cross-correlation
        (should bracket the elastic line).
    pixel_row_stop : int
        End of the spectral window.
    fit_shifts : bool
        If True, fit a polynomial of degree *poly_order* to the batch shifts
        and evaluate it at every image index.
    smooth_shifts : bool
        If True (and *fit_shifts* is False), linearly interpolate the batch
        shifts to per-image resolution and then Gaussian-smooth with
        sigma = *correlation_batch_size*.
    interp_shifts : bool
        If True (and both flags above are False), linearly interpolate the
        batch shifts to per-image resolution without additional smoothing.
    correlation_batch_size : int
        Number of images averaged per batch before correlating.
    poly_order : int
        Degree of the polynomial fit (only used when *fit_shifts* is True).
    """
    print(f"[calculate_shift_new] {spectra.shape[0]} images, "
            f"batch={correlation_batch_size}")
    start_time = time.perf_counter()

    if aligning_range is not None:
        pixel_row_start, pixel_row_stop = aligning_range
    n_images = spectra.shape[0]
    shifts      = np.zeros(n_images)
    real_shifts = np.zeros(n_images)

    # ── 1. Build batch-averaged spectra ───────────────────────────────────
    if correlation_batch_size > 1:
        spec_avg = average_images_in_batches(spectra, np.min((correlation_batch_size, spectra.shape[0]))).T
    else:
        spec_avg = spectra.T.copy()
    # spec_avg shape: (n_rows, n_batches)
    n_batches = spec_avg.shape[1]

    # ── 2. Sequential neighbor-to-neighbor cross-correlation ──────────────
    # real_shifts_batches[0] = 0  (reference)
    # real_shifts_batches[k] = shift of batch k relative to batch k-1
    # cumsum → absolute shift of each batch relative to batch 0
    real_shifts_batches = np.zeros(n_batches)
    for k in range(1, n_batches):
        real_shifts_batches[k] = correlate_spectra_robust(
            spec_avg[:, k], spec_avg[:, k - 1],
            pixel_row_start, pixel_row_stop)

    # Cumulative sum gives absolute drift from the first batch
    real_shifts_batches = np.cumsum(real_shifts_batches)

    real_shifts_batches_1round = np.zeros(n_images)
    for k in range(n_batches):
        s = k * correlation_batch_size
        e = min(s + correlation_batch_size, n_images)
        real_shifts_batches_1round[s:e] = real_shifts_batches[k]

    # ── 2b. Second-round correlation against a preliminary mean spectrum ───
    # Shift each batch spectrum by the first-round estimate, build a mean,
    # then re-correlate every batch against that mean for a better estimate.
    n_rows = spec_avg.shape[0]
    fine_grid_full = np.arange(n_rows, dtype=float)

    # Build preliminary shifted spectra and average them
    shifted_avg = np.zeros_like(spec_avg)
    for k in range(n_batches):
        shifted_avg[:, k] = np.interp(
            fine_grid_full - real_shifts_batches[k],
            fine_grid_full,
            spec_avg[:, k],
            left=0.0, right=0.0)
    prelim_mean = shifted_avg.mean(axis=1)   # shape: (n_rows,)

    # Re-correlate each batch against the preliminary mean
    real_shifts_batches2 = np.zeros(n_batches)
    for k in range(n_batches):
        real_shifts_batches2[k] = correlate_spectra_robust(
            spec_avg[:, k], prelim_mean,
            pixel_row_start, pixel_row_stop)

    # Use the second-round absolute shifts
    real_shifts_batches = real_shifts_batches2

    # ── 3. Map batch shifts → per-image shifts ────────────────────────────
    # Centre of each batch in image-index space (more accurate than batch start)
    index_aux = (np.arange(n_batches) + 0.5) * correlation_batch_size
    all_indices = np.arange(n_images, dtype=float)

    if fit_shifts:
        coeffs = np.polyfit(index_aux, real_shifts_batches, deg=poly_order)
        shifts = np.polyval(coeffs, all_indices)
    elif smooth_shifts:
        shifts = np.interp(all_indices, index_aux, real_shifts_batches,
                                left=real_shifts_batches[0],
                                right=real_shifts_batches[-1])
        shifts = gaussian_filter1d(shifts, sigma=correlation_batch_size)
    elif interp_shifts:
        shifts = np.interp(all_indices, index_aux, real_shifts_batches,
                                left=real_shifts_batches[0],
                                right=real_shifts_batches[-1])
    else:
        # Repeat each batch shift for every image in that batch
        rep = np.repeat(real_shifts_batches, correlation_batch_size)
        if rep.size >= n_images:
            shifts = rep[:n_images].astype(float)
        else:
            pad = np.full(n_images - rep.size, rep[-1] if rep.size > 0 else 0.0, dtype=float)
            shifts = np.concatenate((rep.astype(float), pad))

    # ── 4. Store batch-level shifts for plotting ──────────────────────────
    for k in range(n_batches):
        s = k * correlation_batch_size
        e = min(s + correlation_batch_size, n_images)
        real_shifts[s:e] = real_shifts_batches[k]

    print("")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds. \n")

    return shifts, real_shifts, real_shifts_batches_1round


def correlate_spectra(spec, spec_ref, pixel_start, pixel_stop):
    """
    this function correlates two spectra and returns the lag value to be used for shifting the images

    params:
    spec: the spectrum to be shifted
    spec_ref: the reference spectrum
    pixel_start: the starting pixel
    pixel_stop: the stopping pixel

    return:
    lag: the lag value to be used for shifting the images
    """
    # Ensure spec and spec_ref are 1D arrays
    if spec.ndim != 1 or spec_ref.ndim != 1:
        raise ValueError("Both spec and spec_ref must be 1D arrays.")

    factor = 2  # Change this value as needed

    # Ensure pixel_start and pixel_stop are within bounds
    if pixel_start < 0 or pixel_stop >= len(spec):
        raise ValueError("pixel_start and pixel_stop must be within the bounds of the spectrum.")

    if factor == 1:
        # Skip interpolation if factor is 1
        xx = np.arange(pixel_start, pixel_stop + 1)
        spec = spec[xx]  # Use the original spectrum directly
        spec_ref = spec_ref[xx]  # Use the original reference spectrum directly
    else:
        xx = np.arange(pixel_start, pixel_stop + 1 / factor, 1 / factor)
        spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
        spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

    crosscorr = correlate(spec/np.mean(spec), spec_ref/np.mean(spec_ref))
    lag_values = np.arange(-spec.shape[0] + 1, spec.shape[0], 1)
    lag = lag_values[np.argmax(crosscorr)] / factor

    # if factor > 1:
    #     # xx = np.arange(0, len(spec_ref) + 1 / factor, 1 / factor)
    #     xx = np.linspace(0, len(spec_ref)-1, (len(spec_ref) - 1) * factor + 1)
    #     spec = np.interp(xx, np.arange(0, len(spec)), spec)  # Interpolating the spectrum
    #     spec_ref = np.interp(xx, np.arange(0, len(spec_ref)), spec_ref)  # Interpolating the reference spectrum

    # lag_values, crosscorr = self._custom_cross_correlation(spec, spec_ref, int(pixel_start*factor), int(pixel_stop*factor))
    # crosscorr_restricted = crosscorr[(lag_values >= -10) & (lag_values <= 10)]
    # first_index = np.argmax(lag_values >= -10)
    # lag = (lag_values[np.argmax(crosscorr_restricted)] + first_index) / factor

    return lag


def correlate_spectra_robust(spec, spec_ref, pixel_start, pixel_stop,
                                upsample_factor=5):
    """
    Cross-correlate two spectra within a spectral window and return the
    sub-pixel lag via parabolic interpolation around the CC peak.

    Improvements over the original ``correlate_spectra``:

    * **Fixed upsampling range** - the original used
        ``pixel_stop + 1/factor`` instead of ``pixel_stop + 1``, making
        the interpolation grid almost empty for ``factor > 1``.
    * **True sub-pixel accuracy** - a three-point parabolic fit around
        the cross-correlation maximum yields a continuously-valued lag
        rather than a value quantised to ``1/factor``.
    * **Zero-mean normalisation** - subtracting the mean before
        correlating removes DC-offset bias on the CC peak location.

    Parameters
    ----------
    spec : 1-D ndarray
        Spectrum to be shifted.
    spec_ref : 1-D ndarray
        Reference spectrum.
    pixel_start : int
        Start of the window (inclusive).
    pixel_stop : int
        End of the window (inclusive).
    upsample_factor : int
        Upsampling factor for the spectral window before correlating.
        Higher values give finer sub-pixel resolution.  Default 10.

    Returns
    -------
    lag : float
        Shift of *spec* relative to *spec_ref* in original pixel units
        (positive = spec is shifted to higher pixel indices).
    """
    if spec.ndim != 1 or spec_ref.ndim != 1:
        raise ValueError("Both spec and spec_ref must be 1-D arrays.")
    if pixel_start < 0 or pixel_stop >= len(spec):
        raise ValueError("pixel_start / pixel_stop out of bounds.")

    # ── Upsample the spectral window ──────────────────────────────────────
    orig_grid = np.arange(len(spec))
    fine_grid = np.arange(pixel_start, pixel_stop + 1, 1.0 / upsample_factor)

    s     = np.interp(fine_grid, orig_grid, spec)
    s_ref = np.interp(fine_grid, orig_grid, spec_ref)

    # ── Zero-mean (removes DC bias) ───────────────────────────────────────
    s     = s     - s.mean()
    s_ref = s_ref - s_ref.mean()

    # ── Cross-correlate ───────────────────────────────────────────────────
    cc = correlate(s, s_ref, mode='full')
    lags = np.arange(-(len(s) - 1), len(s))   # integer lags in fine-grid units

    peak_idx = int(np.argmax(cc))

    # ── Parabolic sub-pixel refinement (5-point fit) ──────────────────────
    if 2 <= peak_idx <= len(cc) - 3:
        # Fit a parabola through the five samples around the peak
        idx_window = np.arange(peak_idx - 2, peak_idx + 3)
        x_window = lags[idx_window].astype(float)
        y_window = cc[idx_window]
        coeffs = np.polyfit(x_window, y_window, 2)
        # Vertex of parabola: x_vertex = -b / (2a)
        a, b, _ = coeffs
        fine_lag = -b / (2.0 * a) if abs(a) > 1e-12 else float(lags[peak_idx])
    else:
        fine_lag = float(lags[peak_idx])

    # Convert from fine-grid units back to original pixel units
    return fine_lag / upsample_factor


def average_images_in_batches(spectra, batch_size):
    """
    Averages images in batches from a stack of 1D spectra.

    Parameters
    ----------
    imgs : 3D array[float]
        Array containing the images stacked along the first axis
    max_lc_images : int
        Maximum number of low-count images to consider for averaging

    Returns
    -------
    averaged_imgs : 3D array[float]
        Array containing the 1D RIXS spectra stacked along the second axis
    """
    n_images = spectra.shape[0]
    
    if batch_size == 0:
        raise ValueError("Batch size must be greater than zero. Consider increasing max_lc_images.")
    
    n_batches = n_images // batch_size
    averaged_spectra = np.zeros((n_batches, spectra.shape[1]))

    for i in range(n_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        averaged_spectra[i,:] = np.mean(spectra[start_index:end_index, :], axis=0)       

    return averaged_spectra


def custom_cross_correlation(g, f, window_start, window_end):
    """
    Cross-correlate f and g, evaluating only within [window_start:window_end] of f.
    The shifted g always uses its full values (not zero-padded).
    
    Returns:
        lags: array of lag values
        correlations: correlation value at each lag
    """
    
    f = np.asarray(f)
    g = np.asarray(g)

    N = len(f)
    M = len(g)

    window_indices = np.arange(window_start, window_end)
    norm_f = np.linalg.norm(f[window_indices])
    lags = np.arange(-M + 1, N)  # All possible lags
    
    correlations = []

    for lag in lags:
        # Shift g by lag (positive lag: shift right)
        g_shifted = np.zeros_like(f)
        # if lag==0:
        #     print("ciao")
        for i in range(N):
            g_idx = i - lag  # reverse of convolution
            if 0 <= g_idx < M:
                # if g_idx == 160:
                #     print('ciao')
                g_shifted[i] = g[g_idx]
        
        # Compute dot product only inside the window
        norm_g = np.linalg.norm(g_shifted[window_indices])
        if norm_g == 0:
            correlations.append(0)
            continue
        dot = np.dot(f[window_indices]/norm_f, g_shifted[window_indices]/norm_g)
        correlations.append(dot)

    return lags, np.array(correlations)


def _find_curvature(im,frangex,frangey, plotting=False, deg=2):
    """
    Analyze the curvature of the given image data within specified x and y ranges.

    Parameters
    ----------
    im : numpy.ndarray
        A 2D array representing the image data to be analyzed.
    frangex : list or tuple
        A range of x-coordinates (start, end) to consider for the analysis.
    frangey : list or tuple
        A range of y-coordinates (start, end) to consider for the analysis.
    plotting : bool, optional
        If True, generates plots for visual inspection of the reference and cross-correlation results. Default is False.
    deg : int, optional
        The degree of the polynomial to fit to the shifts. Default is 2 (quadratic fit).

    Returns
    -------
    None
        The method prints the extracted curvature coefficients and stores them in the instance variable `self.curv`.
    """
    ref=np.ones([frangey[1]-frangey[0],im[:,:].shape[0]]) * im[frangey[0]:frangey[1],frangex[0]:frangex[1]].mean(axis=1)[:,np.newaxis]
    crosscorr=fftconvolve(im[frangey[0]-100:frangey[1]+100,:],ref[::-1,:],axes=0)
    if plotting==True:
        f,ax=plt.subplots(2)
        ax[0].plot(ref[:,5])
        ax[0].plot(ref[:,10])
        for i in np.arange(frangex[0],frangex[1],10):
            ax[1].plot(crosscorr[:,i])
    shifts=np.argmax(crosscorr,axis=0)
    curv=np.polyfit(np.arange(im[:,:].shape[0]),shifts,deg=deg)
    print(f"Extracted curvature: f{curv}")

    return curv[1], curv[0]


def _find_aligning_range(avg_spectrum, x_data=None, threshold=0.05):
    """
    Find the position of the first significant peak in the image spectrum.
    Uses a moving average to smooth the data and identifies where signal rises
    above background.
    
    Parameters
    ----------
    avg_spectrum : array-like
        A 1D array representing the average RIXS spectrum.

    x_data : array-like, optional
        A 1D array corresponding to the x-axis values for the spectra.

    threshold : float, optional
        Threshold value above which signal is considered significant,
        as fraction of maximum intensity, default 0.1

    Returns
    ---------
    tuple
        A tuple containing the start and stop indices of the interval around the significant peak in the image spectrum.
    """
    
    print("Attempting to find the range around the elastic line...")

    # Find where signal rises above threshold * max intensity
    threshold_value = threshold * np.max(avg_spectrum)
    # Calculate the moving average with a window of 3
    moving_avg = np.convolve(avg_spectrum, np.ones(3)/3, mode='same')
    peak_start = np.where(moving_avg > threshold_value)[0]
    
    if len(peak_start) == 0:
        raise ValueError("No peak found above threshold")
    
    # Use the last peak position (elastic line) instead of first peak
    elastic_line = peak_start[-1]
    range_start = elastic_line - 50
    range_stop = elastic_line + 15
    
    print(f"Using range: {range_start}, {range_stop} (x_data: {x_data[range_start]:.2f}, {x_data[range_stop]:.2f})")
    
    # Return the range around the elastic line
    return range_start, range_stop