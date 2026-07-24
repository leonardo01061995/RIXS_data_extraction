import matplotlib.pyplot as plt
import os
import time
import numpy as np
import xarray as xr
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, correlate, correlation_lags
from scipy.optimize import curve_fit
from cmcrameri import cm
from IPython.display import display
import pandas as pd



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
    print(f"\t[calculate_shift_new] on {spectra.shape[0]} images, "
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
    real_shifts_batches = real_shifts_batches2 - real_shifts_batches2[0]  # shift relative to first batch

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
    print(f"\tElapsed time: {elapsed_time:.4f} seconds.")

    return shifts, real_shifts, real_shifts_batches_1round


def calculate_shift_mccc(spectra,
                       aligning_range=None,
                       shift_postprocess='none',
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
    
    print(f"\t[calculate_shift_new] on {spectra.shape[0]} images, "
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

    # ── 2. First round: MCCC ──────────────

    real_shifts_batches = multi_channel_cross_correlation(spec_avg.T, pixel_row_start, pixel_row_stop,
                                    upsample_factor=5, max_shift=40)

    real_shifts_batches -= real_shifts_batches[0]  # shift relative to first batch
    real_shifts_batches_1round = np.zeros(n_images)
    for k in range(n_batches):
        s = k * correlation_batch_size
        e = min(s + correlation_batch_size, n_images)
        real_shifts_batches_1round[s:e] = real_shifts_batches[k]


    # ── 2b. Second-round correlation against a preliminary mean spectrum ───
    # # Shift each batch spectrum by the first-round estimate, build a mean,
    # # then re-correlate every batch against that mean for a better estimate.
    n_rows = spec_avg.shape[0]
    fine_grid_full = np.arange(n_rows, dtype=float)

    #interpolate the spectra on the new pixel grid and build preliminary avg
    fine_grid_full = np.arange(spec_avg.shape[0], dtype=float)
    shifted_avg = np.zeros_like(spec_avg)
    for k in range(n_batches):
        shifted_avg[:, k] = np.interp(
            fine_grid_full,
            fine_grid_full - real_shifts_batches[k],
            spec_avg[:, k],
            left=0.0, right=0.0)
    prelim_mean = shifted_avg.mean(axis=1)   # shape: (n_rows,)

    # Re-correlate each batch against the preliminary mean
    real_shifts_batches2 = np.zeros(n_batches)
    for k in range(n_batches):
        real_shifts_batches2[k] = correlate_spectra_robust(
            shifted_avg[:, k], prelim_mean,
            pixel_row_start, pixel_row_stop)

    # Use the second-round absolute shifts
    real_shifts_batches = real_shifts_batches2 - real_shifts_batches2[0]+ real_shifts_batches  # shift relative to first batch

    # real_shifts_batches2 = multi_channel_cross_correlation(shifted_avg.T, pixel_row_start, pixel_row_stop,
    #                             upsample_factor=5, max_shift=40)
    # real_shifts_batches = real_shifts_batches2 - real_shifts_batches2[0] + real_shifts_batches   # shift relative to first batch

    # ── 3. Map batch shifts → per-image shifts ────────────────────────────
    # Centre of each batch in image-index space (more accurate than batch start)
    index_aux = (np.arange(n_batches) + 0.5) * correlation_batch_size
    # index_aux = np.arange(n_batches) * correlation_batch_size
    all_indices = np.arange(n_images, dtype=float)

    # ── Detect jumps > 2 pixels between consecutive batch shifts ──────────────
    diffs = np.abs(np.diff(real_shifts_batches))
    jump_indices = np.where(diffs > 2)[0] + 1  # batch indices where a new group starts

    split_points = np.concatenate(([0], jump_indices, [n_batches]))
    real_shifts_batches_groups = [
        real_shifts_batches[split_points[g]:split_points[g + 1]]
        for g in range(len(split_points) - 1)
    ]
    index_aux_groups = [
        index_aux[split_points[g]:split_points[g + 1]]
        for g in range(len(split_points) - 1)
    ]

    if len(real_shifts_batches_groups) > 1:
        print(f"\tDetected {len(real_shifts_batches_groups)} groups of batches with large shifts between them.")
        for g, grp in enumerate(real_shifts_batches_groups):
            print(f"\t\tGroup {g + 1}: {len(grp)} batches, from batch {split_points[g]} to {split_points[g + 1] - 1}")

    # Image-index boundaries for each group
    image_split_points = np.concatenate((
        [0],
        [int(split_points[g] * correlation_batch_size) for g in range(1, len(split_points) - 1)],
        [n_images]
    ))

    shifts = np.zeros(n_images)
    for g in range(len(real_shifts_batches_groups)):
        grp_shifts    = real_shifts_batches_groups[g]
        grp_idx_aux   = index_aux_groups[g]
        img_start     = int(image_split_points[g])
        img_end       = int(image_split_points[g + 1])
        grp_indices   = np.arange(img_start, img_end, dtype=float)

        if len(grp_shifts) == 0:
            continue
        elif len(grp_shifts) == 1:
            shifts[img_start:img_end] = grp_shifts[0]
        elif shift_postprocess == 'fit':
            deg = min(poly_order, len(grp_shifts) - 1)
            coeffs = np.polyfit(grp_idx_aux, grp_shifts, deg=deg)
            shifts[img_start:img_end] = np.polyval(coeffs, grp_indices)
        elif shift_postprocess == 'smooth':
            shifts[img_start:img_end] = np.interp(
                grp_indices, grp_idx_aux, grp_shifts,
                left=grp_shifts[0], right=grp_shifts[-1])
            shifts[img_start:img_end] = gaussian_filter1d(
                shifts[img_start:img_end], sigma=correlation_batch_size)
        elif shift_postprocess == 'interp':
            shifts[img_start:img_end] = np.interp(
                grp_indices, grp_idx_aux, grp_shifts,
                left=grp_shifts[0], right=grp_shifts[-1])
        else:
            rep = np.repeat(grp_shifts, correlation_batch_size)
            grp_len = img_end - img_start
            if rep.size >= grp_len:
                shifts[img_start:img_end] = rep[:grp_len].astype(float)
            else:
                pad = np.full(grp_len - rep.size,
                              rep[-1] if rep.size > 0 else 0.0, dtype=float)
                shifts[img_start:img_end] = np.concatenate((rep.astype(float), pad))

    # Store per-image batch-level shifts for plotting (section 4)
    for k in range(n_batches):
        s = k * correlation_batch_size
        e = min(s + correlation_batch_size, n_images)
        real_shifts[s:e] = real_shifts_batches[k]

    print("")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\tElapsed time: {elapsed_time:.4f} seconds.")

    return shifts, real_shifts, real_shifts_batches_1round



def multi_channel_cross_correlation(spectra, pixel_start, pixel_stop,
                                    upsample_factor=5, max_shift=40):
    """
    Align N one-dimensional spectra by multi-channel cross-correlation
    (VanDecar & Crosson, BSSA 80, 150 (1990)).
 
    All N(N-1)/2 pairwise lags are measured, then combined by unweighted
    least squares on the complete graph, which reduces to the row mean of
    the antisymmetric lag matrix.
 
    Parameters
    ----------
    spectra : (N, M) array_like
        N spectra sampled on a common pixel grid of length M.
    pixel_start, pixel_stop : int
        Window (in ORIGINAL pixel units) used for the correlation. Choose a
        region dominated by a sharp feature (e.g. the elastic line).
    upsample_factor : int
        Linear-interpolation upsampling of the window. Lag resolution
        becomes 1 / upsample_factor pixels before sub-sample refinement.
    max_shift : float
        Maximum |lag| considered, in ORIGINAL pixel units.
 
    Returns
    -------
    shifts : (N,) ndarray
        Estimated shift of each spectrum, in ORIGINAL pixel units, gauge-fixed
        to zero mean. spectra[i] is displaced by +shifts[i] relative to the
        ensemble; correct it by shifting spectrum i by -shifts[i].
    D : (N, N) ndarray
        Antisymmetric matrix of measured pairwise lags, D[i, j] ~ t_i - t_j.
    quality : (N, N) ndarray
        Normalised cross-correlation peak height for each pair, in [-1, 1].
        Symmetric. Useful as a weight or an outlier flag.
    """
    S = np.asarray(spectra, dtype=float)
    if S.ndim != 2:
        raise ValueError("spectra must be a 2-D array of shape (N, M)")
    N, M = S.shape
 
    if not (0 <= pixel_start < pixel_stop <= M):
        raise ValueError("require 0 <= pixel_start < pixel_stop <= M")
 
    U = int(upsample_factor)
    if U < 1:
        raise ValueError("upsample_factor must be >= 1")
 
    # ---- window, then upsample by linear interpolation --------------------
    x = np.arange(pixel_start, pixel_stop, dtype=float)          # original grid
    n_up = (pixel_stop - pixel_start - 1) * U + 1
    x_up = pixel_start + np.arange(n_up) / U                     # spacing 1/U px
 
    W = np.empty((N, n_up))
    for i in range(N):
        W[i] = np.interp(x_up, x, S[i, pixel_start:pixel_stop])
 
    # ---- remove the DC level ---------------------------------------------
    # A constant background dominates the raw cross-correlation and biases the
    # peak towards zero lag. Subtracting the window mean is not optional.
    W -= W.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(W, axis=1)
    norms[norms == 0] = 1.0
 
    # ---- lag grid, restricted to |lag| <= max_shift -----------------------
    lags = correlation_lags(n_up, n_up, mode="full")             # upsampled units
    max_lag_up = int(round(max_shift * U))
    keep = np.abs(lags) <= max_lag_up
    if not keep.any():
        raise ValueError("max_shift too small for the chosen window")
    lags_keep = lags[keep].astype(float)
 
    D = np.zeros((N, N))
    quality = np.eye(N)
 
    for i in range(N):
        for j in range(i + 1, N):
            cc = correlate(W[i], W[j], mode="full") / (norms[i] * norms[j])
            cc = cc[keep]
            k = int(np.argmax(cc))
 
            # parabolic sub-sample refinement on the upsampled grid
            if 0 < k < len(cc) - 1:
                y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
                denom = y0 - 2.0 * y1 + y2
                dk = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
                dk = np.clip(dk, -0.5, 0.5)
            else:
                dk = 0.0   # peak on the edge of the search range: suspicious
 
            lag_up = lags_keep[k] + dk
            D[i, j] = lag_up / U          # back to original pixel units
            D[j, i] = -D[i, j]
            quality[i, j] = quality[j, i] = cc[k]
 
    # ---- unweighted least squares on the complete graph -------------------
    # equivalent to solving  L t = b  with  sum_i t_i = 0
    shifts = D.mean(axis=1)
 
    return shifts
 
 
def cycle_residuals(D):
    """
    Consistency check: D[i,j] + D[j,k] + D[k,i] should vanish for a perfectly
    self-consistent set of lags. Large values flag a bad spectrum or a
    cross-correlation peak picked on the wrong maximum.
    """
    N = D.shape[0]
    R = D[:, :, None] + D[None, :, :] + D.T[:, None, :]   # R[i,j,k]
    iu = np.array([(i, j, k)
                   for i in range(N) for j in range(i + 1, N)
                   for k in range(j + 1, N)])
    if len(iu) == 0:
        return np.array([])
    return R[iu[:, 0], iu[:, 1], iu[:, 2]]
    


def correlate_spectra_robust(spec, spec_ref, pixel_start, pixel_stop,
                             upsample_factor=5, max_shift=40):
    """
    Drop-in replacement for the original correlate_spectra_robust.

    Sign convention is unchanged: positive return = `spec` is shifted to
    HIGHER pixel indices relative to `spec_ref`.

    Two fixes vs. the original:

    1. NORMALISED cross-correlation.  The original used
       scipy.signal.correlate(s, s_ref) on two segments cut to the SAME
       window.  That raw correlation is NOT normalised for the number of
       overlapping samples, so it carries a triangular "overlap envelope"
       that multiplies the CC and pulls the arg-max toward lag 0.  For a
       sharp, dominant elastic line the bias is negligible, but for a broad
       / low-contrast line it grows with the lag and systematically
       UNDER-estimates large shifts (and enlarging the window does not
       help). Here we use a Pearson-style normalised correlation instead.

    2. The MOVING spectrum is never truncated.  Only the reference is cut to
       [pixel_start, pixel_stop]; `spec` is re-sampled with a sliding offset
       over its FULL values, so its elastic peak cannot fall off the window
       edge at large shifts.  (This is what your own custom_cross_correlation
       already does correctly.)

    The parabolic sub-pixel refinement now only accepts a genuine maximum
    (a < 0); otherwise it keeps the integer-lag peak.
    """
    if spec.ndim != 1 or spec_ref.ndim != 1:
        raise ValueError("Both spec and spec_ref must be 1-D arrays.")
    if pixel_start < 0 or pixel_stop >= len(spec):
        raise ValueError("pixel_start / pixel_stop out of bounds.")

    orig_grid = np.arange(len(spec))
    up = upsample_factor

    # Reference: window + zero-mean (fixed grid)
    fine_ref = np.arange(pixel_start, pixel_stop + 1, 1.0 / up)
    r = np.interp(fine_ref, orig_grid, spec_ref)
    r = r - r.mean()
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return 0.0

    # Slide the moving spectrum over a limited, physical lag range
    nlag = int(round(max_shift * up))
    lags = np.arange(-nlag, nlag + 1)
    cc = np.empty(lags.size, dtype=float)
    for i, L in enumerate(lags):
        seg = np.interp(fine_ref + L / up, orig_grid, spec,
                        left=spec[0], right=spec[-1])
        seg = seg - seg.mean()
        sn = np.linalg.norm(seg)
        cc[i] = 0.0 if sn == 0 else float(np.dot(r, seg) / (r_norm * sn))

    pk = int(np.argmax(cc))

    # Parabolic sub-pixel refinement (only if the vertex is a maximum)
    if 1 <= pk <= cc.size - 2:
        x = lags[pk - 1:pk + 2].astype(float)
        y = cc[pk - 1:pk + 2]
        a, b, _ = np.polyfit(x, y, 2)
        fine_lag = -b / (2.0 * a) if a < -1e-12 else float(lags[pk])
    else:
        fine_lag = float(lags[pk])

    return fine_lag / up




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


def _find_aligning_range(avg_spectrum, x_data=None, threshold=0.05, extended_range=False):
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
    
    print("\tAttempting to find the range around the elastic line...")
    if extended_range:
        multiplication_factor = 2
    else:
        multiplication_factor = 1

    # Find where signal rises above threshold * max intensity
    threshold_value = threshold * np.max(avg_spectrum)
    # Calculate the moving average with a window of 3
    moving_avg = np.convolve(avg_spectrum, np.ones(3)/3, mode='same')
    peak_start = np.where(moving_avg > threshold_value)[0]
    
    if len(peak_start) == 0:
        raise ValueError("No peak found above threshold")
    
    # Use the peak position
    elastic_line = peak_start[0]
    range_start = max(elastic_line - 15*multiplication_factor, 0)
    range_stop = min(elastic_line + 50*multiplication_factor, avg_spectrum.size-1)

    print(f"\tUsing range: {range_start}, {range_stop} (x_data: {x_data[range_start]:.2f}, {x_data[range_stop]:.2f})")
    
    # Return the range around the elastic line
    return range_start, range_stop



def check_variations_parameters(spectra_xarray, attributes_to_exclude, threshold=0.1):
    """
    Check for variations in parameters across different spectra.
    Parameters
    ----------
    spectra_xarray : xarray.Dataset
        The xarray dataset containing the spectra and their attributes.
    attributes_to_exclude : list
        List of attribute names to exclude from the comparison.
    threshold : float, optional
        Threshold for numeric comparison. If the difference between numeric attributes exceeds this value, it will be flagged. Default is 0.1.
    """

    parameter_changes_values = []
    parameter_changes_strings = []
    change_flag_value = False
    change_flag_string = False

    for ii, spec_name in enumerate(spectra_xarray.data_vars):
        if ii==0:
            other_attrs = {key: value for key, value in spectra_xarray[spec_name].attrs.items() 
                if key not in attributes_to_exclude}
            
        for key, value in other_attrs.items():
            if key in spectra_xarray[spec_name].attrs:
                current_value = spectra_xarray[spec_name].attrs[key]
                if isinstance(current_value, (int, float)) and isinstance(value, (int, float)):
                    # Perform numeric comparison
                    if abs(current_value - value) > threshold:
                        change_flag_value = True
                        parameter_changes_values.append(key)
                else:
                    # Perform string comparison
                    if str(current_value) != str(value):
                        change_flag_string = True
                        parameter_changes_strings.append(key)
    parameter_changes_values = list(set(parameter_changes_values))
    parameter_changes_strings = list(set(parameter_changes_strings))

    print("")
    if change_flag_value:
        print(f"******Warning******: Attributes {parameter_changes_values} differs by more than {threshold} between spectra.")
        for key in parameter_changes_values:
            print(f"<{key}> values:  ", end="")
            for spec_name in spectra_xarray.data_vars:
                print(f"{spectra_xarray[spec_name].attrs[key]}", end=", ")
            print("")  # Newline after each parameter

    if change_flag_string:
        print(f"******Warning******: Attributes {parameter_changes_strings} differs between spectra.")
        for key in parameter_changes_strings:
            print(f"<{key}> values: ", end="")
            for spec_name in spectra_xarray.data_vars:
                print(f"{spectra_xarray[spec_name].attrs[key]}", end=", ")
            print("")  # Newline after each parameter
    print("")


def _determine_polarization(hu70ap, hu70cp):
    
    # Determine polarization based on motor positions
    if hu70cp > 30 and hu70ap > 30:
        polarization = 'LV'
    elif -2 < hu70cp < 2 and -2 < hu70ap < 2:
        polarization = 'LH'
    elif 2 <= hu70cp <= 30 and 2 <= hu70ap <= 30:
        polarization = 'C+'
    elif -30 <= hu70cp <= -2 and -30 <= hu70ap <= -2:
        polarization = 'C-'
    else:
        polarization = 'Unknown'
    
    return polarization