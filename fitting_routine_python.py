from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt


class FittingRoutine:
    def __init__(self, function, x, y, initial_guess, 
                 bounds=None,
                 yerr=None,
                 range_x = None,
                 alpha=0.05):
        """
        Initialize the fitting routine.
        
        """
        
        self.function = function
        if y.ndim not in [1, 2]:
            raise ValueError("Input y must be 1D or 2D array.")

        
        if y.ndim == 1:
            #1D fitting
            if not np.all(x[:-1] <= x[1:]):  # Check if x is not sorted
                print("Warning: Input x is not sorted. Automatically sorting x (and re-arranging y).")
                self.x = np.sort(x)
                sorted_indices = np.argsort(x)
                self.y = y[sorted_indices]
                if yerr is not None:
                    self.yerr = yerr[sorted_indices]
                else:
                    self.yerr = None
            else:
                self.x = x
                self.y = y
                self.yerr = yerr

            self.range_x = range_x
        elif y.ndim == 2:
            # 2D fitting: x is expected to be a tuple/list of two meshgrids (X, Y)
            if not (isinstance(x, (tuple, list)) and len(x) == 2):
                raise ValueError("For 2D fitting, x must be a tuple or list of two meshgrids (X, Y).")
            self.X, self.Y = x
            self.y = y
            self.yerr = yerr


        
        self.initial_guess = initial_guess
        if bounds is None:
            self.bounds = (-np.inf * np.ones_like(initial_guess), np.inf * np.ones_like(initial_guess))
        else:
            self.bounds = bounds
        self.alpha = alpha

    def fit_1d(self):
        """
        Perform the fitting routine using least squares optimization.
        
        """
        def residuals(params, x, y, yerr=1.0):
            res = (self.function(x, *params) - y)/yerr
            return res
        
        if self.range_x is None:
            # If no range_x is provided, use the entire range of x
            self.range_x = (self.x.min(), self.x.max())
            
        x_mask = (self.x >= self.range_x[0]) & (self.x <= self.range_x[1])

        x_fit = self.x[x_mask]
        y_fit = self.y[x_mask]

        idxs = np.isfinite(x_fit) & np.isfinite(y_fit)
        x_fit = x_fit[idxs]
        y_fit = y_fit[idxs]

        result = least_squares(residuals, x0=self.initial_guess, bounds=self.bounds, args=(x_fit, y_fit))

        popt = result.x
        try:
            pcov = np.linalg.inv(result.jac.T @ result.jac) * np.sum(result.fun**2) / (result.fun.size - popt.size)
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix encountered. Using pseudo-inverse instead.")
            pcov = np.linalg.pinv(result.jac.T @ result.jac) * np.sum(result.fun**2) / (result.fun.size - popt.size)

        # Translate the position of the Gaussian to the whole image
        self.fitted_curve = self.function(self.x, *popt).reshape(self.y.shape)

        # Calculate the confidence intervals
        self.confidence_intervals = self._estimate_confidence_intervals(popt, pcov, alpha=self.alpha)

        self.popt = popt
        return self.popt, self.confidence_intervals, self.x, self.fitted_curve
    
    def plot(self, title=None, xname='X', yname='Y'):
        """
        Plot the fitted curve along with the data points.
        
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.y, 'o', label='Data')
        plt.plot(self.x, self.fitted_curve, 'r-', label='Fitted Curve')
        plt.title(title)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.legend()
        plt.grid()


    @staticmethod
    def _estimate_confidence_intervals(popt, pcov, alpha=0.05):
        """
        Estimate the confidence intervals for fitted parameters.

        Parameters:
            popt (array): Optimal values for the parameters.
            pcov (2D array): The estimated covariance of popt.
            alpha (float): Significance level (default is 0.05 for 95% confidence interval).

        Returns:
            list of tuples: Confidence intervals for each parameter.
        """
        z_value = norm.ppf(1 - alpha / 2)  # Z-score for normal distribution (1.96 for 95% CI)
        intervals = []
        
        for i, p in enumerate(popt):
            sigma = np.sqrt(pcov[i, i])  # Standard deviation (uncertainty) of parameter
            delta = z_value * sigma  # Confidence interval range
            intervals.append((p - delta, p + delta))
        
        return intervals



class FittingRoutineND:
    def __init__(self, function, coords, data, initial_guess, bounds=None,
                 yerr=None, range_coords=None, alpha=0.05):
        """
        General N-dimensional fitting routine.

        function : callable
            Vectorized model. Called as function(c0, c1, ..., *params), where each
            ci is a coordinate array broadcastable to `data.shape`. Must return an
            array broadcastable to `data.shape`.
        coords : array or tuple/list of arrays
            Either
              (a) full coordinate grids, each with shape == data.shape
                  (e.g. the X, Y from np.meshgrid(...)), or
              (b) 1D axis vectors, one per dimension, which will be meshgridded
                  with indexing="ij" to match data.shape.
        data : ndarray
            N-dimensional array of values to fit.
        """
        self.function = function
        self.coords = coords if isinstance(coords, (tuple, list)) else (coords,)
        self.data = np.asarray(data)
        self.yerr = None if yerr is None else np.asarray(yerr)
        self.initial_guess = np.asarray(initial_guess, dtype=float)
        self.alpha = alpha
        self.ndim = self.data.ndim

        # Build per-point coordinate grids that match data.shape exactly.
        self.coord_grids = self._build_coordinate_grids()

        # Default ranges span each coordinate fully (i.e. select all points).
        if range_coords is None:
            self.range_coords = [(np.nanmin(c), np.nanmax(c)) for c in self.coord_grids]
        else:
            self.range_coords = range_coords

        # Match the 1D class: turn None into (-inf, +inf) so least_squares accepts it.
        if bounds is None:
            self.bounds = (-np.inf * np.ones_like(self.initial_guess),
                            np.inf * np.ones_like(self.initial_guess))
        else:
            self.bounds = bounds

    def _build_coordinate_grids(self):
        """Return a list of coordinate arrays, each with shape == data.shape."""
        coords = [np.asarray(c) for c in self.coords]

        # Case (a): already full grids matching the data.
        if all(c.shape == self.data.shape for c in coords):
            return coords

        # Case (b): 1D axis vectors -> meshgrid over the *actual* coordinates.
        if all(c.ndim == 1 for c in coords):
            if len(coords) != self.data.ndim:
                raise ValueError(
                    f"Got {len(coords)} axis vectors but data has "
                    f"{self.data.ndim} dimensions.")
            grids = np.meshgrid(*coords, indexing="ij")
            if grids[0].shape != self.data.shape:
                raise ValueError(
                    f"Meshgrid shape {grids[0].shape} does not match data shape "
                    f"{self.data.shape}. Check axis lengths / ordering.")
            return grids

        raise ValueError(
            "coords must be either full grids each matching data.shape, "
            "or 1D axis vectors (one per data dimension).")

    def fit_nd(self):
        # Flatten everything to 1D point lists.
        coords_flat = [c.ravel() for c in self.coord_grids]
        data_flat = self.data.ravel()

        if self.yerr is not None:
            yerr_flat = self.yerr.ravel()
        else:
            yerr_flat = None

        # Exclude non-finite data (NaN/inf) so incomplete grids fit only where
        # data is defined. Also drop points with non-finite coordinates.
        mask = np.isfinite(data_flat)
        for c in coords_flat:
            mask &= np.isfinite(c)

        # If yerr is given, a non-finite or non-positive weight is unusable.
        if yerr_flat is not None:
            mask &= np.isfinite(yerr_flat) & (yerr_flat > 0)

        # Optional masking to a sub-region, mirroring range_x in the 1D class.
        for c, (lo, hi) in zip(coords_flat, self.range_coords):
            mask &= (c >= lo) & (c <= hi)

        if not np.any(mask):
            raise ValueError("No valid (finite, in-range) data points to fit.")

        coords_fit = [c[mask] for c in coords_flat]
        data_fit = data_flat[mask]
        yerr_fit = 1.0 if yerr_flat is None else yerr_flat[mask]

        def residuals(params):
            model = self.function(*coords_fit, *params)
            return (np.ravel(model) - data_fit) / yerr_fit

        result = least_squares(residuals, x0=self.initial_guess, bounds=self.bounds)

        popt = result.x
        dof = max(result.fun.size - popt.size, 1)
        try:
            pcov = np.linalg.inv(result.jac.T @ result.jac) * np.sum(result.fun**2) / dof
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix encountered. Using pseudo-inverse instead.")
            pcov = np.linalg.pinv(result.jac.T @ result.jac) * np.sum(result.fun**2) / dof

        self.confidence_intervals, self.errors = self._estimate_confidence_intervals(
            popt, pcov, alpha=self.alpha)
        self.popt = popt

        # Evaluate over the FULL grid and reshape back to data.shape.
        self.fitted_data = self.function(*coords_flat, *popt).reshape(self.data.shape)

        return self.popt, self.fitted_data, self.confidence_intervals, self.errors

    @staticmethod
    def _estimate_confidence_intervals(popt, pcov, alpha=0.05):
        z_value = norm.ppf(1 - alpha / 2)
        intervals, errors = [], []
        for i, p in enumerate(popt):
            sigma = np.sqrt(pcov[i, i])
            delta = z_value * sigma
            intervals.append((p - delta, p + delta))
            errors.append(delta)
        return intervals, errors