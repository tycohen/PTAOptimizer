from os import path
import numpy as np

class OptimizeFrequency(object):
    """
    Class to store arguments to frequencyoptimizer.FrequencyOptimizer

    Attributes:
    ----------

    nsteps: int
            Number of steps in the grid to run when log=True
    dnu: float
         Delta nu grid spacing when log=False
    log_grid: bool
         Use a log-space grid of center frequencies and bandwidths
    frac_bw: bool
         Run in fractional bandwidth mode
    full_bandwidth: bool
         Enforce full bandwidth in calculations
    plot: bool
          Write optimizer grid plots
    plotdir: string
             Directory in which to write plots
    levels: numpy.ndarray
            Array of contour levels for plotting
    colors: list
            List of contour colors for plotting
    lws: list
         List of contour linewidths for plotting
    ncpu: int
          Number of cpus to use for parallel computing
    """
    def __init__(self,
                 nsteps=20,
                 dnu=None,
                 log_grid=True,
                 frac_bw=False,
                 full_bandwidth=False,
                 plot=False,
                 plotdir=".",
                 levels=None,
                 colors=None,
                 lws=None,
                 ncpu=1):
        """
        ___init___ function for the OptimizeFrequency class
        """

        self.nsteps = nsteps
        if not log_grid and dnu is None:
            raise ValueError("'dnu' must be set if log_grid = False")
        self.dnu = dnu
        self.log_grid = log_grid
        self.frac_bw = frac_bw
        self.full_bandwidth = full_bandwidth
        self.ncpu = ncpu
        self.plot = plot
        if isinstance(plotdir, str):
            if path.isdir(plotdir):
                self.plotdir = plotdir
            else:
                raise OSError("Directory '{}' does not exist.".format(plotdir))
        else:
            raise TypeError("'plotdir' must be a string")
        if isinstance(levels, (np.ndarray, type(None))):
            self.levels = levels
        else:
            raise TypeError("'levels' must be None or a numpy.ndarray")            
        if isinstance(colors, (type(None), list)):
            self.colors = colors
        else:
            raise TypeError("'colors' must be None or a list")
        if isinstance(lws, (type(None), list)):
            self.lws = lws
        else:
            raise TypeError("'lws' must be None or a list")

