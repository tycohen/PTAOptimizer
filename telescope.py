class Telescope(object):
    """
    Class to describe the observing telescope

    Attributes
    ----------
    name : string
           name of the telescope
    dec_lim : tup
              Declination limits of telescope (max 90, min -90)
    alt_lim : tup
              Elevation pointing limits of telescope (max 90, min 0)
    lat : float
          Observatory latitude in degrees (max 90, min -90)
    gainmodel : string
                Model to compute elevation dependent gain
                Valid inputs ('cos', 'exp')
    gainexp : float or numpy.ndarray
              Exponent for cosine ('cos') gain model
    """

    def __init__(self,
                 name=None,
                 dec_lim=None,
                 alt_lim=None,
                 lat=None,
                 gainmodel=None,
                 gainexp=None,
                 timefac=None):

        self.name = name
        dec_lim_formatstr = ("Limits must be between (inclusive) -90 and 90 deg. "
                             "Format is (max lim,min lim)")
        if not isinstance(dec_lim, tuple):
            raise TypeError("'dec_lim' must be defined and of type tuple. " +
                            dec_lim_formatstr)
        elif len(dec_lim) != 2:
            raise ValueError("'dec_lim' must be tuple of length 2" +
                            dec_lim_formatstr)
        elif dec_lim[0] > 90. or dec_lim[1] < -90. or dec_lim[1] >= dec_lim[0]:
            raise ValueError(("({},{}) are not valid declination limits. " +
                              dec_lim_formatstr
                             ).format(dec_lim[0], dec_lim[1]))
        else:
            self.dec_lim = dec_lim            
        if lat > 90. or lat < -90.:
            raise ValueError("Observatory latitude must be be between -90 and 90 deg")
        else:
            self.lat = lat
        if self.lat > 0.:
            self.alt_lim = (90., 90. - self.lat + self.dec_lim[1])
        else:
            self.alt_lim = (90., 90. + self.lat - self.dec_lim[0])
        if gainmodel not in (None, 'cos', 'exp'):
            raise ValueError("{} is not a valid gain model".format(gainmodel))
        else:
            self.gainmodel = gainmodel
        self.gainexp = gainexp
        self.timefac = timefac
