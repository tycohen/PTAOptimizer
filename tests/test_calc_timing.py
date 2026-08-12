"""
Integration and unit tests for calc_timing module
"""
import pytest
import numpy as np
import parameterized as ptzd
from ptaoptimizer.pulsar import Pulsar
from ptaoptimizer.pta import PTA
from ptaoptimizer.calc_timing import calc_timing

def _write_rxspec(rxfile_path, nus):
    """
    Minimal receiver spec file: whitespace-separated columns.
    """
    lines = ["#Freq Trx G Eps"]
    for nu in nus:
        lines.append("{:.6f} 20.0 1.0 1.0".format(nu))
    rxfile_path.write_text("\n".join(lines))


@pytest.mark.parametrize(("tint_attr", "tint_calc_timing_arg", "tint_answer"),
                         [(None, 1800., np.full(20, 1800.)),
                          ({"testinstr": 3600.}, 1800., np.full(20, 1800.)),
                          ({"testinstr": 3600.}, None, np.full(20, 3600.))])
def test_calc_timing_per_pulsar_integration_time(tmp_path,
                                                 tint_attr,
                                                 tint_calc_timing_arg,
                                                 tint_answer):
    testpulsar = Pulsar(name="testpulsar1",
                        dec=90.,
                        ra=180.,
                        period=0.002,
                        dm=30.,
                        dtd=300.0,
                        dnud=0.003,
                        taud=0.06,
                        dist=0.7,
                        w50=150.0,
                        weff=450.,
                        uscale=10.,
                        s_1000=1.,
                        spindex=-2,
                        sig_j_single=100.,
                        t_int=tint_attr)
    testpta = PTA(psrlist=[testpulsar])
    nus = np.linspace(1., 2., 20)
    # temporary rxspecfile
    rxdir = tmp_path / "rxspecs"
    rxdir.mkdir()
    rxfile = rxdir / "testinstr.txt"
    _write_rxspec(rxfile, nus)
    calc_timing(testpta,
                nus,
                rxspecfile=str(rxfile),
                t_int=tint_calc_timing_arg,
                dec_lim=(90., 0.),
                lat=45.,
                gainmodel=None,
                gainexp=None)
    np.testing.assert_equal(testpta.psrlist[0].telescope_noise["testinstr"].T,
                            tint_answer)
