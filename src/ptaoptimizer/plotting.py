from os import path
import numpy as np
import matplotlib.pyplot as plt

def passband_shapes(telnoise, instr_name, lower_max=None, upper_min=None,
                    show_gain=True, figsize=(12, 6),
                    ylim=None, grid=False,
                    show_title=False, dashed_gain=False,
                    save=False, savedir="."):
    """
    Inputs: FrequencyOptimizer.TelescopeNoise instance, maximum of lower
    band (optional), minimum of upper band (optional)
    """
    if not path.isdir(savedir):
        raise FileNotFoundError("savedir '{}' does not exist.".format(savedir))
    trx = telnoise.T_rx
    gain = telnoise.gain
    nus = telnoise.rx_nu
    gaincol = "#1b9e77"
    if show_gain:
        trxcol = "#d95f02"
    else:
        trxcol = "black"

    fig, ax1 = plt.subplots(figsize=figsize)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    if grid:
        ax1.grid()
    # left axis
    ax1.plot(nus, trx, color=trxcol, linewidth=2)
    ax1.set_xlabel(r"$\nu$ (GHz)")
    ax1.set_ylabel(r"$T_\mathrm{rcvr}$ (K)", color=trxcol)
    ax1.tick_params(axis='y', labelcolor=trxcol)

    if show_gain:
        #right axis
        if dashed_gain:
            gain_ls = "--"
        else:
            gain_ls = "-"
        ax2 = ax1.twinx()
        ax2.plot(nus, gain, color=gaincol, linewidth=2, ls=gain_ls)
        ax2.set_ylabel("$G$ (K/Jy)", color=gaincol)
        ax2.tick_params(axis='y', labelcolor=gaincol)

    if show_title:
        ax1.set_title(instr_name)
        
    plt.tight_layout()
    if save:
        fname = "{}_passband.png".format(instr_name)
        plt.savefig(path.join(savedir, fname))
    else:
        plt.show()
