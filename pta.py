from os import path
import string

FOOTSYMB_FMT = r"$^{{\rm {}}}$"

class PTA(object):
    """
    Class to store all timed pulsars

    Attributes:
    __________
    name : string (optional)
           PTA name
    psrlist : list
              list of pulsar.Pulsar objects
        """
    def __init__(self,
                 name=None,
                 psrlist=[],
                 *args,
                 **kwargs):
        """
        ___init___ function for the Pulsar class
        """

        self.name = name
        self.psrlist = psrlist

    def get_single_pulsar(self, psr_name):
        """return pulsar.Pulsar object whose name matches 'psr_name'
        """
        for p in self.psrlist:
            if p.name == psr_name:
                return p
        else:
            raise ValueError("No pulsar named {} in PTA".format(psr_name))

    def sigma_best(self, exclude=[], strip_str=None):
        """
        Get the best instrument for each pulsar
        and return list of tuples of (pulsar name, instrument, sigma_tot)
        
        Parameters
        ----------
        exclude : list
                  list of telescope name substrings to exclude
        strip_str : string
                  remove this string from instrument name results
        """
        if strip_str is None:
            strip_str = ""
        if not isinstance(exclude, list):
            raise TypeError("'exclude' must be a list of substrings not "
                            "{}".format(type(exclude)))
        best_instr_list = []
        for p in self.psrlist:
            best_tup = sorted([(p.name, k.replace(strip_str, ''), v['sigma_tot'])
                               for k, v in iter(p.sigmas.items())
                               if not any([e in k for e in exclude])],
                              key=lambda t: (t[2] < 0., t[2]))[0]
            best_instr_list.append(best_tup)
        return best_instr_list

    def sigma_2best(self, exclude=[], strip_str=None):
        """
        Get the best and 2nd instrument for each pulsar
        and return list of tuples of (pulsar name, instrument, % diff)
        Set exclude = list of substrings to ignore a particular telescope
        
        Parameters
        ----------
        exclude : list
                  list of telescope name substrings to exclude
        strip_str : string
                  remove this string from instrument name results
        """
        if strip_str is None:
            strip_str = ""
        if not isinstance(exclude, list):
            raise TypeError("'exclude' must be a list of substrings not "
                            "{}".format(type(exclude)))
        best_instr_list = []
        for p in self.psrlist:
            sigmas_sorted = sorted([(p.name, k.replace(strip_str, ''),
                                     v['sigma_tot'])
                                    for k, v in iter(p.sigmas.items())
                                    if not any([e in k for e in exclude])],
                                   key=lambda t: (t[2] < 0., t[2]))
            best = sigmas_sorted[0]
            secbest = sigmas_sorted[1]
            
            best_instr_list.append((best[0], best[1], best[2],
                                    secbest[1], secbest[2]))
        return best_instr_list

    def write_to_txt(self, filename):
        """Write total RMS for each pulsar at each instrument to file"""
        key_names = sorted([k for k in self.psrlist[0].get_instr_keys()])
        instr_names = sorted([k.replace("_logain", "") for k in key_names])
        lines = []
        header = "\t".join(["# name"] + instr_names)
        lines.append(header)
        for p in self.psrlist:
             s = "\t".join([p.name] + [str(p.sigmas[k]['sigma_tot']) for k in key_names])
             lines.append(s)
        all_lines = "\n".join(lines)
        with open(filename, 'w') as f:
            f.write(all_lines)
        return

    def write_2best_to_markdown(self, filename, exclude=[]):
        valid_extensions = (".md", ".markdown")
        if not filename.endswith(valid_extensions):
            raise ValueError("'filename' must end with {}".format(valid_extensions))
        header = """
|  Pulsar |   Best Telescope(s)     |Total RMS (&mu;s)|   2nd Best Telescope(s)     |Total RMS (&mu;s)|
|---------|-------------------------|-----------------|-------------------------|-----------------|
"""
        rows = []
        for t in sorted(self.sigma_2best(exclude=exclude),
                        key=lambda x: x[0]):
            rows.append("| {} | {} | {:.4f} | {} | {:.4f} |".format(*t))
        with open(filename, "w") as f:
            f.write(header + "\n".join(rows))
        return

    def make_deluxetable(self, exclude_names=[], save=True, savedir=".",
                         split_row_idx=None,
                         longtable=False,
                         fontsize=r"scriptsize",
                         comments_macro="table caption",
                         footnote_dict=None):
        """
        Make a publication-quality AASTex deluxetable

        Parameters
        ----------
        exclude_names : list
            names of pulsars to exclude from the table
        save : bool
            save to file with name self.name + 'psr_params.tex'
        savedir : str
            path to save directory
        split_row_idx : None
            optional list of indices to the left of which to split the table
        vertically, created len(split_row_idx) separate tables
        longtable : bool
            adds a \startlongtable before table environment
        fontsize : raw str
            latex named font size (no backslash)
        comments_macro : str
            optional custom latex macro for inserting content into \tablecomments
        """
        if not path.isdir(savedir):
            raise FileNotFoundError("'savedir' {} does not exist.".format(savedir))
        if split_row_idx is not None and len(split_row_idx) == 0:
            raise ValueError("split_row_idx must be a list with "
                             "at least one value.")
        psrs_incl = sorted([p for p in self.psrlist if p.name not in exclude_names],
                           key=lambda p: float(p.name[1:5]))
        npsr = len(psrs_incl)
        name_column = {"attr": "name", "label": "Pulsar", "unit": "", "fmt": "{}"}
        cols = [
            name_column,
            # {"attr": "ra", "label": "R.A.", "unit": r"($^\circ$)",
            #  "fmt": "{:.4f}"}
            # {"attr": "dec", "label": "Dec.", "unit": r"($^\circ$)",
            #  "fmt": "{:.4f}"}
            {"attr": "period", "label": r"$P$", "unit": "(ms)",
             "fmt":  lambda x: "{:.2f}".format(x * 1000.)},
            {"attr": "dm", "label": r"DM", "unit": r"($\mathrm{pc\,cm^{-3}}$)",
             "fmt": "{:.2f}"},
            {"attr": "taud", "label": r"$\tau_d$", "unit": r"($\mathrm{\mu s}$)",
             "fmt": lambda x: sci_latex(x, ndp=1)},
            {"attr": "dtd", "label": r"$\Delta \tau_d$", "unit": r"(s)",
             "fmt": "{:.1f}"},
            {"attr": "dnud", "label": r"$\Delta \nu_d$", "unit": r"(GHz)",
             "fmt": lambda x: sci_latex(x, ndp=1)},
            {"attr": "dist", "label": r"$d$", "unit": r"(kpc)",
             "fmt": "{:.2f}"},
            {"attr": "s_1000", "label": r"$S_{1000}$", "unit": "(mJy)",
             "fmt": "{:.2f}"},
            {"attr": "spindex", "label": r"$\alpha$", "unit": "",
             "fmt": "{:.2f}"},
            {"attr": "w50", "label": r"$W_{50}$", "unit": "($\mathrm{\mu s}$)",
             "fmt": "{:.2f}"},
            {"attr": "weff", "label": r"$W_\mathrm{eff}$", "unit": "($\mathrm{\mu s}$)",
             "fmt": "{:.2f}"},
            {"attr": "uscale", "label": r"$U_\mathrm{scale}$", "unit": "",
             "fmt": "{:.2f}"},
            {"attr": "sig_j_single", "label": r"$\sigma_\mathrm{J,1}$",
             "unit": "($\mathrm{\mu s}$)",
             "fmt": "{:.2f}"},
            {"attr": "redamp", "label": r"$A_\mathrm{red}$",
             "unit": "", "fmt": lambda x: sci_latex(x, ndp=2)},
            {"attr": "redgamma", "label": r"$\gamma_\mathrm{red}$",
             "unit": "", "fmt": "{:.2f}"}
            ]
        col_fmt = "l" + "c" * (len(cols) - 1)
        tablecaption = (r"\tablecaption{{Adopted Pulsar Parameters"
                        "\label{{tab:{0}_pulsar_params}}}}".format(self.name))
        tabhead_lines = [
            r"\begin{{deluxetable}}{{{}}}".format(col_fmt),
            r"\tabletypesize{{\{}}}".format(fontsize),
            tablecaption,
            r"\tablehead{",
            r" & ".join(["\colhead{{{}}}".format(c["label"]) for c in cols]) + "\\\\",
            r" & ".join(["\colhead{{{}}}".format(c["unit"]) for c in cols]) + "\\\\",
            r"}",
            r"\startdata"
        ]
        footbase = r"\tablenotetext{{{}}}{{\vspace{{-2ex}}{}}}"
        if footnote_dict is None:
            footnote_dict = {}
            footlst = []; sortfoot = []
        else:
            footnote_dict = footnote_order(footnote_dict, psrs_incl, cols)
            footitems = [k2 for k1 in footnote_dict.keys()
                         for k2 in footnote_dict[k1].values()]
            sortfoot = sorted(footitems,
                              key=lambda d: len(d["footsymb"]) * 25 +\
                              string.ascii_lowercase.index(d["footsymb"]))
            footlst = [footbase.format(d["footsymb"], d["text"])
                       for d in sortfoot]
        if longtable:
            tabhead_lines.insert(0, r"\startlongtable")
        data = [" & ".join([apply_foot(apply_fmt(c["fmt"], getattr(p, c["attr"])),
                                       footnote_dict, p.name, c["attr"])
                            for c in cols]) \
                + ("\\\\" if i < npsr - 1 else "")
                for i, p in enumerate(psrs_incl)]
        tablecomments = r"\tablecomments{{{}}}".format(comments_macro)
        tabfoot_lines = [
            r"\enddata",
            tablecomments,
            r"\end{deluxetable}"
            ]
        if split_row_idx is not None: # create len(split_row_idx) separate tables
            data_split = [data[a:b] for a, b in zip([0] + split_row_idx,
                                                    split_row_idx + [None])]
            footlst_split = [[footbase.format(d["footsymb"], d["text"])
                              for d in sortfoot
                              if any([FOOTSYMB_FMT.format(d["footsymb"]) in l
                                      for l in s])]
                             for s in data_split]
            print(footlst_split)
            new_caption = r"\tablecaption{continued}"
            split_tabs = []
            for i, (d, ftl) in enumerate(zip(data_split, footlst_split)):
                if ftl:
                    tabfoot_lines.insert(-1, "\n".join(ftl))
                split_tabs.append("\n".join(["\n".join(tabhead_lines),
                                             "\n".join(d),
                                             "\n".join(tabfoot_lines)]))
                table = "\n".join(split_tabs)
                if i == 0: # 'Continued' caption after first sub-table
                    tabhead_lines.insert(0, r"\addtocounter{table}{-1}")
                    tabhead_lines = [new_caption if l == tablecaption else l
                                     for l in tabhead_lines]
                    tabfoot_lines.remove(tablecomments)
        else:
            if footnote_dict is not None:
                tabfoot_lines.insert(-1, "\n".join(footlst))
            table = "\n".join(["\n".join(tabhead_lines),
                               "\n".join(data),
                               "\n".join(tabfoot_lines)])
        if save:
            fname = "{}_psr_params.tex".format(self.name)
            with open(path.join(savedir, fname), "w") as out:
                out.write(table)
        return table

def sci_latex(x, ndp=1):
    mant, exp = f"{x:.{ndp}e}".split("e")
    if exp in ("+00", "+01"):
        return f"{x:.{ndp}f}"
    else:
        return rf"${mant} \times 10^{{{int(exp)}}}$"

def apply_fmt(fmt, value):
    if callable(fmt):
        return fmt(value)
    return fmt.format(value)
    
def footnote_order(footnote_dict, psrs_incl, cols):
    """
    Parameters:
    ----------
    footnote_dict : pulsar dict of attribute dict containing footnote text 
                    for each attr that requires a footnote
    psrs_incl : list of pulsar.Pulsar objects  to include in the table
    cols : list of table column dictionaries

    Returns:
    -------
    footnote_dict with 'footsymb' key,value added for each attr key
    """
    nfoot = len([k2 for k1 in footnote_dict.keys()
                 for k2 in footnote_dict[k1].keys()])
    letters = string.ascii_lowercase
    footsymb = [letters[i % 25] * (i // 25 + 1) for i in range(nfoot)]
    i = 0
    for p in psrs_incl:
        for c in cols:
            try:
                d = footnote_dict[p.name][c["attr"]]
            except KeyError:
                continue
            d['footsymb'] = footsymb[i]
            i += 1
    return footnote_dict

def apply_foot(valstr, footnote_dict, n, attr):
    try:
        symb = footnote_dict[n][attr]['footsymb']
    except KeyError:
        return valstr
    return valstr + FOOTSYMB_FMT.format(symb)
