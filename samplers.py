import numpy as np

def spiky_t0_vec(nsamp, n_psrs, t_max, t_min, t_budget, delta_t):
    """
    Generate nsamp integration time allocation vectors that obey
        sum(t0) = t_budget 
    with one random pulsar close to upper bound t_max
    """
    valid_tmax_type = (list, tuple, np.ndarray)
    if not isinstance(t_max, valid_tmax_type):
        raise TypeError("t_max must be a vector of upper time bounds")
    if n_psrs < 2:
        raise ValueError("n_psrs must be >= 2")
    t_max = np.asarray(t_max)
    spike_idx = np.random.randint(0, n_psrs, size=nsamp)
    t_spike = t_max[spike_idx] - delta_t
    t_reduced = (t_budget - t_spike) / (n_psrs - 1)
    t_reduced[t_reduced <= 0.] = t_min
    t0 = np.full((nsamp, n_psrs), t_reduced[:, None])
    t0[np.arange(nsamp), spike_idx] = t_spike
    return t0

def feasible_equal_budget(tmin, tmax, budget, tol=1e-12):
    tmin = np.asarray(tmin, float)
    tmax = np.asarray(tmax, float)
    if np.any(tmax < tmin):
        return False
    smin = float(tmin.sum())
    smax = float(tmax.sum())
    if not (smin - tol <= budget <= smax + tol):
        return False
    return True

def make_feasible_start(tmin, tmax, budget):
    """
    Construct a feasible starting point without clipping-to-min artifacts:
    start at tmin, distribute slack proportionally to available headroom.
    (Not a sampler; just a deterministic feasible seed.)
    """
    tmin = np.asarray(tmin, float)
    tmax = np.asarray(tmax, float)
    B = float(budget)

    if not feasible_equal_budget(tmin, tmax, B):
        raise ValueError("Infeasible: budget not in [sum(tmin), sum(tmax)].")

    slack = B - float(tmin.sum())
    headroom = tmax - tmin
    if slack <= 0:
        return tmin.copy()
    if np.all(headroom <= 0):
        return tmin.copy()

    w = headroom / headroom.sum()
    t0 = tmin + slack * w
    t0 = np.minimum(np.maximum(t0, tmin), tmax)

    err = B - float(t0.sum())
    if abs(err) > 1e-9:
        idxs = np.where((t0 > tmin + 1e-9) & (t0 < tmax - 1e-9))[0]
        if idxs.size == 0:
            idxs = np.where((t0 < tmax - 1e-9) if err > 0 else (t0 > tmin + 1e-9))[0]
        if idxs.size == 0:
            return t0
        j = int(idxs[0])
        t0[j] += err
        t0[j] = min(max(t0[j], tmin[j]), tmax[j])
    return t0

def random_budget_direction(N, rng):
    """
    Random direction d with sum(d)=0 (stays on equality hyperplane).
    """
    d = rng.normal(size=N)
    d -= d.mean()  # ensures sum(d)=0
    norm = np.linalg.norm(d)
    if norm == 0:
        return random_budget_direction(N, rng)
    return d / norm

def step_limits(t, d, tmin, tmax):
    """
    Find alpha range s.t. t + alpha d stays within [tmin, tmax].
    """
    alpha_lo = -np.inf
    alpha_hi =  np.inf
    for ti, di, lo, hi in zip(t, d, tmin, tmax):
        if di > 0:
            alpha_lo = max(alpha_lo, (lo - ti) / di)
            alpha_hi = min(alpha_hi, (hi - ti) / di)
        elif di < 0:
            alpha_lo = max(alpha_lo, (hi - ti) / di)  # di<0
            alpha_hi = min(alpha_hi, (lo - ti) / di)
        else:
            continue
    return alpha_lo, alpha_hi

def hit_and_run_budget_sampler(n_samples, tmin, tmax, budget,
                               burnin=200, thin=1, seed=0,
                               logpdf=None):
    """
    Samples t vectors from {tmin[i] <= t0[i] <= tmax[i], sum(t0)=budget}.

    If logpdf is None: stationary distribution is uniform over the feasible polytope.
    If logpdf is provided: uses MH accept/reject to target density proportional to exp(logpdf(t))
    on the feasible polytope.
    """
    tmin = np.asarray(tmin, float)
    tmax = np.asarray(tmax, float)
    B = float(budget)
    N = tmin.size
    if tmax.size != N:
        raise ValueError("tmin and tmax must have same length.")
    if not feasible_equal_budget(tmin, tmax, B):
        raise ValueError("Infeasible: budget not in [sum(tmin), sum(tmax)].")

    rng = np.random.default_rng(seed)

    # Default: uniform target
    if logpdf is None:
        def logpdf(t):
            return 0.0

    # Feasible start
    t = make_feasible_start(tmin, tmax, B)
    logp = float(logpdf(t))
    if not np.isfinite(logp):
        raise ValueError("Initial point has non-finite logpdf; check logpdf/tmin/tmax.")

    out = []
    n_total_steps = burnin + n_samples * thin
    for step in range(n_total_steps):
        d = random_budget_direction(N, rng)          # sum(d)=0
        a_lo, a_hi = step_limits(t, d, tmin, tmax)   # feasible line segment

        if not np.isfinite(a_lo) or not np.isfinite(a_hi) or a_hi <= a_lo:
            continue

        # Proposal: uniform along feasible chord
        alpha = rng.uniform(a_lo, a_hi)
        t_prop = t + alpha * d

        # numerical cleanup (should be tiny)
        t_prop = np.minimum(np.maximum(t_prop, tmin), tmax)
        # keep sum exactly (tiny drift only)
        t_prop += (B - float(t_prop.sum())) / N

        # MH accept/reject to get desired distribution on the polytope
        logp_prop = float(logpdf(t_prop))
        if np.isfinite(logp_prop):
            if np.log(rng.uniform()) < (logp_prop - logp):
                t = t_prop
                logp = logp_prop
        # else reject

        if step >= burnin and ((step - burnin) % thin == 0):
            out.append(t.copy())

    return np.array(out)
