from __future__ import division
from pymatch_ds import *
import sys
import numpy as np

def import_sas(df):
    """
    Parameters
    ----------
    df: data frame to be decoded

    Returns
    -------
    None. decoded inline

    """
    column_types = df.dtypes
    
    #bytes, to be converted
    byte_columns = column_types[column_types == 'object']
    
    #converting and encoding necessary columns
    for col in byte_columns.index:
        df[col] = df[col].str.decode('utf-8')

def progress(i, n, prestr = ''):
    sys.stdout.write('\r{}: {}\{}'.format(prestr, i, n))
    
def drop_static_cols(df, yvar, cols = None):
    if not cols:
        cols = list(df.columns)
    # will be static for both groups
    cols.pop(cols.index(yvar))
    for col in df[cols]:
        n_unique = len(np.unique(df[col]))
        if n_unique == 1:
            df.drop(col, axis = 1, inplace = True)
            sys.stdout.write('\rStatic column dropped: {}'.format(col))
    return df

def std_diff(a, b):
    sd = np.std(a.append(b))
    med = (np.median(a) - np.median(b)) * 1.0 / sd
    mean = (np.mean(a) - np.mean(b)) * 1.0 / sd
    return med, mean

def grouped_permutation_test(f, t, c, n_samples = 1000):
    truth = f(t, c)
    comb = np.concatenate((t, c))
    times_geq = 0
    samp_arr = []
    for i in range(n_samples):
        tn = len(t)
        combs = comb[:]
        np.random.shuffle(combs)
        tt = combs[:tn]
        cc = combs[tn:]
        sample_truth = f(np.array(tt), np.array(cc))
        if sample_truth >= truth:
            times_geq += 1
        samp_arr.append(sample_truth)
    return (times_geq * 1.0) / n_samples, truth

def _chi2_distance(tb, cb):
    dist = 0
    for b in set(np.union1d(list(tb.keys()), list(cb.keys()))):
        if b not in tb:
            tb[b] = 0
        if b not in cb:
            cb[b] = 0
        xi, yi = tb[b], cb[b]
        dist += ((xi - yi) ** 2) * 1.0 / (xi + yi)
    return dist * 1.0 / 2

def which_bin_hist(t, c):
    comb = np.concatenate((t, c))
    bins = np.arange(np.percentile(comb, 99), step = 10)
    t_binned = np.digitize(t, bins)
    c_binned = np.digitize(c, bins)
    return t_binned, c_binned, bins

def bin_hist(t, c, bins):
    tc, cc = Counter(t), Counter(c)

    def idx_to_value(d, bins):
        result = {}
        for k, v, in d.items():
            result[int(bins[k-1])] = v
        return result

    return idx_to_value(tc, bins), idx_to_value(cc, bins)

def chi2_distance(t, c):
    tb, cb, bins = which_bin_hist(t, c)
    tb, cb = bin_hist(tb, cb, bins)
    return _chi2_distance(tb,cb)

def ks_boot(tr, co, nboots = 1000):
    nx = len(tr)
    w = np.concatenate((tr, co))
    obs = len(w)
    cutp = nx
    bbcount = 0
    ss = []
    fs_ks, _ = stats.ks_2samp(tr, co)
    for bb in range(nboots):
        sw = np.random.choice(w, obs, replace = True)
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)
        ss.append(s_ks)
        if s_ks >= fs_ks:
            bbcount += 1
    ks_boot_pval = bbcount * 1.0 / nboots
    return ks_boot_pval
