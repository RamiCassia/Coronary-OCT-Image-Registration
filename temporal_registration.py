
from os import remove
from scipy import signal
from matplotlib import pylab as plt
import numpy as np
import dtw
from tqdm import tqdm


#### ==== functions ==== ####
   
# Transformation, pre-processing and parameter selection functions
def moving_average(x, h):
    n = len(x)
    hhalf = int(np.floor((h - 1)/2))
    y = [None] * n

    start_ind = list(range(0, hhalf))
    end_ind = list(range(n-hhalf,n))
    middle_ind = list(range(hhalf, n - hhalf))
    
    for i in start_ind:
        y[i] = np.mean(x[0:(2*i + 1)])
    
    for i in end_ind:
        y[i] = np.mean(x[(n + 1 - 2*(n - i) ):(n+1)])

    for i in middle_ind:
        y[i] = np.mean(x[(i-hhalf):(i+hhalf+1)])

    return(y)
    

def moving_median(x, h):
    n = len(x)
    hhalf = int(np.floor((h - 1)/2))
    y = [None] * n

    start_ind = list(range(0, hhalf))
    end_ind = list(range(n-hhalf,n))
    middle_ind = list(range(hhalf, n - hhalf))
    
    for i in start_ind:
        y[i] = np.median(x[0:(2*i + 1)])
    
    for i in end_ind:
        y[i] = np.median(x[(n + 1 - 2*(n - i) ):(n+1)])

    for i in middle_ind:
        y[i] = np.median(x[(i-hhalf):(i+hhalf+1)])

    return(y)    


def discretize(x, c):
    return([int(t >= c) for t in x])


def remove_small_branches(s, min_branch_size = 1):
    # s is a binary sequence. Output only includes those 1, which lie in a neighborhood of at least min_branch_size 1s.
    # note that with the default value min_branch_size = 1, the output is simply s itself.
    s_filtered = [None] * len(s)
    for i in range(len(s)):
        min_cond = 0 
        all_one = [False] * min_branch_size
        for j in range(min_branch_size):
            start_ind = max(0, i + j - (min_branch_size-1))
            end_ind = min(len(s) - 1, i + j)
            all_one[j] = np.prod(s[start_ind:(end_ind + 1)]) != 0 # check whether all 1
            
        is_branch = np.sum(all_one) > 0
        s_filtered[i] = int(is_branch)
    
    return(s_filtered)


def minimal_overlap_edge_branches(t1, t2, cutoff = 0.5, min_branch_size = 1):
    n1 = len(t1)
    n2 = len(t2)

    t1_branch = discretize(t1, cutoff)
    t2_branch = discretize(t2, cutoff)

    t1_big_branch = remove_small_branches(t1_branch, min_branch_size=min_branch_size)
    t2_big_branch = remove_small_branches(t2_branch, min_branch_size=min_branch_size)

    t1_branch_ind = [i for i, x in enumerate(t1_big_branch) if x == 1]
    t2_branch_ind = [i for i, x in enumerate(t2_big_branch) if x == 1]

    if len(t1_branch_ind) == 0 or len(t2_branch_ind) == 0: # At least one time series has no branches big enough
        raise ValueError('One of the inputs does not show any branches of sufficient size.')
    
    else:
        t1_start_first_branch = min(t1_branch_ind)
        t1_end_last_branch = max(t1_branch_ind)
        t2_start_first_branch = min(t2_branch_ind)
        t2_end_last_branch = max(t2_branch_ind)
        max_branch_dist = max(t1_start_first_branch, t2_start_first_branch, n1 - t1_end_last_branch, n2 - t2_end_last_branch)
        minimal_overlap = max_branch_dist + min_branch_size

    return(minimal_overlap)
   

# The core function applying dynamic time warping as implemented in dtw
def shifted_window_dtw(t1, t2, minimal_overlap, step_pattern):
    n1 = len(t1)
    n2 = len(t2)

    # Next we will try a whole bunch of different shifts and window sizes and see what works best
    t2_shifts = list(range(-(n2 - minimal_overlap), (n1 - minimal_overlap) + 1))
    window_pars = [None] * len(t2_shifts)
    # which parts of t1 and t2 are in the overlapping area for each shift?
    for i, s in enumerate(t2_shifts):
        t1_start = max(0, s)
        t1_end = min(n1 - 1, s + n2 - 1)
        t2_start = max(0, -s)
        t2_end = min(n2 - 1, n1 - s - 1)
        window_pars[i] = (s, t1_start, t1_end, t2_start, t2_end)

    
    fit_values = [None] * len(window_pars)
    j = 0

    for k in tqdm(range(len(window_pars))):
        try:
            _, t1_start, t1_end, t2_start, t2_end = window_pars[k]
            t1_temp = t1[t1_start:(t1_end+1)]
            t2_temp = t2[t2_start:(t2_end+1)]
            alignment = dtw.dtw(t1_temp, t2_temp, step_pattern = step_pattern)
            fit_values[k] = alignment.normalizedDistance
        except:
            j = j+1  
            
    # this normalisation by length shouldn't be necessary since dtw gives normalizedDistance anyway
    # but somehow it works better in the sense that (unsurprisingly) it also chooses matches longer than
    # minimal_overlap.
    # for i in range(len(window_pars)):
    #    fit_values[i] = fit_values[i]/window_pars[i][0]

    # determine the best possible shift/overlap-size combination
    print(j, " out of ", len(window_pars), " alignments were unsuccessful.")

    target_window_ind = np.argmin(fit_values)
    return(window_pars[target_window_ind])


# A wrapper function combining dynamic time warping with different pre-processing options and hyperparameter input
def temporal_registration(t1, t2, minimal_overlap = "auto", step_pattern = "symmetricP2", smoother = None, smoothing_window = None, min_branch = 1, discr_cutoff = 0.5, discretize_input = False, drop_small_branches = False):
    '''
    This is a wrapper function. It combines the functionality of shifted_window_dtw with the different options of transforming the data and selecting parameters. 
    INPUT:
        t1, t2...Time series of interest. Should be numeric lists with values in >= 0 and <= 1.
        minimal_overlap...The minimal amount of overlap deemed informative. 
            Reason for minimal_overlap-specification: otherwise trivial solution for overlaps of size 1.
            Possible values: 
                A single integer smaller than min(len(t1), len(t2)) giving the minimal overlap.
                "quarter_min_n"...in this case the minimal_overlap is chosen as min(len(t1), len(t2))/4.
                "auto"............discretizes a copy of the inputs in t1 and t2 through setting them 1 iff >= discr_cutoff.
                    This then gives predicted branches. Predicted branches of size below min_branch are removed, i.e. corresponding values set 0.
                    Next, the distance of the outermost of the remaining predicted branches within any of the two time series is determined.
                    The maximal distance + min_branch is chosen as the minimal_overlap, in order to ensure, that at least one predicted branch per time series is included in the fit.
        step_pattern...allowed step patterns for dynamic time warping, see documentation of dtw packages.
        smoother...the kind of smoothing/pre-processing that should be applied to t1 and t2 before applying dtw (but after the minimal overlap is calculated if minimal_overlap = "auto").
            Possible vlaues:
                "moving_average"...calculates the moving average of t1 and t2 with window size smoothing_window
                "moving_median"....calculates the moving median of t1 and t2 with window size smoothing_window
        smoothing_window...single integer. Argument to smoother function.
        discretize_input...boolean giving whether to discretize t1 and t2 for usage in dtw, by setting them 1 iff >= discr_cutoff.
        drop_small_branches...boolean giving whether to drop branches smaller than min_branch in discretized t1 and t2.
            If discretize_input = False but drop_small_branches = True, the result will be the undiscretized t1 and t2, where the list elements are unaffected if remove_small_branches() would 
            have classed this element as a branch, and the element set to 0 otherwise.     
                

    # OUTPUT:
        A list of three lists: [optimal_pars, t1_trafo, t2_trafo, t1_indices, t2_indices].
        optimal_pars...a list of 5 elements [shift, t1_start, t1_end, t2_start, t2_end] 
            giving how much to shift t2 relative to t1 and which portions where matched through dtw, i.e. t1[t1_start:(t1_end + 1)] and correspondingly for t2.
        t1_trafo...the transformed values of t1, based on arguments supplied to smoother (and then smoothing_window), discretize_input (and then discr_cutoff, drop_small_branches (and then min_branch))
        t2_trafo...see t1_trafo.
        t1_indices and t2_indices... those give two integer lists, specifying which index of t1 matches which index of t2
    '''
    n1 = len(t1)
    n2 = len(t2)

    if minimal_overlap == "quarter_min_n":
        minimal_overlap = int(np.floor(min(n1, n2)/4))
    elif minimal_overlap == "auto":
        minimal_overlap = minimal_overlap_edge_branches(t1, t2, cutoff = discr_cutoff, min_branch_size = min_branch)

    if not smoother is None and not smoothing_window is None:
        if smoother == "moving_average":
            t1 = moving_average(t1, h = smoothing_window)
            t2 = moving_average(t2, h = smoothing_window)
        elif smoother == "moving_median":
            t1 = moving_median(t1, h = smoothing_window)
            t2 = moving_median(t2, h = smoothing_window)
        else:
            raise ValueError('Unknown smoothing procedure requested.')

    if discretize_input:
        t1 = discretize(t1, c = discr_cutoff)
        t2 = discretize(t2, c = discr_cutoff)
        
    if drop_small_branches:
        is_branch_t1 = remove_small_branches(t1, min_branch_size=min_branch)
        is_branch_t2 = remove_small_branches(t2, min_branch_size=min_branch)
        t1 = [t1[i] * is_branch_t1[i] for i in range(len(t1))]
        t2 = [t2[i] * is_branch_t2[i] for i in range(len(t2))]


    res = shifted_window_dtw(t1 = t1, t2 = t2, minimal_overlap = minimal_overlap, step_pattern = step_pattern)
    
    _, t1_start, t1_end, t2_start, t2_end = res
    t1_overlap = t1[t1_start:(t1_end + 1)]
    t2_overlap = t2[t2_start:(t2_end+1)]
    alignment = dtw.dtw(t1_overlap, t2_overlap, step_pattern = step_pattern)

    t1_indices = list(alignment.index1 + t1_start)
    t2_indices = list(alignment.index2 + t2_start)


    return([res, t1, t2, t1_indices, t2_indices])


# Create noisy data from signal
def add_beta_noise(s1, s2, c = 0.05):
    # now we create the messy data - "t1 = noisy s1" and "t2 = noisy s2".
    # c is uncertainty parameter in (0, 0.5). 
    # Mean of signal will be c if signal == 0, and 1 - c if signal == 1.
    # Can use this to simulate differing neural network performance. 
    # data will be created from beta-distribution with adequately chosen parameters

    n1 = len(s1)
    n2 = len(s2)
    
    t1 = [None] * n1
    t2 = [None] * n2

    for i in range(n1):
        a1 = s1[i] * (1 - 2*c) + c # maps 0 to c and 1 to 1-c
        b1 = 1 - a1 # chosen so that mean is a1
        t1[i] = np.random.beta(a = a1, b = b1)

    for i in range(n2):
        a2 = s2[i] * (1 - 2*c) + c 
        b2 = 1 - a2
        t2[i] = np.random.beta(a = a2, b = b2)

    return([t1, t2])




