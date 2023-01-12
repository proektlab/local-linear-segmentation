"""Logistic regression segmentation"""
import numpy as np
import numpy.ma as ma
from sklearn.linear_model import LogisticRegression

import LogisticReg_calculations as lr_c


def test(window1, window2, theta1, theta2, n, per, use_prev_choice=True):
    """
    Test for a break between window1 and window2
    window1: (inputs, decisions) for the first window (including previous choice, if using)
    window2: (inputs, decisions) for the second window, which extends past window1
    theta1: (weights, bias) inferred logistic regression parameters from window1
    theta2: (weights, bias) inferred logistic regression parameters from window2
    n: number of null model simulations to use to infer likelihood ratio distribution
    per: percentile of likelihood ratio distribution to use as a threshold
    """
    ll2 = lr_c.log_likelihood(*window2, *theta2)
    ll1 = lr_c.log_likelihood(*window2, *theta1)
    log_ratio = ll2 - ll1

    # null model - assumes that the last coefficient is for previous choice
    null_log_ratio_dist = lr_c.r_null(window2[0], len(window1[0]), *theta1, n, use_prev_choice)
    thresh = np.nanpercentile(null_log_ratio_dist, per)
    is_outside_dist = log_ratio >= thresh
    return is_outside_dist

