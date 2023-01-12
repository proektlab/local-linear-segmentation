import numpy as np
cimport numpy as np
import numpy.ma as ma
from sklearn.linear_model import LogisticRegression

cdef double sigmoid(double x):
    """(a.k.a. expit)"""
    return 1 / (1 + np.exp(-x))

cpdef log_likelihood(double[:, :] inputs not None, double[:] y not None, double[:] coefs not None, double intercept,
                     bool use_prev_choices=True):
    """
    Calculate the log likelihood of the inputs/outputs given logisitc regression model with coefs and intercept
    """
    cdef Py_ssize_t n_time = inputs.shape[0]
    cdef Py_ssize_t n_vars = inputs.shape[1]
    if use_prev_choices:
        n_vars += 1

    assert n_time == y.shape[0], 'Number of samples does not match between inputs and y'
    assert n_vars == coefs.shape[0], 'Number of coefs does not match input shape'

    cdef double accum_ll = 0
    cdef double temp
    cdef double prev_y
    cdef Py_ssize_t t
    for t in range(n_time):
        if use_prev_choices:
            if t == 0:
                prev_y = 0
            else:
                prev_y = y[t-1]
            temp = np.dot(inputs[t, :], coefs[:-1]) + coefs[-1] * prev_y + intercept
        else:
            temp = np.dot(inputs[t, :], coefs) + intercept
        temp = sigmoid(temp)
        temp = y[t] * np.log(temp) + (1-y[t]) * np.log(1-temp)
        accum_ll += temp

    return accum_ll

cdef gen_obs(double[:, :] inputs not None, double[:] coefs not None, double intercept,
             double[:] y=None, double prev_choice_coef=0.):
    """
    Simulates a decision process based on the given logistic regression model
    """
    cdef Py_ssize_t n_time = inputs.shape[0]
    cdef Py_ssize_t n_vars = inputs.shape[1]
    assert n_vars == coefs.shape[0], 'Number of coefs does not match input shape'

    sim = np.zeros(n_time, dtype=np.double)
    randvals = np.random.rand(n_time)
    y = sim
    cdef double[:] rand_view = randvals

    cdef double prev_choice = 0.
    cdef double curr_sum
    cdef double prob
    cdef Py_ssize_t t

    for t in range(n_time):
        curr_sum = intercept + prev_choice_coef * prev_choice + np.dot(inputs[t, :], coefs)
        prob = sigmoid(curr_sum)
        y[t] = 1 if rand_view[t] < prob else -1
        prev_choice = y[t]

cpdef r_null(double[:, :] inputs not None, int window1_len, double[:] coefs not None, double intercept, int n,
             bool use_prev_choices=True, double prev_choice_coef=0.):
    """
    Returns the distribution of r (likelihood ratio) for null logistic regression models based on generated outputs,
    without a change in model parameters.
    Inputs should not include "previous choice."
    """
    cdef Py_ssize_t n_vars = inputs.shape[1]
    if use_prev_choices:
        n_vars += 1

    rs = np.zeros(n, dtype=np.double)
    cdef double[:] rs_view = rs
    win1_coefs = np.zeros(n_vars, dtype=np.double)
    win2_coefs = np.zeros(n_vars, dtype=np.double)
    cdef double[:] win1_coefs_view = win1_coefs
    cdef double[:] win2_coefs_view = win2_coefs
    cdef double win1_intercept
    cdef double win2_intercept
    cdef double[:] y_view

    cdef Py_ssize_t i
    for i in range(n):
        # Simulate a time series of the size of window2 using the null model
        gen_obs(inputs, coefs, intercept, prev_choice_coef, y_view)

        # fit models to windows of these observations of size window1 and window2
        win1_intercept = fit_model(inputs[:window1_len], y_view[:window1_len], use_prev_choices, win1_coefs_view)
        win2_intercept = fit_model(inputs, y_view, use_prev_choices, win2_coefs_view)

        # obtain the likelihood ratio
        rs_view[i] = (log_likelihood(inputs, y_view, win2_coefs_view, win2_intercept, use_prev_choices) -
                      log_likelihood(inputs, y_view, win1_coefs_view, win1_intercept, use_prev_choices))

    return rs

cpdef fit_model(double[:, :] inputs not None, double[:] y not None, bool use_prev_choices=True,
                double[:] coefs_all=None):
    """
    Fits a logistic regression model. The inputs should not include 'previous choice.'
    Returns the intercept and sets the out-parameter coefs_all to the coefficients.
    """
    if use_prev_choices:
        inputs_np = np.asarray(inputs)
        prev_choices = np.insert(np.asarray(y[:-1]), 0, 0.)
        inputs_aug = np.column_stack((inputs_np, prev_choices))
    else:
        inputs_aug = inputs

    model = LogisticRegression(penalty='none')
    model.fit(inputs_aug, y)
    coefs_all[:] = np.squeeze(model.coef_)
    cdef double intercept = model.intercept_[0]
    return intercept
