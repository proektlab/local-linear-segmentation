import numpy as np
cimport numpy as np
import numpy.ma as ma
from sklearn.linear_model import LogisticRegression

cdef double sigmoid(double x):
    """(a.k.a. expit)"""
    return 1 / (1 + np.exp(-x))

cpdef log_likelihood(double[:, :] inputs not None, double[:] y not None, double[:] coefs not None, double intercept):
    """
    Calculate the log likelihood of the inputs/outputs given logisitc regression model with coefs and intercept
    """
    cdef Py_ssize_t n_time = inputs.shape[0]
    cdef Py_ssize_t n_vars = inputs.shape[1]

    assert n_time == y.shape[0], 'Number of samples does not match between inputs and y'
    assert n_vars == coefs.shape[0], 'Number of coefs does not match input shape'

    cdef double accum_ll = 0
    cdef double temp
    cdef double prev_y
    cdef Py_ssize_t t
    for t in range(n_time):
        temp = np.dot(inputs[t, :], coefs) + intercept
        temp = sigmoid(temp)
        temp = y[t] * np.log(temp) + (1-y[t]) * np.log(1-temp)
        accum_ll += temp

    return accum_ll

cdef gen_obs(double[:, :] inputs not None, double[:] coefs not None, double intercept,
             double[:] y=None, bool use_prev_choice=True):
    """
    Simulates a decision process based on the given logistic regression model
    use_prev_choice specifies whether the previous choice is included in inputs. If so,
    it must be treated separately in the simulation.
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
        if use_prev_choice:
            curr_sum = intercept + coefs[-1] * prev_choice + np.dot(inputs[t, :-1], coefs[:-1])
        else:
            curr_sum = intercept + np.dot(inputs[t, :], coefs)
        prob = sigmoid(curr_sum)
        y[t] = 1 if rand_view[t] < prob else -1
        prev_choice = y[t]

cpdef r_null(double[:, :] inputs not None, int window1_len, double[:] coefs not None, double intercept, int n,
             bool use_prev_choice=True):
    """
    Returns the distribution of r (likelihood ratio) for null logistic regression models based on generated outputs,
    without a change in model parameters.
    """
    cdef Py_ssize_t n_vars = inputs.shape[1]

    rs = np.zeros(n, dtype=np.double)
    cdef double[:] rs_view = rs
    win1_coefs = np.zeros(n_vars, dtype=np.double)
    win2_coefs = np.zeros(n_vars, dtype=np.double)
    cdef double[:] win1_coefs_view = win1_coefs
    cdef double[:] win2_coefs_view = win2_coefs
    cdef double win1_intercept
    cdef double win2_intercept
    cdef double[:] y_view

    sim_inputs = np.copy(inputs)
    cdef double[:, :] sim_inputs_view = sim_inputs

    cdef Py_ssize_t i
    for i in range(n):
        # Simulate a time series of the size of window2 using the null model
        gen_obs(inputs, coefs, intercept, y_view, use_prev_choice)
        if use_prev_choice:
            # replace last column with choices from this simulation
            sim_inputs_view[1:, -1:] = y_view[:-1]

        # fit models to windows of these observations of size window1 and window2
        win1_intercept = fit_model(sim_inputs_view[:window1_len], y_view[:window1_len], win1_coefs_view)
        win2_intercept = fit_model(sim_inputs_view, y_view, win2_coefs_view)

        # obtain the likelihood ratio
        rs_view[i] = (log_likelihood(sim_inputs_view, y_view, win2_coefs_view, win2_intercept) -
                      log_likelihood(sim_inputs_view, y_view, win1_coefs_view, win1_intercept))

    return rs

cpdef fit_model(double[:, :] inputs not None, double[:] y not None, double[:] coefs=None):
    """
    Fits a logistic regression model.
    Returns the intercept and sets the out-parameter coefs_all to the coefficients.
    """
    model = LogisticRegression(penalty='none')
    model.fit(inputs, y)
    coefs[:] = np.squeeze(model.coef_)
    cdef double intercept = model.intercept_[0]
    return intercept
