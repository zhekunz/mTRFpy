import sys
import random
import math
import warnings
from itertools import product
from typing import List, Union
from collections.abc import Iterable

import array_api_compat

from mtrf.matrices import (
    covariance_matrices,
    banded_regularization,
    _check_data,
    _check_length,
    _get_xy,
    Array,
    ArrayList,
    lags_idx,
    is_scalar,
    array_device,
    creation_kwargs,
    asarray_like,
    weights_to_trf,
    fit_weights_with_covariance_matrices,
)


def neg_mse(y: Array, y_pred: Array) -> Array:
    """
    Compute negative mean suqare error (mse) between predicted
    and observed data

    Parameters
    ----------
    y: Array
        samples-by-features matrix of observed data.
    y_pred: Array
        samples-by-features matrix of predicted data.

    Returns
    -------
    neg_mse: Array
        Negative mse (-mse) for each feature in y.
    """
    xp = array_api_compat.array_namespace(y)
    mse = xp.mean((y - y_pred) ** 2, axis=0)
    return -mse


def pearsonr(y: Array, y_pred: Array) -> Array:
    """
    Compute Pearson's correlation coefficient between predicted
    and observed data

    Parameters
    ----------
    y: np.ndarray
        samples-by-features matrix of observed data.
    y_pred: np.ndarray
        samples-by-features matrix of predicted data.

    Returns
    -------
    r: np.ndarray
        Pearsons r for each feature in y.
    """
    xp = array_api_compat.array_namespace(y)
    numerator = xp.mean(
        (y - xp.mean(y, axis=0)) * (y_pred - xp.mean(y_pred, axis=0)), axis=0
    )
    denominator = xp.std(y, axis=0, correction=0) * xp.std(
        y_pred, axis=0, correction=0
    )
    r = numerator / denominator
    return r


def crossval(
    model,
    stimulus: ArrayList,
    response: ArrayList,
    fs: int,
    tmin: float,
    tmax: float,
    regularization: Union[int, float],
    k: int = -1,
    seed: Union[int, None] = None,
    average: Union[bool, List[int]] = True,
    verbose: bool = True,
) -> Union[float, Array]:
    """
    Test model metric using k-fold cross-validation.

    Input data is randomly shuffled and separated into k parts of with approximately
    the same number of trials. The first k-1 parts are used for training and the kth
    part for testing the model.

    Parameters
    ----------
    model: TRF
        Base model used for cross-validation.
    stimulus: list of array-like
        Each element must contain one trial's stimulus in a two-dimensional
        samples-by-features array (second dimension can be omitted if there is
        only a single feature.
    respnse: list of array-like
        Each element must contain one trial's response in a two-dimensional
        samples-by-channels array.
    fs: int
        Sample rate of stimulus and response in hertz.
    tmin: float
        Minimum time lag in seconds.
    tmax: float
        Maximum time lag in seconds.
    regularization: float or int
        Value for the lambda parameter regularizing the regression.
    k: int
        Number of data splits, if -1, do leave-one-out cross-validation.
    seed: int
        Seed for the random number generator.
    average: bool or list or numpy.ndarray
        If True (default), average correlation and mean squared error across all
        predictions (e.g. channels in the case of forward modelling). If `average`
        is an array of integers only average the predicted features at those indices.
        If `False`, return each predicted feature's metric.

    Returns
    -------
    metric: float or array-like
        Metric as computed by the metric function in the attribute `model.metric`.
    """
    stimulus, xps = _check_data(stimulus)
    response, xpr = _check_data(response)
    stimulus, response, n_trials = _check_length(stimulus, response)
    assert n_trials >= k, f"Not enough trials for {k}-fold cross-validation"
    if not xps == xpr:
        raise TypeError("stimulus and response trials must be of the same type!")
    elif array_device(stimulus[0]) != array_device(response[0]):
        raise TypeError("stimulus and response trials must be on the same device!")
    else:
        xp = xps
    if isinstance(regularization, Iterable):
        raise ValueError(
            "Crossval only accepts a single scalar for regularization! "
            "For cross-validation with multiple regularization values use `nested_crossval`!"
        )
    if len(stimulus) < 2:
        raise ValueError("Cross-validation requires at least two trials!")
    trf = model.copy()
    # if seed is not None:
    #     random.seed(seed)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = lags_idx(xp, tmin, tmax, fs)
    if trf.preload:
        cov_xx, cov_xy = covariance_matrices(
            x, y, lags, model.zeropad, preload=True
        )
    else:
        cov_xx, cov_xy = None, None
    metric = _crossval(
        model,
        x,
        y,
        cov_xx,
        cov_xy,
        lags,
        fs,
        regularization,
        k,
        xp,
        average,
        verbose,
        seed=seed,
    )
    return metric


def nested_crossval(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    bands=None,
    k=-1,
    average=True,
    seed=None,
    verbose=True,
):
    """
    Unbiased estimate of model accuracy when fitting the regularization parameter.

    This fuction divides the data into k parts and runs two nested
    cross-validation loops: the outer loop selects k-1 parts to optimize the
    regularization value and the kth part to test the final model's accuracy.
    The inner loop uses cross-validation to determine the best regularization
    value as in the `fit` method. The data are rotated so that each of the
    k segments is used to test the final model's accuracy once. The average
    correlation and mean squared error across all folds is an unbiased estimate
    of the model's accuracy because the test data was not part of the optimization
    process.

    Parameters
    ----------
    stimulus: list
        Each element must contain one trial's stimulus in a two-dimensional
        samples-by-features array (second dimension can be omitted if there is
        only a single feature.
    response: list
        Each element must contain one trial's response in a two-dimensional
        samples-by-channels array.
    fs: int
        Sample rate of stimulus and response in hertz.
    tmin: float
        Minimum time lag in seconds.
    tmax: float
        Maximum time lag in seconds.
    regularization: list or float or int
        Values for the regularization parameter lambda. The model is fitted
        separately for each value and the one yielding the highest accuracy
        is chosen (correlation and mean squared error of each model are returned).
    bands: list or None
        Must only be provided when using banded ridge regression. Size of the
        features for which a regularization parameter is fitted, in the order they
        appear in the stimulus matrix. For example, when the stimulus consists of
        an envelope vector and a 16-band spectrogram, bands would be [1, 16].
    k: int
        Number of data splits for cross validation, defaults to 5.
        If -1, do leave-one-out cross-validation.
    average: bool or list or numpy.ndarray
        If True (default), average correlation and mean squared error across all
        predictions (e.g. channels in the case of forward modelling). If `average`
        is an array of integers only average the predicted features at those indices.
    seed: int
        Seed for the random number generator.
    verbose: bool
        If True (default), show a progress bar during fitting.

    Returns
    -------
    metric_test: numpy.ndarray
        Metric as computed by the metric function defined in the attribute
        `TRF.metric` for all k test sets.
    best_regularization: numpy.ndarray
        Optimal regularization values for all k training sets.
    """
    stimulus, xps = _check_data(stimulus)
    response, xpr = _check_data(response)
    stimulus, response, n_trials = _check_length(stimulus, response)
    assert (
        n_trials >= k + 1 and n_trials >= 3
    ), f"Not enough trials for {k}-fold nested cross-validation!"
    k = _check_k(k, n_trials)
    if not xps == xpr:
        raise TypeError("stimulus and response trials must be of the same type!")
    elif array_device(stimulus[0]) != array_device(response[0]):
        raise TypeError("stimulus and response trials must be on the same device!")
    else:
        xp = xps
    if average is False and not is_scalar(regularization):
        raise ValueError("Average must be True or a list of indices!")
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    lags = lags_idx(xp, tmin, tmax, fs)
    if model.method == "banded":
        regularization = asarray_like(xp, regularization, x[0])
        coefficients = list(product(regularization, repeat=len(bands)))
        regularization = [
            banded_regularization(len(lags), c, bands, xp) for c in coefficients
        ]

    if model.preload:
        cov_xx, cov_xy = covariance_matrices(x, y, lags, model.zeropad)
    else:
        cov_xx, cov_xy = None, None

    splits = _trial_splits(n_trials, k)
    n_splits = len(splits)
    metric_test = xp.zeros(n_splits, **creation_kwargs(x[0]))
    best_regularization = []
    for split_i in range(n_splits):
        idx_test = splits[split_i]
        idx_train_val = sum(splits[:split_i] + splits[split_i + 1 :], [])
        if not is_scalar(regularization):
            metric = xp.zeros(len(regularization), **creation_kwargs(x[0]))
            for ir in _progressbar(
                range(len(regularization)),
                "Hyperparameter optimization",
                verbose=verbose,
            ):
                if cov_xx is not None:
                    cov_idx = asarray_like(
                        xp, idx_train_val, cov_xx, dtype=xp.int64
                    )
                    cov_xx_train = cov_xx[cov_idx, :, :]
                    cov_xy_train = cov_xy[cov_idx, :, :]
                else:
                    cov_xx_train, cov_xy_train = None, None
                metric[ir] = _crossval(
                    model.copy(),
                    [x[i] for i in idx_train_val],
                    [y[i] for i in idx_train_val],
                    cov_xx_train,
                    cov_xy_train,
                    lags,
                    fs,
                    regularization[ir],
                    k - 1,
                    xp,
                    seed=seed,
                    average=average,
                    verbose=verbose,
                )
            best_idx = int(xp.argmax(metric).item())
            regularization_split_i = regularization[best_idx]
        else:
            regularization_split_i = regularization
        fold_model = model.copy()
        fold_model._train(
            [x[i] for i in idx_train_val],
            [y[i] for i in idx_train_val],
            fs,
            tmin,
            tmax,
            regularization_split_i,
            # xp,
        )
        _, metric_test[split_i] = fold_model.predict(
            [stimulus[i] for i in idx_test], [response[i] for i in idx_test]
        )
        best_regularization.append(regularization_split_i)
    return metric_test, best_regularization


def _crossval(
    model,
    x,
    y,
    cov_xx,
    cov_xy,
    lags,
    fs,
    regularization,
    k,
    xp,
    average=True,
    verbose=True,
    seed=None,
):
    n_trials = len(x)
    k = _check_k(k, n_trials)
    splits = _trial_splits(n_trials, k, seed=seed, shuffle=True)

    if average is True:
        metric = xp.zeros(k, **creation_kwargs(x[0]))
    else:
        metric = xp.zeros((k, y[0].shape[-1]), **creation_kwargs(x[0]))

    for isplit in _progressbar(range(len(splits)), "Cross-validating", verbose=verbose):
        idx_val = splits[isplit]
        idx_train = sum(splits[:isplit] + splits[isplit + 1 :], [])
        if cov_xx is None:
            x_train = [x[i] for i in idx_train]
            y_train = [y[i] for i in idx_train]
            cov_xx_hat, cov_xy_hat = covariance_matrices(
                x_train, y_train, lags, model.zeropad, preload=False
            )
        else:
            cov_idx = asarray_like(xp, idx_train, cov_xx, dtype=xp.int64)
            cov_xx_hat = cov_xx[cov_idx].mean(axis=0)
            cov_xy_hat = cov_xy[cov_idx].mean(axis=0)
        w = fit_weights_with_covariance_matrices(
            cov_xx_hat, cov_xy_hat, fs, regularization, model.method
        )
        trf = model.copy()
        trf.times = asarray_like(xp, lags, x[0]) / fs
        trf.bias, trf.fs = w[0:1], fs
        if trf.bias.ndim == 1:
            trf.bias = xp.expand_dims(trf.bias, 1)
        trf.weights = weights_to_trf(
            w[1:], x[0].shape[-1], len(lags), y[0].shape[-1], xp
        )
        x_test, y_test = [x[i] for i in idx_val], [y[i] for i in idx_val]
        # because we are working with covariance matrices, we have to check direction
        # to pass the right variable as stimulus and response to TRF.predict
        if model.direction == 1:
            _, metric_test = trf.predict(x_test, y_test, None, average)
        else:
            _, metric_test = trf.predict(y_test, x_test, None, average)
        metric[isplit] = metric_test
    return metric.mean(axis=0)


def permutation_distribution(
    model,
    stimulus,
    response,
    fs,
    tmin,
    tmax,
    regularization,
    n_permute,
    k=-1,
    seed=None,
    average=True,
    verbose=True,
):
    """
    Estimate the distribution of correlation coefficients and mean squared error
    under random permutation.

    For each permutation, stimulus and response trials are randomly shuffled and
    split into `k` segments. Then `k-1` segments are used to train and the remaining
    segment is used to test the model. The resulting permutation distribution reflects
    the expected correlation and error if there is no causal relationship between
    stimulus and response. To save time, the models are computed for all possible
    combinations of response and stimulus and then sampled and averaged during
    permutation.

    Parameters
    ----------
    model: model.TRF
        Base model used for cross-validation.
    stimulus: list
        Each element must contain one trial's stimulus in a two-dimensional
        samples-by-features array (second dimension can be omitted if there is
        only a single feature.
    response: list
        Each element must contain one trial's response in a two-dimensional
        samples-by-channels array.
    fs: int
        Sample rate of stimulus and response in hertz.
    tmin: float
        Minimum time lag in seconds.
    tmax: float
        Maximum time lag in seconds.
    regularization: float or int
        Value for the lambda parameter regularizing the regression.
    k: int
        Number of data splits, if -1, do leave-one-out cross-validation.
    seed: int
        Seed for the random number generator.
    average: bool or list or numpy.ndarray
        If True (default), average metric across all predicted features (e.g. channels
        in the case of forward modelling). If `average` is an array of indices only
        average the metric for those features. If `False`, return each feature's metric.
    Returns
    -------
    metric: float or numpy.ndarray
        Metric as computed by the metric function in  the attribute `model.metric`
        for each permutation.
    """
    stimulus, xps = _check_data(stimulus)
    response, xpr = _check_data(response)
    stimulus, response, n_trials = _check_length(stimulus, response)
    assert n_trials >= k, f"Not enough trials for {k}-fold cross-validation"
    if not xps == xpr:
        raise TypeError("stimulus and response trials must be of the same type!")
    else:
        xp = xps
    if seed:
        xp.random.seed(seed)
    x, y, tmin, tmax = _get_xy(stimulus, response, tmin, tmax, model.direction)
    min_len = min([len(x_i) for x_i in x])
    stimulus = [s[:min_len] for s in stimulus]
    response = [r[:min_len] for r in response]
    k = _check_k(k, n_trials)
    if n_permute > math.factorial(n_trials):
        warnings.warn(
            f"n_permute ({n_permute}) exceeds the number of unique permutations "
            f"({math.factorial(n_trials)}) for {n_trials} trials. "
            "Some permutations will be repeated."
        )
    idx = xp.arange(n_trials)
    combinations = xp.transpose(xp.meshgrid(idx, idx)).reshape(-1, 2)
    models = []
    for c in _progressbar(combinations, "Preparing models", verbose=verbose):
        trf = model.copy()
        trf.train(stimulus[c[0]], response[c[1]], fs, tmin, tmax, regularization)
        models.append(trf)
    metric = []
    for iperm in _progressbar(range(n_permute), "Permuting", verbose=verbose):
        idx = []
        for i in range(len(x)):  # make sure each x only appears once
            idx.append(random.choice(xp.where(combinations[:, 0] == i)[0]))
        random.shuffle(idx)
        idx = xp.array_split(idx, k)
        perm_metric = []
        for isplit in range(k):
            idx_val = idx[isplit]
            idx_train = xp.concatenate(idx[:isplit] + idx[isplit + 1 :])
            perm_model = xp.mean([models[i] for i in idx_train])
            stimulus_val = [stimulus[combinations[i][0]] for i in idx_val]
            response_val = [response[combinations[i][1]] for i in idx_val]
            _, fold_metric = perm_model.predict(
                stimulus_val, response_val, None, average
            )
            perm_metric.append(fold_metric)
        metric.append(xp.mean(xp.stack(perm_metric), axis=0))

    return xp.stack(metric)


def _progressbar(it, prefix="", size=50, out=sys.stdout, verbose=True):
    count = len(it)

    def show(j, verbose):
        x = int(size * j / count)
        if verbose:
            print(
                "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
                end="\r",
                file=out,
                flush=True,
            )

    show(0, verbose)
    for i, item in enumerate(it):
        yield item
        show(i + 1, verbose)
    if verbose:
        print("\n", flush=True, file=out)


def _check_k(k, n_trials):
    if not n_trials > 1:
        raise ValueError("Cross validation requires multiple trials!")
    if n_trials < k:
        raise ValueError("Number of splits can't be greater than number of trials!")
    if k == -1:  # do leave-one-out cross-validation
        k = n_trials
    return k


def _trial_splits(n_trials, k, seed=None, shuffle=False):
    """Split host-side trial indices without creating device scalar indices."""
    indices = list(range(n_trials))
    if shuffle:
        random.Random(seed).shuffle(indices)
    quotient, remainder = divmod(n_trials, k)
    sizes = [quotient + (i < remainder) for i in range(k)]
    splits = []
    start = 0
    for size in sizes:
        splits.append(indices[start : start + size])
        start += size
    return splits
