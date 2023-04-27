#!/usr/bin/env python
# utils cite from CPSC 540.py
# Author: Shurui Feng

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

from detector import NP, Bayes
from classifier import LDA

# make sure we're working in the directory this file lives in,
# for simplicity with imports and relative paths
os.chdir(Path(__file__).parent.resolve())

# question code
from utils import (
    load_dataset,
    main,
    handle,
    run,
)


def eval_models(models, ds_name):
    # X, y, Xtest, ytest = load_dataset(ds_name, "X", "y", "Xtest", "ytest")
    x, y = load_dataset(ds_name, "x", "y")
    for model in models:
        yhat = model.predict(x)
        yield np.mean(yhat != y)


def eval_model(model, ds_name):
    return next(eval_models([model], ds_name))


@handle("data")
def generate_test_data():
    N = 1000
    xs = []
    ys = []
    for _ in range(N):
        rand = random.random()
        if rand < 0.6:
            ys.append(1)
            xs.append(norm.rvs(loc=1, scale=1, size=1)[0])
        else:
            ys.append(0)
            xs.append(norm.rvs(loc=-1, scale=1, size=1)[0])
    d = {'x': xs, 'y': ys}
    df = pd.DataFrame(data=d)
    df.to_pickle("some_data.pkl")


@handle("detector")
def detector():
    model = NP()
    print(f"Neyman-Pearson detector test error: {eval_model(model, 'test_data'):.1%}")

    model = Bayes()
    print(f"Bayes-Optimum detector test error: {eval_model(model, 'test_data'):.1%}")


@handle("classifier")
def detector():
    x, y = load_dataset('train_data', "x", "y")
    model = LDA(x.to_numpy(), y.to_numpy())
    print(f"LDA test error: {eval_model(model, 'test_data'):.1%}")


if __name__ == "__main__":
    main()
