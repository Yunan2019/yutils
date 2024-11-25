import os
import requests
import traceback
import numpy as np
import pandas as pd
import scipy.io as sio
from shutil import rmtree
from functools import wraps
from itertools import product
from scipy.stats import rankdata
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator



__all_LDL_metrics__ = ['Cheb', 'Canber', 'Clark', 'KL', 'Cosine', 'Intersec', 'Rho']
__all_LDL_datasets__ = ['FG-Net', 'Flickr', 'Human_Gene', 'M2B', 'MOR-PH', 'Movie', 'Natural_Scene', 
                        'RAF_ML', 'SBU_3DFE', 'SCUT_FBP', 'SJAFFE', 'Twitter', 'Yeast_alpha', 'Yeast_cdc',
                        'Yeast_cold', 'Yeast_diau', 'Yeast_dtt', 'Yeast_elu', 'Yeast_heat', 'Yeast_spo',
                        'Yeast_spo5', 'Yeast_spoem', 'emotion6', 'fbp5500', 'Abstract_Painting']

def _check_validity_for_LDL_metrics_(metric_fn):
    @wraps(metric_fn)
    def _func(inputs, targets):
        assert inputs.shape == targets.shape
        if len(inputs.shape) == 1:
            inputs, targets = inputs[None,:], targets[None,:]
        assert np.allclose(inputs.sum(1), 1)
        assert np.allclose(targets.sum(1), 1)
        assert ((inputs >= 0) & (targets >= 0)).all()
        return metric_fn(inputs, targets)
    return _func

@_check_validity_for_LDL_metrics_
def Intersec(inputs, targets):
    targets = targets.copy()
    mask = np.where(inputs < targets)
    targets[mask] = inputs[mask]
    return targets.sum(1)

@_check_validity_for_LDL_metrics_
def Cosine(inputs, targets):
    s = (targets * inputs).sum(1)
    m = np.linalg.norm(targets, ord=2, axis=1) * np.linalg.norm(inputs, ord=2, axis=1)
    return s / m

@_check_validity_for_LDL_metrics_
def Clark(inputs, targets):
    return np.sqrt((np.power(inputs - targets + 1e-9, 2) / np.power(inputs + targets + 1e-9, 2)).sum(1))

@_check_validity_for_LDL_metrics_
def Cheb(inputs, targets):
    return np.max(np.abs(targets - inputs), 1)

@_check_validity_for_LDL_metrics_
def Canber(inputs, targets):
    return (np.abs(targets - inputs + 1e-9) / (targets + inputs + 1e-9)).sum(1)

@_check_validity_for_LDL_metrics_
def KL(inputs, targets):
    return ( targets * (np.log(targets + 1e-9) - np.log(inputs + 1e-9)) ).sum(1)

@_check_validity_for_LDL_metrics_
def Rho(inputs, targets):
    rA, rB = rankdata(inputs, axis=1), rankdata(targets, axis=1)
    cov = ((rA - np.mean(rA, axis=1, keepdims=True)) * (rB - np.mean(rB, axis=1, keepdims=True))).mean(axis=1)
    std = np.std(rA, axis=1) * np.std(rB, axis=1)
    rho = cov / (std + 1e-9)
    return rho

def report(inputs, targets, metrics='all', ds_name=None, estimator=None, out_form='print'):
    '''
    `estimator`: The instantiation of an estimator
    `type`:
        'print': print pretty table
        'pandas': pandas Series
    `metrics`:
        1. all
        2. ['Rho', func], func is a function, not be lambda.
    '''
    assert out_form in ['print', 'pandas']
    assert (isinstance(metrics, list)) or (metrics == 'all')


    if metrics == 'all':
        metrics = __all_LDL_metrics__ 
    mfs = {'Cheb': Cheb, 'Canber': Canber, 'Clark': Clark, 'KL': KL, 'Cosine': Cosine, 'Intersec': Intersec, 'Rho': Rho}
    scores = [mfs[f](inputs, targets).mean() if isinstance(f, str) else f(inputs, targets).mean() for f in metrics]
    scores = np.round(np.array(scores), 3).tolist()
    showls = [m if isinstance(m, str) else m.__name__ for m in metrics]
    if estimator is not None:
        showls = ['estimator'] + showls
        if type(estimator) is type:
            scores = [estimator.__name__] + scores
        else:
            scores = [estimator.__class__.__name__] + scores
    if ds_name is not None:
        showls = ['dataset'] + showls
        scores = [ds_name] + scores
    if out_form == 'print':
        from prettytable import PrettyTable
        tb = PrettyTable()
        tb.field_names = showls
        tb.add_row(scores)
        print(tb)
    elif out_form == 'pandas':
        from pandas import Series
        res = Series(data=scores, index=showls)
        return res

def load_dataset(name, dir='Datasets'):
    if not os.path.exists(dir):
        print(f'Directory {dir} does not exist, creating it.')
        os.makedirs(dir)
    dataset_path = os.path.join(dir, name+'.mat')
    if not os.path.exists(dataset_path):
        print(f'Dataset {name}.mat does not exist, downloading it now, please wait...')
        url = f'https://raw.githubusercontent.com/SpriteMisaka/PyLDL/main/dataset/{name}.mat'
        response = requests.get(url)
        if response.status_code == 200:
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            print(f'Dataset {name}.mat downloaded successfully.')
        else:
            raise ValueError(f'Failed to download {name}.mat')
    data = sio.loadmat(dataset_path)
    return data['features'], data['labels']

def binarize(labels, t=0.5):
    labels = labels.copy()
    num_ins, num_labs = labels.shape
    logical_label = np.zeros_like(labels)
    for i in range(num_ins):
        y = labels[i]
        s = 0
        for _ in range(num_labs):
            j = np.argmax(y)
            s += y[j]
            labels[i, j], logical_label[i,j] = 0, 1.0
            if s >= t:
                break
    return logical_label

def single_model_eval(estimator, ds_name, data=None, ds_seed=0, test_size=0.3, feat_preprocesser='MinMaxScaler', 
            label_preprocesser=None, metrics='all', out_form='print'):
    '''
    Evaluate `estimator` on the dataset `ds`.
    `estimator` must have `fit`, `predict` (LDL) and `enhance` (LE).
    Requirement: 
        ds_seed, test_size: train_test_split(X, D, test_size=test_size, random_state=ds_seed)
        feat_preprocesser: sklearn.preprocessing estimators
        label_preprocesser: None --> an LDL task; a python function --> other tasks like LE, noisy LDL.
    '''
    if data is None:
        X, D = load_dataset(ds_name)
    else:
        X, D = data
    Xr, Xs, Dr, Ds = train_test_split(X, D, test_size=test_size, random_state=ds_seed)
    
    # preprocess the feature matrix
    if feat_preprocesser == 'MinMaxScaler':
        scaler = MinMaxScaler().fit(Xr)
        Xr, Xs = scaler.transform(Xr), scaler.transform(Xs)
    elif not (feat_preprocesser is None):
        scaler = feat_preprocesser.fit(Xr)
        Xr, Xs = scaler.transform(Xr), scaler.transform(Xs)

    # train models
    try:
        if label_preprocesser is None:
            model = estimator.fit(Xr, Dr)
        else:
            model = estimator.fit(Xr, label_preprocesser(Dr))
        Dhat = model.predict(Xs)
        Drec = model.predict(Xr) if label_preprocesser is None else model.enhance(Xr)
    except:
        print("Some errors appear in `fit`!")
        e = traceback.format_exc(); print(e); Dhat = None; Drec = None
    
    # output
    metrics = __all_LDL_metrics__ if metrics == 'all' else metrics
    showls = [m if isinstance(m, str) else m.__name__ for m in metrics]
    if out_form == 'print':
        if (Dhat is None) or (Drec is None): return
        print("# Training Performance:")
        report(Drec, Dr, metrics=metrics, ds_name=ds_name, estimator=estimator, out_form='print')
        print("# Testing Performance:")
        report(Dhat, Ds, metrics=metrics, ds_name=ds_name, estimator=estimator, out_form='print')
    else:
        if (Dhat is None) or (Drec is None):
            ser = pd.Series(data=[''] * len(showls) * 2 + [ds_name, estimator.__class__.__name__],
                index=[m + '-train' for m in showls] + [m + '-test' for m in showls] + ['dataset', 'estimator'])
        else:
            ser_tr = report(Drec, Dr, metrics=metrics, ds_name=ds_name, estimator=estimator, 
                out_form='pandas')[showls].rename({m: m + '-train' for m in showls})
            ser_ts = report(Dhat, Ds, metrics=metrics, ds_name=ds_name, estimator=estimator, 
                out_form='pandas')[showls + ['dataset', 'estimator']].rename({m: m + '-test' for m in showls})
            ser = pd.concat([ser_tr, ser_ts])
        ser['ds_seed'], ser['test_size'], ser['model_params'] = ds_seed, test_size, str(estimator.get_params())
        ser['feat_preprocesser'] = feat_preprocesser if feat_preprocesser is not None else 'None'
        return ser

def multiple_model_eval(estimator, ds_name, data=None, ds_seeds=range(10), test_size=0.3, 
                        feat_preprocesser='MinMaxScaler', label_preprocesser=None, metrics='all', n_jobs=1):
    res = Parallel(n_jobs=n_jobs)(delayed(single_model_eval)(estimator, ds_name, data=data, test_size=test_size, 
            feat_preprocesser=feat_preprocesser, label_preprocesser=label_preprocesser, metrics=metrics,
            out_form='pandas', ds_seed=seed) for seed in ds_seeds)
    for i in range(len(res)):
        res[i].name = i
    return pd.DataFrame(res)

def overall_model_eval(estimator, datasets, param_grid, out_file=None, ds_seeds=range(10), test_size=0.3, 
            feat_preprocesser='MinMaxScaler', label_preprocesser=None, metrics='all', max_eval=None, random_seed=123, n_jobs=1):
    '''
    Requirement:
        estimator: an estimator class
        datasets: a python dictionary or a list of dataset names:
            {'Emotion6': (X1, D1), 'Movie': (X2, D2)}
        param_grid: a python dictionary like:
            {'param1': [1e-1, 1e-2, 1e-3, ...],
            'param2': [1e-2, 1e-3, 1e-4, ...]}
        ds_seeds: list
        max_eval:
            How many combinations of hyperparameters will be evaluated
    '''
    # create a folder to save intermediate results
    np.random.seed(random_seed)
    def mkdir(path):
        path = path.strip()
        path = path.rstrip("\\")
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print('Success: The folder `' + path + '` is created.')
            return True
        else:
            print('Failure: The folder `' + path + '` already exists.')
            return False
    if out_file is None:
        temporary_file = '%s_overall_model_eval_temp_files' % estimator.__name__
    else: temporary_file = '%s_overall_model_eval_temp_files' % out_file
    folder_state = mkdir(temporary_file)
    if not folder_state:
        return None
    
    # create hyperparameter space
    params, params_name_ls = [], []
    for name in param_grid.keys():
        params_name_ls.append(name)
        params.append(param_grid[name])
    params.append(ds_seeds)
    if max_eval is None:
        param_combination = product(*params)
    else:
        paramls = list(product(*params))
        param_combination = []
        for seed in ds_seeds:
            temp = []
            for param_tup in paramls:
                if param_tup[-1] == seed:
                    temp.append(param_tup)
            np.random.shuffle(temp)
            param_combination.extend(temp[:max_eval])
    
    # create datasets
    if isinstance(datasets, list):
        datasets = {name: load_dataset(name) for name in datasets}

    # base
    def _func(p, counter):
        model_param = {name: p[j] for j, name in enumerate(params_name_ls)}
        ds_seed = p[-1]
        temp = pd.DataFrame()
        for i, (ds_name, (X, D)) in enumerate(datasets.items()):
            ser = single_model_eval(estimator(**model_param), ds_name, data=(X, D), test_size=test_size, 
                                    feat_preprocesser=feat_preprocesser, label_preprocesser=label_preprocesser, 
                                    out_form='pandas', ds_seed=ds_seed, metrics=metrics)
            temp = pd.concat([temp, pd.DataFrame(ser).T])
            temp.to_csv(temporary_file+'/%d.csv' % counter)
        return temp
    
    # main
    res = Parallel(n_jobs=n_jobs)(delayed(_func)(p, i) for i, p in enumerate(param_combination))
    res = pd.concat(res, axis=0)
    res.index = range(res.shape[0])
    out_file = estimator.__name__ if out_file is None else out_file
    res.to_csv('%s.csv' % out_file)
    rmtree(temporary_file)



class GeneraLDRegressor(BaseEstimator):
    '''
    NeuraLDL(criterion=(nn.MSELoss, {'reduction': 'mean'}))
    '''
    def __init__(self, criterion=None, lr=1e-3, max_iter=1000, tolerance_grad=1e-5, 
                 tolerance_change=1.4901161193847656e-8, random_seed=123):
        if criterion is None:
            self.criterion = (nn.MSELoss, {})
        else:
            self.criterion = criterion if isinstance(criterion, tuple) else (criterion, {})
        self.max_iter = max_iter
        self.lr = lr
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.random_seed = random_seed

    def fit(self, X, D):
        torch.manual_seed(self.random_seed)
        X, D = torch.FloatTensor(X), torch.FloatTensor(D) + 1e-5
        model = nn.Sequential(nn.Linear(X.shape[1], D.shape[1]), nn.Softmax(dim=1))
        for p in model.parameters():
            nn.init.zeros_(p)
        params = list(model.parameters())
        optimizer = torch.optim.LBFGS(params, lr=self.lr, max_iter=self.max_iter, tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change, history_size=5, line_search_fn='strong_wolfe', max_eval=None)
        criterion = self.criterion[0](**self.criterion[1])
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            Dhat = model(X)
            loss = criterion(Dhat, D).mean()
            if loss.requires_grad:
                loss.backward()
            return loss
        optimizer.step(closure)
        self.model = model
        return self

    def predict(self, X):
        X = torch.FloatTensor(X)
        with torch.no_grad():
            return self.model(X).numpy()