# %%
# Libraries
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
import pickle
import gc
import traceback
import logging
import random

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeavePGroupsOut,
    train_test_split,
    KFold,
    PredefinedSplit
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
from sklearn.utils import resample
from sklearn.cross_decomposition import PLSRegression

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import scipy.stats as st
from scipy.stats import zscore as zscore

from joblib import Parallel, delayed

from adapt.feature_based import FA
from adapt.instance_based import (
    TrAdaBoostR2,
    TwoStageTrAdaBoostR2,
    BalancedWeighting
)

# -----------------------------------------------------------------------------
# Disable GPU
# -----------------------------------------------------------------------------

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

AA_TARGET_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_REPEATS = 10
BASE_SEED = 42

# -----------------------------------------------------------------------------
# Command-line argument for fold number
# -----------------------------------------------------------------------------

if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 4
else:
    fold_num = 0

print(f"Fold {fold_num}")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

abcd_dir = '/nesi/nobackup/uoo03493/farzane/abcd/'
root_dir = '/nesi/nobackup/uoo03493/farzane/abcd/Ethnicity/'

tables_path = root_dir

fold_base_path = os.path.join(root_dir, 'folds/tsp/')
fold_dir = os.path.join(fold_base_path, f'Fold_{fold_num}/')
pls_dir = os.path.join(fold_dir, 'pls/')

if not os.path.isdir(fold_dir):
    os.mkdir(fold_dir)

if not os.path.isdir(pls_dir):
    os.mkdir(pls_dir)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

features_name = 'abccSmri'

featuresd = joblib.load(
    abcd_dir + features_name + '.joblib'
)

demo = (
    pd.read_csv(
        abcd_dir + 'demo_nesi.csv',
        index_col=0
    )
    .dropna()
)

targs = (
    pd.read_csv(
        abcd_dir + 'cog_all.csv',
        index_col=0,
        low_memory=False
    )
    .dropna()
)

targs.index = targs.index.str.replace('_', '')

# -----------------------------------------------------------------------------
# Helper to extract PLS coefficients from fitted adapt model
# -----------------------------------------------------------------------------

def extract_pls_from_adapt(fa_model):
    """
    Try to extract an inner fitted PLSRegression object
    (or its coefficients) from a fitted TwoStageTrAdaBoostR2 instance.

    Returns:
        coef
        y_mean
        estimator_obj_or_None
    """

    candidates = [
        'final_estimator_',
        'estimator_',
        'estimator',
        'model',
        'regressor_'
    ]

    coef = None
    y_mean = None
    est_obj = None

    for attr in candidates:

        if hasattr(fa_model, attr):

            try:
                est = getattr(fa_model, attr)

                # Pipeline case
                if hasattr(est, 'named_steps'):
                    last = list(est.named_steps.values())[-1]
                    est = last

                # Ensemble case
                if isinstance(est, (list, tuple, np.ndarray)):

                    for candidate in reversed(est):

                        if (
                            hasattr(candidate, 'coef_')
                            or isinstance(candidate, PLSRegression)
                        ):
                            est_obj = candidate
                            break

                else:
                    est_obj = est

                if est_obj is not None:

                    if hasattr(est_obj, 'coef_'):

                        coef = est_obj.coef_.copy()

                        y_mean = getattr(
                            est_obj,
                            'y_mean_',
                            getattr(est_obj, 'y_mean', None)
                        )

                        return coef, y_mean, est_obj

            except Exception:
                pass

    try:

        if hasattr(fa_model, 'estimators_'):

            for est in reversed(fa_model.estimators_):

                if hasattr(est, 'coef_'):

                    return (
                        est.coef_.copy(),
                        getattr(
                            est,
                            'y_mean_',
                            getattr(est, 'y_mean', None)
                        ),
                        est
                    )

    except Exception:
        pass

    return None, None, None

# -----------------------------------------------------------------------------
# FA with PLS function
# -----------------------------------------------------------------------------

def run_fa_pls(
    path_out,
    target_name,
    pls_dict,
    x_tr,
    x_te,
    y_tr,
    y_te,
    train_WAidx,
    train_AAidx,
    test_WAidx,
    test_AAidx,
    nc
):

    try:

        # ---------------------------------------------------------------------
        # Scaling
        # ---------------------------------------------------------------------

        scalerx = StandardScaler()

        x_tr_scaled = scalerx.fit_transform(x_tr.values)

        x_tr_scaleddf = pd.DataFrame(
            x_tr_scaled,
            index=x_tr.index,
            columns=x_tr.columns
        )

        x_te_scaled = scalerx.transform(x_te.values)

        x_te_scaleddf = pd.DataFrame(
            x_te_scaled,
            index=x_te.index,
            columns=x_te.columns
        )

        scalery = StandardScaler()

        y_tr_scaled = scalery.fit_transform(y_tr.values)

        y_tr_scaleddf = pd.DataFrame(
            y_tr_scaled,
            index=y_tr.index,
            columns=[target_name]
        )

        y_te_scaled = scalery.transform(y_te.values)

        y_te_scaleddf = pd.DataFrame(
            y_te_scaled,
            index=y_te.index,
            columns=[target_name]
        )

        # ---------------------------------------------------------------------
        # WA source matrices
        # ---------------------------------------------------------------------

        x_tr_wa = x_tr_scaleddf.loc[train_WAidx].values

        y_tr_wa = (
            y_tr_scaleddf.loc[train_WAidx]
            .values
            .ravel()
        )

        # ---------------------------------------------------------------------
        # Fixed params
        # ---------------------------------------------------------------------

        best_nc = max(2, nc)

        base_estimator = PLSRegression(
            n_components=best_nc,
            scale=False
        )

        best_total = 30
        best_fs = 10
        best_lr = 0.1

        train_AAidx_sorted = sorted(train_AAidx)

        # ---------------------------------------------------------------------
        # Worker function
        # ---------------------------------------------------------------------

        def fa_single_repeat(rep, aa_n):

            try:

                seed = BASE_SEED + aa_n * 100 + rep

                rng = np.random.RandomState(seed)

                aa_subset_idx = rng.choice(
                    train_AAidx_sorted,
                    size=aa_n,
                    replace=False
                )

                x_aa = (
                    x_tr_scaleddf
                    .loc[aa_subset_idx]
                    .values
                )

                y_aa = (
                    y_tr_scaleddf
                    .loc[aa_subset_idx]
                    .values
                    .ravel()
                )

                fa_model = TwoStageTrAdaBoostR2(
                    estimator=base_estimator,
                    n_estimators=best_total,
                    n_estimators_fs=best_fs,
                    lr=best_lr,
                    Xt=x_aa,
                    yt=y_aa,
                    random_state=seed,
                    verbose=0
                )

                fa_model.fit(
                    x_tr_wa,
                    y_tr_wa.ravel()
                )

                tr_pred = fa_model.predict(x_tr_scaled)
                te_pred = fa_model.predict(x_te_scaled)

                # -------------------------------------------------------------
                # AA MAE
                # -------------------------------------------------------------

                if len(test_AAidx) > 0:

                    y_te_scaled_aa = (
                        y_te_scaleddf
                        .loc[test_AAidx]
                        .values
                        .flatten()
                    )

                    te_pred_aa = (
                        pd.Series(
                            te_pred.ravel(),
                            index=x_te_scaleddf.index
                        )
                        .loc[test_AAidx]
                        .values
                        .flatten()
                    )

                    mae_aa = mean_absolute_error(
                        y_te_scaled_aa,
                        te_pred_aa
                    )

                else:
                    mae_aa = np.nan

                # -------------------------------------------------------------
                # Extract coefficients
                # -------------------------------------------------------------

                try:
                    coef, y_mean, est_obj = (
                        extract_pls_from_adapt(fa_model)
                    )

                except Exception:
                    coef, y_mean, est_obj = (
                        None,
                        None,
                        None
                    )

                return {
                    "rep": rep,
                    "aa_subset_idx": aa_subset_idx.tolist(),
                    "tr_pred": tr_pred,
                    "te_pred": te_pred,
                    "mae_aa": mae_aa,
                    "coef": (
                        coef.copy()
                        if coef is not None
                        else None
                    ),
                    "y_mean": (
                        y_mean
                        if y_mean is not None
                        else None
                    )
                }

            except Exception as e:

                return {
                    "rep": rep,
                    "error": str(e),
                    "trace": traceback.format_exc()
                }

        # ---------------------------------------------------------------------
        # Loop over AA sizes
        # ---------------------------------------------------------------------

        for sim_round, aa_n in enumerate(AA_TARGET_SIZES):

            if aa_n > len(train_AAidx):
                continue

            print(f"\nAA size = {aa_n}")

            size_key = f"AA_{aa_n}"

            results = Parallel(
                n_jobs=-1,
                backend='loky'
            )(
                delayed(fa_single_repeat)(rep, aa_n)
                for rep in range(N_REPEATS)
            )

            results = sorted(
                results,
                key=lambda d: d.get("rep", -1)
            )

            # -------------------------------------------------------------
            # Containers
            # -------------------------------------------------------------

            per_repeat_te = {}
            per_repeat_tr = {}
            per_repeat_idx = {}
            per_repeat_mae = {}

            all_te_arrays = []
            all_tr_arrays = []

            best_mae = np.inf
            best_rep = None
            best_coef = None
            best_y_mean_ = None
            best_train_idx_for_best = None
            best_aa_subset_for_refit = None

            # -------------------------------------------------------------
            # Populate containers
            # -------------------------------------------------------------

            for r in results:

                rep = r.get("rep", None)

                if rep is None:
                    continue

                rep_key = f"rep_{rep}"

                if "error" in r:

                    per_repeat_idx[rep_key] = None

                    per_repeat_tr[rep_key] = pd.DataFrame(
                        np.full(
                            (len(x_tr_scaleddf), 1),
                            np.nan
                        ),
                        index=x_tr_scaleddf.index,
                        columns=[target_name]
                    )

                    per_repeat_te[rep_key] = pd.DataFrame(
                        np.full(
                            (len(x_te_scaleddf), 1),
                            np.nan
                        ),
                        index=x_te_scaleddf.index,
                        columns=[target_name]
                    )

                    per_repeat_mae[rep_key] = np.nan

                    print(
                        f"Repeat {rep} error: "
                        f"{r.get('error')}"
                    )

                    continue

                tr_df = pd.DataFrame(
                    r["tr_pred"],
                    index=x_tr_scaleddf.index,
                    columns=[target_name]
                )

                te_df = pd.DataFrame(
                    r["te_pred"],
                    index=x_te_scaleddf.index,
                    columns=[target_name]
                )

                per_repeat_tr[rep_key] = tr_df
                per_repeat_te[rep_key] = te_df

                per_repeat_idx[rep_key] = r["aa_subset_idx"]

                per_repeat_mae[rep_key] = r["mae_aa"]

                all_tr_arrays.append(
                    tr_df.values.flatten()
                )

                all_te_arrays.append(
                    te_df.values.flatten()
                )

                mae_val = r["mae_aa"]

                if (
                    (not np.isnan(mae_val))
                    and (mae_val < best_mae)
                ):

                    best_mae = mae_val
                    best_rep = rep

                    best_coef = (
                        r["coef"].copy()
                        if r.get("coef", None) is not None
                        else None
                    )

                    best_y_mean_ = r.get(
                        "y_mean",
                        None
                    )

                    best_train_idx_for_best = (
                        r["aa_subset_idx"]
                    )

                    best_aa_subset_for_refit = (
                        r["aa_subset_idx"]
                    )

            # -------------------------------------------------------------
            # Average predictions
            # -------------------------------------------------------------

            if len(all_te_arrays) == 0:

                tr_y_pls = pd.DataFrame(
                    np.full(
                        (len(x_tr_scaleddf), 1),
                        np.nan
                    ),
                    index=x_tr_scaleddf.index,
                    columns=[target_name]
                )

                te_y_pls = pd.DataFrame(
                    np.full(
                        (len(x_te_scaleddf), 1),
                        np.nan
                    ),
                    index=x_te_scaleddf.index,
                    columns=[target_name]
                )

            else:

                te_avg = np.mean(
                    np.stack(all_te_arrays, axis=0),
                    axis=0
                )

                tr_avg = np.mean(
                    np.stack(all_tr_arrays, axis=0),
                    axis=0
                )

                tr_y_pls = pd.DataFrame(
                    tr_avg,
                    index=x_tr_scaleddf.index,
                    columns=[target_name]
                )

                te_y_pls = pd.DataFrame(
                    te_avg,
                    index=x_te_scaleddf.index,
                    columns=[target_name]
                )

            # -------------------------------------------------------------
            # Performance
            # -------------------------------------------------------------

            perf = {
                'nmse': [],
                'r2': [],
                'pr': [],
                'mae': []
            }

            # -------------------------
            # Train
            # -------------------------

            try:

                perf['nmse'].append(
                    mean_squared_error(
                        y_tr_scaled,
                        tr_y_pls.values.flatten()
                    )
                )

                perf['r2'].append(
                    r2_score(
                        y_tr_scaled,
                        tr_y_pls.values.flatten()
                    )
                )

                perf['pr'].append(
                    pearsonr(
                        y_tr_scaled.flatten(),
                        tr_y_pls.values.flatten()
                    )[0]
                )

                perf['mae'].append(
                    mean_absolute_error(
                        y_tr_scaled,
                        tr_y_pls.values.flatten()
                    )
                )

            except Exception:

                perf['nmse'].append(np.nan)
                perf['r2'].append(np.nan)
                perf['pr'].append(np.nan)
                perf['mae'].append(np.nan)

            # -------------------------
            # Test all
            # -------------------------

            try:

                perf['nmse'].append(
                    mean_squared_error(
                        y_te_scaled,
                        te_y_pls.values.flatten()
                    )
                )

                perf['r2'].append(
                    r2_score(
                        y_te_scaled,
                        te_y_pls.values.flatten()
                    )
                )

                perf['pr'].append(
                    pearsonr(
                        y_te_scaled.flatten(),
                        te_y_pls.values.flatten()
                    )[0]
                )

                perf['mae'].append(
                    mean_absolute_error(
                        y_te_scaled,
                        te_y_pls.values.flatten()
                    )
                )

            except Exception:

                perf['nmse'].append(np.nan)
                perf['r2'].append(np.nan)
                perf['pr'].append(np.nan)
                perf['mae'].append(np.nan)

            # -------------------------
            # Test AA
            # -------------------------

            if len(test_AAidx) > 0:

                try:

                    y_te_scaled_aa = (
                        y_te_scaleddf
                        .loc[test_AAidx]
                        .values
                        .flatten()
                    )

                    te_y_p_aa_avg = (
                        te_y_pls
                        .loc[test_AAidx]
                        .values
                        .flatten()
                    )

                    perf['nmse'].append(
                        mean_squared_error(
                            y_te_scaled_aa,
                            te_y_p_aa_avg
                        )
                    )

                    perf['r2'].append(
                        r2_score(
                            y_te_scaled_aa,
                            te_y_p_aa_avg
                        )
                    )

                    perf['pr'].append(
                        pearsonr(
                            y_te_scaled_aa.flatten(),
                            te_y_p_aa_avg.flatten()
                        )[0]
                    )

                    perf['mae'].append(
                        mean_absolute_error(
                            y_te_scaled_aa,
                            te_y_p_aa_avg
                        )
                    )

                except Exception:

                    perf['nmse'].append(np.nan)
                    perf['r2'].append(np.nan)
                    perf['pr'].append(np.nan)
                    perf['mae'].append(np.nan)

            else:

                perf['nmse'].append(np.nan)
                perf['r2'].append(np.nan)
                perf['pr'].append(np.nan)
                perf['mae'].append(np.nan)

            # -------------------------
            # Test WA
            # -------------------------

            if len(test_WAidx) > 0:

                try:

                    y_te_scaled_wa = (
                        y_te_scaleddf
                        .loc[test_WAidx]
                        .values
                        .flatten()
                    )

                    te_y_p_wa_avg = (
                        te_y_pls
                        .loc[test_WAidx]
                        .values
                        .flatten()
                    )

                    perf['nmse'].append(
                        mean_squared_error(
                            y_te_scaled_wa,
                            te_y_p_wa_avg
                        )
                    )

                    perf['r2'].append(
                        r2_score(
                            y_te_scaled_wa,
                            te_y_p_wa_avg
                        )
                    )

                    perf['pr'].append(
                        pearsonr(
                            y_te_scaled_wa.flatten(),
                            te_y_p_wa_avg.flatten()
                        )[0]
                    )

                    perf['mae'].append(
                        mean_absolute_error(
                            y_te_scaled_wa,
                            te_y_p_wa_avg
                        )
                    )

                except Exception:

                    perf['nmse'].append(np.nan)
                    perf['r2'].append(np.nan)
                    perf['pr'].append(np.nan)
                    perf['mae'].append(np.nan)

            else:

                perf['nmse'].append(np.nan)
                perf['r2'].append(np.nan)
                perf['pr'].append(np.nan)
                perf['mae'].append(np.nan)

            performance = pd.DataFrame(
                perf,
                index=[
                    'train',
                    'test',
                    'test_aa',
                    'test_wa'
                ]
            )

            print(performance)

            # -------------------------------------------------------------
            # Refit best model
            # -------------------------------------------------------------

            best_model_obj = None

            if best_aa_subset_for_refit is not None:

                try:

                    x_aa_best = (
                        x_tr_scaleddf
                        .loc[best_aa_subset_for_refit]
                        .values
                    )

                    y_aa_best = (
                        y_tr_scaleddf
                        .loc[best_aa_subset_for_refit]
                        .values
                        .ravel()
                    )

                    fa_model_refit = (
                        TwoStageTrAdaBoostR2(
                            estimator=base_estimator,
                            n_estimators=best_total,
                            n_estimators_fs=best_fs,
                            lr=best_lr,
                            Xt=x_aa_best,
                            yt=y_aa_best,
                            random_state=(
                                BASE_SEED
                                + aa_n * 100
                                + best_rep
                            ),
                            verbose=0
                        )
                    )

                    fa_model_refit.fit(
                        x_tr_wa,
                        y_tr_wa.ravel()
                    )

                    best_model_obj = fa_model_refit

                except Exception as e:

                    best_model_obj = None

                    print(
                        "Refit best model failed:",
                        str(e)
                    )

            # -------------------------------------------------------------
            # Save outputs
            # -------------------------------------------------------------

            if key not in pls_dict:
                pls_dict[key] = {}

            if size_key not in pls_dict[key]:

                pls_dict[key][size_key] = {
                    'model': {},
                    'data': {}
                }

            selections = {}

            for rep_k in per_repeat_te.keys():

                selections[rep_k] = {
                    "train_idx": per_repeat_idx[rep_k],
                    "yptest": per_repeat_te[rep_k],
                    "yptrain": per_repeat_tr[rep_k],
                    "mae_test_aa": per_repeat_mae[rep_k]
                }

            pls_dict[key][size_key]['data'] = {
                'selections': selections,
                'Xtrain': x_tr_scaleddf.copy(),
                'Xtest1': x_te_scaleddf.copy(),
                'yptrain': tr_y_pls,
                'yptest': te_y_pls,
                'yttrain': y_tr_scaleddf.copy(),
                'yttest': y_te_scaleddf.copy()
            }

            best_info = {
                'perf': performance,
                'best_n_estimators': best_total,
                'best_n_estimators_fs': best_fs,
                'best_learning_rate': best_lr,
                'best_n_components': best_nc,
                'AA_target_n': aa_n,
                'seed_base': BASE_SEED,
                'N_repeats': N_REPEATS,
                'best_rep': best_rep,
                'best_mae_test_aa': (
                    best_mae
                    if np.isfinite(best_mae)
                    else np.nan
                ),
                'best_train_idx': best_train_idx_for_best
            }

            if best_coef is not None:
                best_info['best_coefs'] = best_coef.copy()

            elif best_model_obj is not None:
                print('best_coef was None')

            best_info['best_model_obj'] = best_model_obj

            if best_y_mean_ is not None:
                best_info['best_y_mean_'] = best_y_mean_

            pls_dict[key][size_key]['model'].update(
                best_info
            )

            gc.collect()

    except Exception:
        logging.error(traceback.format_exc())

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

print(f"\nStarting Fold {fold_num}...")

Fold = joblib.load(
    root_dir + 'abcc_2split.joblib'
)

train_index = Fold[fold_num][0]
test_index = Fold[fold_num][1]

pls_outs = pd.read_csv(
    root_dir + 'pls_nc_table.csv'
)

targ_list = ['nihtbx_totalcomp_uncorrected']

for targ in targ_list:

    target_name = targ.split('_')[1][0:5] + '_'

    pls_dict = {}
    pls_dict_wa = {}
    pls_dict_aa = {}
    pls_dict_aawa = {}

    for key, featu in featuresd.items():

        ncomp = pls_outs.loc[
            pls_outs["feature"] == key,
            f"{fold_num}_nc_All"
        ].values[0]

        feat = featu.dropna()

        common_ind = list(
            set(feat.index).intersection(
                set(demo.index),
                set(targs.index)
            )
        )

        feature = feat.loc[common_ind]
        demo_table = demo.loc[common_ind]

        train_idx = list(
            set(common_ind).intersection(
                set(train_index)
            )
        )

        test_idx = list(
            set(common_ind).intersection(
                set(test_index)
            )
        )

        print(
            f"Fold {fold_num} - "
            f"Train size: {len(train_idx)}, "
            f"Test size: {len(test_idx)}"
        )

        eth1 = demo_table[
            demo_table['race_ethnicity'] == 1
        ].index

        eth2 = demo_table[
            demo_table['race_ethnicity'] == 2
        ].index

        train_WAidx = list(
            set(eth1).intersection(
                set(train_idx)
            )
        )

        train_AAidx = list(
            set(eth2).intersection(
                set(train_idx)
            )
        )

        test_WAidx = list(
            set(eth1).intersection(
                set(test_idx)
            )
        )

        test_AAidx = list(
            set(eth2).intersection(
                set(test_idx)
            )
        )

        print(
            f"Fold {fold_num} - "
            f"Train wa size: {len(train_WAidx)}, "
            f"Train aa size: {len(train_AAidx)}"
        )