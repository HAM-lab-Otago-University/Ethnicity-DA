# %%

# Libraries
import pandas as pd
import numpy as np
import os
import sys
import joblib
import warnings
import pickle
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeavePGroupsOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import scipy.stats as st
from scipy.stats import zscore as zscore
from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
import gc
import traceback
import logging
from sklearn.model_selection import KFold
import random
# Keep adapt imports even if not used so script structure is similar
from adapt.feature_based import FA
from adapt.instance_based import TrAdaBoostR2
from adapt.instance_based import TwoStageTrAdaBoostR2
from adapt.instance_based import BalancedWeighting
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Incremental AA sizes: include 0 (WA-only baseline) then increments
AA_TARGET_SIZES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_REPEATS = 30
BASE_SEED = 42
# Command-line argument for fold number
if len(sys.argv) > 1:
    fold_num = int(sys.argv[1]) % 4
else:
    fold_num = 0
print(f"Fold {fold_num}")

# Paths
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

# Load data
features_name = 'abccSmri'
featuresd = joblib.load(abcd_dir + features_name + '.joblib')
demo = pd.read_csv(abcd_dir + 'demo_nesi.csv', index_col=0).dropna()
targs = pd.read_csv(abcd_dir + 'cog_all.csv', index_col=0, low_memory=False).dropna()
targs.index = targs.index.str.replace('_', '')

# -----------------------------------------------------------------------------
# PLS incremental (no adaptation) function
# -----------------------------------------------------------------------------

def run_pls_incr(path_out, target_name, pls_dict, x_tr, x_te, y_tr, y_te,
                 train_WAidx, train_AAidx, test_WAidx, test_AAidx, nc):

    try:
        # ------------------------------------------------------------------
        # 1. Scaling (same as your script)
        # ------------------------------------------------------------------
        scalerx = StandardScaler()
        x_tr_scaled = scalerx.fit_transform(x_tr.values)
        x_tr_scaleddf = pd.DataFrame(x_tr_scaled, index=x_tr.index, columns=x_tr.columns)

        x_te_scaled = scalerx.transform(x_te.values)
        x_te_scaleddf = pd.DataFrame(x_te_scaled, index=x_te.index, columns=x_te.columns)

        scalery = StandardScaler()
        y_tr_scaled = scalery.fit_transform(y_tr.values)
        y_tr_scaleddf = pd.DataFrame(y_tr_scaled, index=y_tr.index, columns=[target_name])

        y_te_scaled = scalery.transform(y_te.values)
        y_te_scaleddf = pd.DataFrame(y_te_scaled, index=y_te.index, columns=[target_name])

        best_nc = max(2, nc)

        # Pre-sort AA for deterministic selection
        train_AAidx_sorted = sorted(train_AAidx)

        # ================================================================
        # Definition of ONE repeat (worker function for joblib)
        # ================================================================
        def run_one_repeat(rep, aa_n, train_WAidx, train_AAidx_sorted):

            seed = BASE_SEED + aa_n * 100 + rep
            rng = np.random.RandomState(seed)

            # ----------------------------------------------------------
            # Choose training subset
            # ----------------------------------------------------------
            if aa_n == 0:
                train_subset_idx = list(train_WAidx)
            else:
                aa_subset_idx = rng.choice(train_AAidx_sorted, size=aa_n, replace=False)
                train_subset_idx = list(set(train_WAidx).union(set(aa_subset_idx)))

            # ----------------------------------------------------------
            # Fit PLS
            # ----------------------------------------------------------
            X_train_sub = x_tr_scaleddf.loc[train_subset_idx].values
            y_train_sub = y_tr_scaleddf.loc[train_subset_idx].values.ravel()

            pls_model = PLSRegression(n_components=best_nc, scale=False)
            pls_model.fit(X_train_sub, y_train_sub)

            # ----------------------------------------------------------
            # Predictions
            # ----------------------------------------------------------
            tr_pred = pls_model.predict(x_tr_scaled)
            te_pred = pls_model.predict(x_te_scaled)

            tr_df = pd.DataFrame(tr_pred, index=x_tr.index, columns=[target_name])
            te_df = pd.DataFrame(te_pred, index=x_te.index, columns=[target_name])

            # ----------------------------------------------------------
            # Compute MAE on test AA (scaled space)
            # ----------------------------------------------------------
            if len(test_AAidx) > 0:
                y_te_scaled_aa = y_te_scaleddf.loc[test_AAidx].values.flatten()
                te_aa = te_df.loc[test_AAidx].values.flatten()
                mae_aa = mean_absolute_error(y_te_scaled_aa, te_aa)
            else:
                mae_aa = np.nan

            return {
                "rep": rep,
                "train_idx": train_subset_idx,
                "tr_df": tr_df,
                "te_df": te_df,
                "mae_aa": mae_aa,
                "coefs": pls_model.coef_.copy(),
                "y_mean": getattr(pls_model, "y_mean_", None),
            }

        # ------------------------------------------------------------------
        # 2. Main loop over AA sizes
        # ------------------------------------------------------------------
        for sim_round, aa_n in enumerate(AA_TARGET_SIZES):

            if aa_n > 0 and aa_n > len(train_AAidx):
                continue

            print(f"\nAA size = {aa_n}")
            size_key = f"AA_{aa_n}"

            # Run repetitions in parallel
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(run_one_repeat)(rep, aa_n, train_WAidx, train_AAidx_sorted)
                for rep in range(N_REPEATS)
            )

            # Sort results to ensure rep_0, rep_1, ...
            results = sorted(results, key=lambda d: d["rep"])

            # ================================================================
            # Build containers for saving
            # ================================================================
            per_repeat_te = {}
            per_repeat_tr = {}
            per_repeat_idx = {}
            per_repeat_mae = {}

            all_te_arrays = []
            all_tr_arrays = []

            best_mae = np.inf
            best_rep = None
            best_coefs = None
            best_y_mean = None
            best_train_idx_for_best = None

            for r in results:
                rep_key = f"rep_{r['rep']}"

                per_repeat_te[rep_key] = r["te_df"]
                per_repeat_tr[rep_key] = r["tr_df"]
                per_repeat_idx[rep_key] = r["train_idx"]
                per_repeat_mae[rep_key] = r["mae_aa"]

                all_te_arrays.append(r["te_df"].values.flatten())
                all_tr_arrays.append(r["tr_df"].values.flatten())

                if (not np.isnan(r["mae_aa"])) and r["mae_aa"] < best_mae:
                    best_mae = r["mae_aa"]
                    best_rep = r["rep"]
                    best_coefs = r["coefs"]
                    best_y_mean = r["y_mean"]
                    best_train_idx_for_best = r["train_idx"]

            # ================================================================
            # Average predictions across repeats
            # ================================================================
            te_avg = np.mean(np.stack(all_te_arrays, axis=0), axis=0)
            tr_avg = np.mean(np.stack(all_tr_arrays, axis=0), axis=0)

            tr_y_pls = pd.DataFrame(tr_avg, index=x_tr.index, columns=[target_name])
            te_y_pls = pd.DataFrame(te_avg, index=x_te.index, columns=[target_name])

            # ================================================================
            # Compute performance on averaged predictions
            # ================================================================
            perf = {'nmse': [], 'r2': [], 'pr': [], 'mae': []}

            # train
            perf['nmse'].append(mean_squared_error(y_tr_scaled, tr_avg))
            perf['r2'].append(r2_score(y_tr_scaled, tr_avg))
            perf['pr'].append(pearsonr(y_tr_scaled.flatten(), tr_avg.flatten())[0])
            perf['mae'].append(mean_absolute_error(y_tr_scaled, tr_avg))

            # test all
            perf['nmse'].append(mean_squared_error(y_te_scaled, te_avg))
            perf['r2'].append(r2_score(y_te_scaled, te_avg))
            perf['pr'].append(pearsonr(y_te_scaled.flatten(), te_avg.flatten())[0])
            perf['mae'].append(mean_absolute_error(y_te_scaled, te_avg))

            # test AA
            if len(test_AAidx) > 0:
                y_te_scaled_aa = y_te_scaleddf.loc[test_AAidx].values.flatten()
                te_avg_aa = te_y_pls.loc[test_AAidx].values.flatten()
                perf['nmse'].append(mean_squared_error(y_te_scaled_aa, te_avg_aa))
                perf['r2'].append(r2_score(y_te_scaled_aa, te_avg_aa))
                perf['pr'].append(pearsonr(y_te_scaled_aa, te_avg_aa)[0])
                perf['mae'].append(mean_absolute_error(y_te_scaled_aa, te_avg_aa))
            else:
                perf['nmse'].append(np.nan)
                perf['r2'].append(np.nan)
                perf['pr'].append(np.nan)
                perf['mae'].append(np.nan)

            # test WA
            if len(test_WAidx) > 0:
                y_te_scaled_wa = y_te_scaleddf.loc[test_WAidx].values.flatten()
                te_avg_wa = te_y_pls.loc[test_WAidx].values.flatten()
                perf['nmse'].append(mean_squared_error(y_te_scaled_wa, te_avg_wa))
                perf['r2'].append(r2_score(y_te_scaled_wa, te_avg_wa))
                perf['pr'].append(pearsonr(y_te_scaled_wa, te_avg_wa)[0])
                perf['mae'].append(mean_absolute_error(y_te_scaled_wa, te_avg_wa))
            else:
                perf['nmse'].append(np.nan)
                perf['r2'].append(np.nan)
                perf['pr'].append(np.nan)
                perf['mae'].append(np.nan)

            performance = pd.DataFrame(perf, index=['train','test','test_aa','test_wa'])
            print(performance)

            # ================================================================
            # Save outputs to pls_dict
            # ================================================================
            if key not in pls_dict:
                pls_dict[key] = {}
            if size_key not in pls_dict[key]:
                pls_dict[key][size_key] = {'model': {}, 'data': {}}

            # save per-repeat details
            selections = {}
            for rep_k in per_repeat_te.keys():
                selections[rep_k] = {
                    "train_idx": per_repeat_idx[rep_k],
                    "yptest": per_repeat_te[rep_k],
                    "yptrain": per_repeat_tr[rep_k],
                    "mae_test_aa": per_repeat_mae[rep_k],
                }

            pls_dict[key][size_key]['data'] = {
                'selections': selections,
                'Xtest1': x_te_scaleddf.copy(),
                'yptrain': tr_y_pls,
                'yptest': te_y_pls,
                'yttrain': y_tr_scaleddf.copy(),
                'yttest': y_te_scaleddf.copy()
            }

            # save model-level info
            best_info = {
                'perf': performance,
                'n_components': best_nc,
                'AA_target_n': aa_n,
                'seed_base': BASE_SEED,
                'N_repeats': N_REPEATS,
                'best_rep': best_rep,
                'best_mae_test_aa': (best_mae if np.isfinite(best_mae) else np.nan),
                'best_train_idx': best_train_idx_for_best,
                'best_coefs': best_coefs,
                'best_y_mean_': best_y_mean
            }

            pls_dict[key][size_key]['model'] = best_info

            gc.collect()

    except Exception as e:
        logging.error(traceback.format_exc())


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

print(f"\nStarting Fold {fold_num}...")
Fold = joblib.load(root_dir + 'abcc_2split.joblib')

train_index = Fold[fold_num][0]
test_index = Fold[fold_num][1]

pls_outs = pd.read_csv(root_dir + 'pls_nc_table.csv')

targ_list = ['nihtbx_totalcomp_uncorrected']

for targ in targ_list:

    target_name = targ.split('_')[1][0:5] + '_'

    pls_dict = {}
    pls_dict_wa = {}
    pls_dict_aa = {}
    pls_dict_aawa = {}

    for key, featu in featuresd.items():

        ncomp = pls_outs.loc[pls_outs["feature"] == key, f"{fold_num}_nc_All"].values[0]

        feat = featu.dropna()

        common_ind = list(set(feat.index).intersection(set(demo.index), set(targs.index)))
        feature = feat.loc[common_ind]
        demo_table = demo.loc[common_ind]

        train_idx = list(set(common_ind).intersection(set(train_index)))
        test_idx = list(set(common_ind).intersection(set(test_index)))

        print(f"Fold {fold_num} - Train size: {len(train_idx)}, Test size: {len(test_idx)}")

        eth1 = demo_table[demo_table['race_ethnicity'] == 1].index
        eth2 = demo_table[demo_table['race_ethnicity'] == 2].index

        train_WAidx = list(set(eth1).intersection(set(train_idx)))
        train_AAidx = list(set(eth2).intersection(set(train_idx)))
        test_WAidx = list(set(eth1).intersection(set(test_idx)))
        test_AAidx = list(set(eth2).intersection(set(test_idx)))

        print(f"Fold {fold_num} - Train wa size: {len(train_WAidx)}, Train aa size: {len(train_AAidx)}")

        x_tr = feature.loc[train_idx]
        x_te = feature.loc[test_idx]

        y_tr = pd.DataFrame(
            targs.loc[train_idx][targ].values,
            index=x_tr.index,
            columns=[target_name]
        )
        y_te = pd.DataFrame(
            targs.loc[test_idx][targ].values,
            index=x_te.index,
            columns=[target_name]
        )

        print('running incremental PLS (no adaptation)')

        run_pls_incr(
            pls_dir, targ, pls_dict,
            x_tr, x_te, y_tr, y_te,
            train_WAidx, train_AAidx,
            test_WAidx, test_AAidx,
            ncomp
        )

    print('save to disk')

    joblib.dump(
        pls_dict,
        pls_dir + target_name + features_name + '_pls_output_std_All_noDA_incr.joblib',
        compress=0
    )

    gc.collect()