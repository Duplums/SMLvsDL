"""
We aim at giving learning curves of classical ML models on OpenBHB for:
* Train/Val/Test Split with 2 left-out studies
* VBM, Quasi-Raw and FSL pre-processing

"""

import numpy as np
import os, logging
import sys
import nibabel
from copy import deepcopy
sys.path.extend(["../", ".", "../../", "../../pylearn-mulm"])
import argparse
from datasets.open_bhb import OpenBHB, SubOpenBHB
from datasets.bhb_10k import BHB
from dl_training.preprocessing.combat import CombatModel
import pandas as pd
from sml_training.sk_trainer import MLTrainer
from dl_training.datamanager import Zscore, StandardScalerBiased
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, SGDClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, RFE, f_classif, f_regression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# nb_training_samples = [100, 500, 1000, 3000, 5000]
# total_nb_folds = [5, 5, 3, 3, 3]

logger = logging.getLogger("SMLvsDL")

def red_dim(X_tr, y_tr, *X_tests, meth, classif=True, nFeats=784, post_norm=False):
    X_tests_ = []
    if meth == 'UFS':
        # 1. UFS
        score_func = f_classif if classif else f_regression
        ufs = SelectKBest(score_func=score_func, k=nFeats)
        X_tr = ufs.fit_transform(X_tr, y_tr)
        for X_te in X_tests:
            X_tests_.append(ufs.transform(X_te))
    elif meth == 'RFE':
        # 2. RFE
        estim = SVC(kernel="linear", C=1) if classif else SVR(kernel="linear")
        rfe = RFE(estim, n_features_to_select=nFeats, step=0.20)
        rfe = rfe.fit(X_tr, y_tr)
        X_tr = X_tr[:, rfe.support_]
        for X_te in X_tests:
            X_tests_.append(X_te[:, rfe.support_])
    elif meth == 'GRP':
        # 3. GRP
        grp = GaussianRandomProjection(n_components=nFeats)
        X_tr = grp.fit_transform(X_tr, y_tr)
        for X_te in X_tests:
            X_tests_.append(grp.transform(X_te))
    else:
        print('Check Dim. Red. Method')
    if post_norm:
        logger.info("Applying post-normalization...")
        ss = StandardScaler().fit(X_tr)
        X_tr = ss.transform(X_tr)
        for i in range(len(X_tests_)):
            X_tests_[i] = ss.transform(X_tests_[i])

    logger.info('{} X_train {} '.format(meth, X_tr.shape) +
                ' '.join(['X_test ({}) {}'.format(i, X_te.shape) for i, X_te in enumerate(X_tests_)]))
    return X_tr, X_tests_


def residualize(formula_res: str, formula_full: str, df: pd.DataFrame, train_data: np.ndarray,
                *tests: (pd.DataFrame, np.ndarray), type: str="linear", discrete_vars=None,
                continuous_vars=None):
    """
    :param formula_res: e.g "site"
    :param formula_full: e.g "age + sex + site"
    :param df: DataFrame shape (n,) whose columns contain names defined in formula full
    :param train_data: numpy array of shape (n, *)
    :param tests: tuple (df_test, data_test) to be transformed
    :param type: str either "linear" for lin. reg. adjusted or "combat"
    NOTE: Only the "res" values are used to residualize tests data (e.g only "site")
    """
    assert type in ["linear", "combat"], "Unknown residualizer: %s"%type
    all_df = [df]
    train_index = np.arange(len(df))
    test_indexes, offset = [], len(df)
    for (df_test, data_test) in tests:
        all_df.append(df_test)
        test_indexes.append(offset + np.arange(len(df_test)))
        offset += len(df_test)
    all_df = pd.concat(all_df, ignore_index=True)

    if type == "linear":
        from mulm.residualizer import Residualizer
        residualizer = Residualizer(data=all_df, formula_res=formula_res, formula_full=formula_full)
        Zres = residualizer.get_design_mat(all_df)
        train_data_ = residualizer.fit_transform(train_data, Zres[train_index])
    else:
        residualizer = CombatModel()
        train_data_ = residualizer.fit_transform(train_data, all_df[["site"]].values[train_index],
                                                 discrete_covariates=all_df[discrete_vars].values[train_index],
                                                 continuous_covariates=all_df[continuous_vars].values[train_index])
    tests_res = []
    for ((df_test, data_test), test_index) in zip(tests, test_indexes):
        if type == "linear":
            tests_res.append(residualizer.transform(data_test,  Zres[test_index]))
        else:
            if not (set(all_df["site"].values[train_index]) >= set(all_df["site"].values[test_index])):
                missing_sites = set(all_df["site"].values[test_index]) - set(all_df["site"].values[train_index])
                print("Sites {} were not seen during ComBat fit(). No transform() for this test set".format(missing_sites))
                tests_res.append(data_test)
            else:
                tests_res.append(residualizer.transform(data_test, all_df[["site"]].values[test_index],
                                                        discrete_covariates=all_df[discrete_vars].values[test_index],
                                                        continuous_covariates=all_df[continuous_vars].values[test_index]))
    logger.info("Residualization %s performed !"%type)
    return (train_data_, *tests_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--saving_dir", required=True, type=str)
    parser.add_argument("--preproc", required=True, choices=["quasi_raw", "vbm"])
    parser.add_argument("--pb", required=True, choices=["age", "sex"])
    parser.add_argument("--model", required=True, choices=["Ridge", "SVR", "SVC", "LogisticRegression",
                                                           "ElasticNet", "RandomForestClassifier",
                                                           "RandomForestRegressor"])
    parser.add_argument("--N_train", type=int, required=True)
    parser.add_argument("--njobs", default=1, type=int)
    parser.add_argument("--folds", nargs='+', type=int)
    parser.add_argument("--nb_folds", type=int, required=True)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--no_reduc", action="store_true")
    parser.add_argument("--scaler", default="standard", choices=["standard", "zscore", "none"])
    parser.add_argument("--residualize", type=str, choices=["linear", "combat"])
    parser.add_argument("--post_norm", action="store_true")
    parser.add_argument("--red_meth", nargs="+", choices=["UFS", "RFE", "GRP"])
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--test_name", type=str, default="Test")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    root = args.root
    saving_dir = args.saving_dir

    scorings = {"age": "neg_mean_absolute_error",
                "sex": "balanced_accuracy"}
    try:
        m_vbm = nibabel.load(os.path.join(root, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
        m_quasi_raw = nibabel.load(os.path.join(root, "mni_raw_brain-mask_1.5mm.nii.gz"))
    except FileNotFoundError:
        raise FileNotFoundError("Brain masks not found. You can find them in /masks directory "
                                "and mv them to this directory: %s" % args.root)

    masks = {"vbm": m_vbm.get_data() != 0, "quasi_raw": m_quasi_raw.get_data() != 0}

    models = {"age": [Ridge, SVR, RandomForestRegressor, ElasticNet],
              "sex": [SVC, LogisticRegression, RandomForestClassifier, SGDClassifier]}

    post_norm = args.post_norm # If True, apply standard scaling to data reduced

    models_hyperparams = {"age": {"Ridge": {'alpha': [1e-1, 1, 10, 10**2, 10**3], 'fit_intercept': [True],
                                            'solver': ['svd', 'cholesky', 'sag']},
                                  "ElasticNet": {'alpha': 10. ** np.arange(-1, 4),
                                                 'l1_ratio': [.1, .5, .9]},
                                  "SVR": {'kernel': ['rbf'], 'C': [1, 1e-1, 1e1, 1e2]}
                                  },
                          "sex": {"SVC": {'kernel': ['rbf'], 'C': [1, 1e-1, 1e1, 1e2]},
                                  "LogisticRegression": {'C': 10. ** np.arange(-3, 2), 'solver': ['sag'],
                                                         'penalty': ['l2'],
                                                         'n_jobs': [6],
                                                         'fit_intercept': [True]},
                                  "ElasticNet": {'alpha': 10. ** np.arange(-1, 4),
                                                 'l1_ratio': [.1, .5, .9],
                                                 'loss': ['log'],
                                                 'penalty': ['elasticnet']},
                                  }
                          }

    dbs = {"train_val_test": "OpenBHB-Train-Val-Test", "cv": "OpenBHB-CV"}
    logger = logging.getLogger("SMLvsDL")

    red_methods = args.red_meth or ["UFS", "RFE", "GRP"]

    preproc = args.preproc
    label = args.pb
    pb = args.pb
    if pb == "sex" and args.model == "ElasticNet":
        model = SGDClassifier
    else:
        model = eval(args.model)
    assert model in models[pb]
    scheme = "train_val_test"
    db = dbs[scheme]
    if args.scaler == "standard":
        scaler = StandardScaler
    elif args.scaler == "zscore":
        scaler = Zscore
    else:
        scaler = None

    hyperparams = models_hyperparams[pb][args.model]
    mask = masks[args.preproc]
    scoring = scorings[pb]
    design_cols = ["age", "sex", "site"]
    labels_index = {d: i for (i,d) in enumerate(design_cols)}

    val = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="val", target=design_cols)
    test = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="test", target=design_cols)
    test_intra = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="test_intra", target=design_cols)
    val_data, target_val = val.get_data(mask=mask, dtype=np.float32)
    test_data, target_test = test.get_data(mask=mask, dtype=np.float32)
    test_intra_data, target_test_intra = test_intra.get_data(mask=mask, dtype=np.float32)
    y_val, y_test, y_test_intra = target_val[:, labels_index[label]].ravel(),  \
                                  target_test[:, labels_index[label]].ravel(), \
                                  target_test_intra[:, labels_index[label]].ravel()
    folds = list(args.folds or range(args.nb_folds))
    folds_per_job = np.array_split(folds, np.ceil(len(folds)/args.njobs))
    # Parallelize across folds
    for folds_job_i in folds_per_job:
        jobs = []
        logger.info("Processing folds {f} - Model {m} - N={s} - Label {l} - Preproc {p}".format(f=folds_job_i,
                                                                                                     m=args.model,
                                                                                                     s=args.N_train,
                                                                                                     l=label,
                                                                                                     p=preproc))
        for fold in folds_job_i:
            if args.N_train <= 6000:
                train = SubOpenBHB(root, preproc=preproc, scheme="train_val_test", split="train",
                                   target=design_cols, N_train_max=args.N_train,
                                   nb_folds=args.nb_folds, fold=fold, stratify=design_cols)
            else:
                train = BHB(root, preproc=preproc, scheme="train_val_test", split="train",
                            target=design_cols, fold=fold)
                logger.info("BHB set as training set: N(train)=%i"%len(train))

            (train_data, target_tr) = train.get_data(mask=mask, dtype=np.float32)
            y_tr = target_tr[:, labels_index[label]].ravel()

            # 0) Eventually Residualization with Linear Adjusted Reg. (NO DATA LEAKAGE on neither Age or Sex) or
            #    ComBAT (Data Leakage when transforming test set)
            if args.residualize is not None:
                df = pd.DataFrame(train.target, columns=design_cols)
                df_val = pd.DataFrame(val.target, columns=design_cols)
                df_test = pd.DataFrame(test.target, columns=design_cols)
                df_test_intra = pd.DataFrame(test_intra.target, columns=design_cols)
                df['site'] = df['site'].astype(object)
                df_val['site'] = df_val['site'].astype(object)
                df_test['site'] = df_test['site'].astype(object)
                df_test_intra['site'] = df_test_intra['site'].astype(object)

                if args.residualize == "linear":
                    (train_data_, val_data_, test_data_, test_intra_data_) = residualize("site", "site + age + sex",
                                                                                         df, train_data,
                                                                                         (df_val, val_data),
                                                                                         (df_test, test_data),
                                                                                         (df_test_intra, test_intra_data),
                                                                                         type=args.residualize)
                elif args.residualize == "combat":
                    (train_data_, val_data_, test_data_, test_intra_data_) = residualize("site", "site + age + sex",
                                                                             df, train_data,
                                                                             (df_val, val_data),
                                                                             (df_test, test_data),
                                                                             (df_test_intra, test_intra_data),
                                                                             type=args.residualize,
                                                                             continuous_vars=["age"],
                                                                             discrete_vars=["sex"])

            # 1) Normalization
            if scaler is not None:
                if args.residualize is not None:
                    ss = scaler().fit(train_data_)
                    train_data_ = ss.transform(train_data_)
                    val_data_ = ss.transform(val_data_)
                    test_data_ = ss.transform(test_data_)
                    test_intra_data_ = ss.transform(test_intra_data_)
                else:
                    ss = scaler().fit(train_data)
                    train_data_ = ss.transform(train_data)
                    val_data_ = ss.transform(val_data)
                    test_data_ = ss.transform(test_data)
                    test_intra_data_ = ss.transform(test_intra_data)
                logger.info('Data Normalized ...')
            model_name = type(model()).__name__
            exp_name = '{m}_{pb}_{db}_fold{k}_epoch{e}.pkl'.format(m=model_name, pb=pb.capitalize(),
                                                                   db='OpenBHB-Train-Val-Test',
                                                                   k=fold, e=100)
            if args.no_reduc:
                saving_dir_ = os.path.join(saving_dir, preproc, model_name, pb.capitalize(), 'N_%i' % args.N_train)
                if args.cv: val_data_ = None
                if args.test: train_data_ = None
                save_model = True
                if (issubclass(model, SVR) or issubclass(model, SVC)) and args.N_train > 6000:
                    save_model = False # too much memory
                X_test = [test_intra_data_, test_data_]
                y_test = [y_test_intra, y_test]
                test_names = ["%s_Intra-OpenBHB"%args.test_name, "%s_OpenBHB"%args.test_name]
                trainer = MLTrainer(model(), deepcopy(hyperparams), train_data_, y_tr,
                                    X_val=val_data_, y_val=y_val,
                                    X_test=X_test,
                                    y_test=y_test,
                                    test_names=test_names,
                                    exp_name=exp_name, saving_dir=saving_dir_, save_model=save_model,
                                    scoring=scoring, n_jobs=5, logger=logger)
                trainer.start()
                jobs.append(trainer)
            else:
                for meth in red_methods:
                    train_red, (val_red, test_red, test_intra_red) = red_dim(train_data_, y_tr, val_data_,
                                                                             test_data_, test_intra_data_,
                                                                             meth=meth,
                                                                             classif=(pb == "sex"),
                                                                             post_norm=(post_norm))
                    if args.cv: val_red = None
                    if args.test: train_red = None
                    saving_dir_ = os.path.join(saving_dir, preproc, model_name, pb.capitalize(), meth,
                                               'N_%i' % args.N_train)
                    test_names = ["%s_Intra-OpenBHB"%args.test_name, "%s_OpenBHB"%args.test_name]
                    trainer = MLTrainer(model(), deepcopy(hyperparams), train_red, y_tr,
                                        X_val=val_red, y_val=y_val,
                                        X_test=[test_intra_red, test_red],
                                        y_test=[y_test_intra, y_test],
                                        test_names=test_names,
                                        exp_name=exp_name, saving_dir=saving_dir_,
                                        scoring=scoring, n_jobs=5, logger=logger)
                    trainer.start()
                    jobs.append(trainer)
        for job in jobs: job.join()
