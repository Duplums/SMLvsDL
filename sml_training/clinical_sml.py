"""
We aim at giving learning curves of classical ML models on OpenBHB for:
* Train/Val/Test Split with 2 left-out studies
* VBM, Quasi-Raw and FSL pre-processing

"""

import numpy as np
import os, logging
import nibabel
from copy import deepcopy
import argparse
from datasets.clinical_multisites import SCZDataset, BipolarDataset, ASDDataset, SubSCZDataset, \
    SubBipolarDataset, SubASDDataset
from sml_training.age_sex_sml import residualize
import pandas as pd
from sml_training.sk_trainer import MLTrainer
from dl_training.datamanager import Zscore
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, RFE, f_classif, f_regression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# nb_training_samples = [100, 500, 1000, 3000, 5000]
# total_nb_folds = [5, 5, 3, 3, 3]

logger = logging.getLogger("sml_training")

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
        estim = SVC(kernel="linear", C=1, class_weight="balanced") if classif else SVR(kernel="linear")
        rfe = RFE(estim, n_features_to_select=nFeats, step=0.25)
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="Root directory to data")
    parser.add_argument("--saving_dir", required=True, type=str, help="Saving directory")
    parser.add_argument("--preproc", required=True, choices=["quasi_raw", "vbm"])
    parser.add_argument("--model", required=True, choices=["SVC", "LogisticRegression",
                                                           "ElasticNet", "RandomForestClassifier"])
    parser.add_argument("--pb", required=True, choices=["scz", "bipolar", "asd"])
    #parser.add_argument("--N_train", type=int, required=True)... To be continued
    parser.add_argument("--njobs", default=1, type=int)
    parser.add_argument("--folds", nargs='+', type=int)
    parser.add_argument("--nb_folds", type=int, required=True)
    parser.add_argument("--no_reduc", action="store_true")
    parser.add_argument("--N_train", type=int)
    parser.add_argument("--scaler", default="standard", choices=["standard", "zscore", "none"])
    parser.add_argument("--mask", default="reduced", choices=["std", "reduced"])
    parser.add_argument("--residualize", type=str, choices=["linear", "combat"])
    parser.add_argument("--post_norm", action="store_true")
    parser.add_argument("--nfeatures", type=int, default=784)
    parser.add_argument("--red_meth", nargs="+", choices=["UFS", "RFE", "GRP"])
    parser.add_argument("--folder", type=str, default="")
    parser.add_argument("--test_name", type=str, default="Test")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    root = args.root
    saving_dir = args.saving_dir

    scoring = "balanced_accuracy"

    if args.mask == "std":
        masks = {"vbm": np.load(os.path.join(root, "mask_open-bhb_vbm.npy")),
                 "quasi_raw": np.load(os.path.join(root, "mask_open-bhb_quasi_raw.npy"))}
    else:
        m_vbm = nibabel.load(os.path.join(args.root, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
        m_quasi_raw = nibabel.load(os.path.join(root, "mni_raw_brain-mask_1.5mm.nii.gz"))
        masks = {"vbm": m_vbm.get_data() != 0, "quasi_raw": m_quasi_raw.get_data() != 0}

    models = [SVC, LogisticRegression, RandomForestClassifier, SGDClassifier]

    post_norm = args.post_norm  # If True, apply standard scaling to data reduced

    N_train = args.N_train

    models_hyperparams = {"SVC": {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 3), "class_weight": ["balanced"]},
                          "RandomForestClassifier": {'n_estimators': [100, 150, 200],
                                                     'min_samples_split': [2, 5, 10],
                                                     'min_samples_leaf': [1, 2, 4],
                                                     'n_jobs': [6],
                                                     'class_weight': ['balanced']},
                          "LogisticRegression": {'C': 10. ** np.arange(-1, 3), 'solver': ['sag'], 'penalty': ['l2'],
                                                 'n_jobs': [6],
                                                 'fit_intercept': [True],
                                                 'class_weight': ["balanced"]},
                          "ElasticNet": {'alpha': 10. ** np.arange(-1, 2),
                                         'l1_ratio': [.1, .5, .9],
                                         'loss': ['log'],
                                         'penalty': ['elasticnet'],
                                         "class_weight": ["balanced"]}}

    all_datasets = {
        "scz": SCZDataset if N_train is None else SubSCZDataset,
        "bipolar": BipolarDataset if N_train is None else SubBipolarDataset,
        "asd": ASDDataset if N_train is None else SubASDDataset
    }

    logger = logging.getLogger("dl_training")

    red_methods = args.red_meth or ["UFS", "RFE", "GRP"]

    preproc = args.preproc
    label = "diagnosis"
    pb = args.pb
    if args.model == "ElasticNet":
        model = SGDClassifier
    else:
        model = eval(args.model)

    assert model in models

    scheme = "train_val_test"

    if args.scaler == "standard":
        scaler = StandardScaler
    elif args.scaler == "zscore":
        scaler = Zscore
    else:
        scaler = None

    hyperparams = models_hyperparams[args.model]
    mask = masks[args.preproc]

    dataset_cls = all_datasets[args.pb]

    val = dataset_cls(root, preproc=preproc, split="val", target=label)
    test = dataset_cls(root, preproc=preproc, split="test", target=label)
    test_intra = dataset_cls(root, preproc=preproc, split="test_intra", target=label)
    val_data, y_val = val.get_data(mask=mask, dtype=np.float32)
    test_data, y_test = test.get_data(mask=mask, dtype=np.float32)
    test_intra_data, y_test_intra = test_intra.get_data(mask=mask, dtype=np.float32)

    y_val = y_val.ravel().astype(np.int)
    y_test = y_test.ravel().astype(np.int)
    y_test_intra = y_test_intra.ravel().astype(np.int)

    folds = list(args.folds or range(args.nb_folds))
    folds_per_job = np.array_split(folds, np.ceil(len(folds)/args.njobs))
    # Parallelize across folds
    for folds_job_i in folds_per_job:
        jobs = []
        logger.info("Processing folds {f} - Model {m} - N={s} - Label {l} - Preproc {p}".format(f=folds_job_i,
                                                                                                m=args.model,
                                                                                                s=N_train or "(All)",
                                                                                                l=label,
                                                                                                p=preproc))
        for fold in folds_job_i:
            if N_train is None:
                train = dataset_cls(root, preproc=preproc, split="train", target=label) # fold is ignored currently
            else:
                train = dataset_cls(root, N_train_max=N_train, nb_folds=args.nb_folds, fold=fold,
                                    preproc=preproc, split="train", target=label)

            (train_data, y_tr) = train.get_data(mask=mask, dtype=np.float32)
            y_tr = y_tr.ravel().astype(np.int)
            # 0) Eventually Residualization with Linear Adjusted Reg. (NO DATA LEAKAGE on neither Age or Sex)
            if args.residualize is not None:
                design_cols = ["age", "sex", "site", "diagnosis"]
                df = pd.DataFrame(train.all_labels.astype(np.float32), columns=design_cols)
                df_val = pd.DataFrame(val.all_labels.astype(np.float32), columns=design_cols)
                df_test = pd.DataFrame(test.all_labels.astype(np.float32), columns=design_cols)
                df_test_intra = pd.DataFrame(test_intra.all_labels.astype(np.float32), columns=design_cols)
                categorical_cols = ["sex", "site", "diagnosis"]
                df[categorical_cols] = df[categorical_cols].astype(object)
                df_val[categorical_cols] = df_val[categorical_cols].astype(object)
                df_test[categorical_cols] = df_test[categorical_cols].astype(object)
                df_test_intra[categorical_cols] = df_test_intra[categorical_cols].astype(object)

                if args.residualize == "linear":
                    (train_data_, val_data_, test_data_, test_intra_data_) = residualize("site + age + sex",
                                                                                         "site + age + sex + diagnosis",
                                                                                         df, train_data,
                                                                                         (df_val, val_data),
                                                                                         (df_test, test_data),
                                                                                         (df_test_intra,
                                                                                          test_intra_data),
                                                                                         type=args.residualize)
                elif args.residualize == "combat":
                    (train_data_, val_data_, test_data_, test_intra_data_) = residualize("site + age + sex",
                                                                                         "site + age + sex + diagnosis",
                                                                                         df, train_data,
                                                                                         (df_val, val_data),
                                                                                         (df_test, test_data),
                                                                                         (df_test_intra,
                                                                                          test_intra_data),
                                                                                         type=args.residualize,
                                                                                         continuous_vars=["age"],
                                                                                         discrete_vars=["sex", "diagnosis"])
            # 1) Normalization
            if scaler is not None:
                if args.residualize is not None:
                    ss = scaler().fit(train_data_)
                    train_data_ = ss.transform(train_data_)
                    val_data_ = ss.transform(val_data_)
                    #if args.residualize != "combat":
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
            exp_name = '{m}_{pb}_{db}_fold{k}_epoch{e}.pkl'.format(m=model_name, pb="Dx",
                                                                   db=args.pb.upper()+"Dataset",
                                                                   k=fold, e=100)
            if args.no_reduc:
                if N_train is None:
                    saving_dir_ = os.path.join(saving_dir, preproc, model_name, "Dx")
                else:
                    saving_dir_ = os.path.join(saving_dir, preproc, model_name, "Dx", "N_%i"%N_train)
                if args.test: train_data_ = None
                X_tests = [test_intra_data_, test_data_] #if args.residualize != "combat" else [test_intra_data_]
                y_tests = [y_test_intra, y_test] #if args.residualize != "combat" else [y_test_intra]
                test_names = ["%s_Intra"%args.test_name, "%s"%args.test_name] #if args.residualize != "combat" \
                    #else ["%s_Intra"%args.test_name]
                trainer = MLTrainer(model(), deepcopy(hyperparams), train_data_, y_tr,
                                    X_val=val_data_, y_val=y_val,
                                    X_tests=X_tests,
                                    y_tests=y_tests,
                                    test_names=test_names,
                                    exp_name=exp_name, saving_dir=saving_dir_, save_model=True,
                                    scoring=scoring, n_jobs=5, logger=logger)
                trainer.start()
                jobs.append(trainer)
            else:
                for meth in red_methods:
                    if args.residualize != "combat":
                        train_red, (val_red, test_red, test_intra_red) = red_dim(train_data_, y_tr, val_data_,
                                                                                 test_data_, test_intra_data_, meth=meth,
                                                                                 classif=True,
                                                                                 post_norm=post_norm,
                                                                                 nFeats=args.nfeatures)
                    else:
                        train_red, (val_red, test_red, test_intra_red) = red_dim(train_data_, y_tr, val_data_,
                                                                       test_data_, test_intra_data_, meth=meth,
                                                                       classif=True,
                                                                       post_norm=post_norm,
                                                                       nFeats=args.nfeatures)
                    if N_train is None:
                        saving_dir_ = os.path.join(saving_dir, preproc, model_name, "Dx", meth)
                    else:
                        saving_dir_ = os.path.join(saving_dir, preproc, model_name, "Dx", meth, "N_%i"%N_train)
                    if args.test: train_red = None
                    X_tests = [test_intra_red, test_red]# if args.residualize != "combat" else [test_intra_red]
                    y_tests = [y_test_intra, y_test] #if args.residualize != "combat" else [y_test_intra]
                    test_names = ["%s_Intra"%args.test_name, "%s"%args.test_name]# if args.residualize != "combat" \
                        #else ["%s_Intra"%args.test_name]
                    trainer = MLTrainer(model(), deepcopy(hyperparams), train_red, y_tr,
                                        X_val=val_red, y_val=y_val,
                                        X_tests=X_tests,
                                        y_tests=y_tests,
                                        test_names=test_names,
                                        exp_name=exp_name, saving_dir=saving_dir_,
                                        scoring=scoring, n_jobs=5, logger=logger)
                    trainer.start()
                    jobs.append(trainer)
        for job in jobs: job.join()



