import os
import pickle
import logging
from dl_training.utils import get_chk_name
from dl_training.core import Base
from dl_training.datamanager import OpenBHBDataManager
from dl_training.self_supervision.sim_clr import SimCLR
from sklearn.preprocessing import LabelEncoder
from dl_training.history import History
from dl_training.training import BaseTrainer
from dl_training.transforms import *
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn


class BaseTester():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.num_classes, args, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net, args=self.args)
        self.logger = logging.getLogger("pynet")
        self.kwargs_test = dict()

        if self.args.pretrained_path and self.manager.number_of_folds > 1:
            self.logger.warning('Several folds found while a unique pretrained path is set!')

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("Test_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}".format(fold, epoch)
                model = Base(model=self.net, loss=self.loss,
                             metrics=self.args.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)
                model.testing(self.manager.get_dataloader(test=True, fold_index=fold).test,
                              with_visuals=False,
                              with_logit=self.args.with_logit,
                              predict=self.args.predict,
                              saving_dir=self.args.checkpoint_dir,
                              exp_name=exp_name,
                              standard_optim=getattr(self.net, 'std_optim', True),
                              **self.kwargs_test)
    
    def get_folds_to_test(self):
        if self.args.folds is not None and len(self.args.folds) > 0:
            folds = self.args.folds
        else:
            folds = list(range(self.args.nb_folds))
        return folds

    def get_epochs_to_test(self):
        if self.args.test_all_epochs:
            # Get all saved points and test them
            starting_epoch = self.args.start_from or self.args.nb_epochs_per_saving
            epochs_tested = [list(range(starting_epoch, self.args.nb_epochs,
                                        self.args.nb_epochs_per_saving)) + [
                                 self.args.nb_epochs - 1] for _ in range(self.args.nb_folds)]
        elif self.args.test_best_epoch:
            # Get the best point of each fold according to a certain metric (early stopping)
            metric = self.args.test_best_epoch
            h_val = History.load_from_dir(self.args.checkpoint_dir, "Validation_%s" % (self.args.exp_name or ""),
                                          self.args.nb_folds - 1, self.args.nb_epochs - 1)
            epochs_tested = h_val.get_best_epochs(metric, highest=True).reshape(-1, 1)
        else:
            # Get the last point and test it, for each fold
            epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.args.nb_folds)]

        return epochs_tested


class OpenBHBTester(BaseTester):
    """
    We perform 2 kind of tests:
    * Test-Intra where we test on the left-out test set with intra-site images
    * Test-Inter where we test on inter-site images (never-seen site)
    """
    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            tests = ["", "Intra_"]
            (residualizer, Zres) = (None, None)
            if self.args.db == "open_bhb" and self.manager.residualize is not None:
                (residualizer, Zres) = self.manager.fit_residualizer(["test", "test_intra", "train"], fold)
                if self.manager.residualize == "combat":
                    tests = ["Intra_"]
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                for t in tests:
                    if self.args.outfile_name is None:
                        outfile = "%s%s"%("Test_"+t, self.args.exp_name)
                    else:
                        outfile = "%s%s"%(t, self.args.outfile_name)
                    exp_name = outfile + "_fold{}_epoch{}".format(fold, epoch)
                    loader = self.manager.get_dataloader(test=(t==""),
                                                         test_intra=(t!=""),
                                                         fold_index=fold,
                                                         residualizer=residualizer,
                                                         Zres=Zres)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.args.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    model.testing(loader.test, with_visuals=False,
                                  with_logit=self.args.with_logit,
                                  predict=self.args.predict,
                                  saving_dir=self.args.checkpoint_dir,
                                  exp_name=exp_name,
                                  standard_optim=getattr(self.net, 'std_optim', True),
                                  **self.kwargs_test)

class NNRepresentationTester(BaseTester):
    """
    Test the representation of a given network by passing the training set and testing set through all the
    network's blocks and dumping the new vectors on disk (with eventually the labels to predict)
    CONVENTION:
        - we assume <network_name>_block%i exists for i=1..4
        - we assume to perform cross-validation with several pre-trained model (we use all the training set
          on the downstream task).
    """
    def __init__(self, args):
        self.args = args
        ## Several networks to test corresponding to a partial version of the whole network
        self.nets = [BaseTrainer.build_network(args.net+'_block%i'%i, args.num_classes, args, in_channels=1)
                     for i in range(4, 5)]
        ## Little hack: if we use the whole training set (no N_train_max), then we set the nb of folds to 1 and
        ## we test on several pre-trained models. Otherwise, the fold nb of the pre-trained model must be set and
        ## we make the nb of fine-tuning folds vary.
        true_nb_folds = args.nb_folds
        if args.N_train_max is None:
            args.nb_folds = 1
        else:
            assert args.folds != None, "If N_train_max is set, the --folds param must be set."
        self.manager = BaseTrainer.build_data_manager(args)
        if args.N_train_max is None:
            args.nb_folds = true_nb_folds
        ## Useless, just to avoid weird issues
        self.loss = BaseTrainer.build_loss(args.loss, net=self.nets[0], args=self.args)
        ## Usual logger and warning
        self.logger = logging.getLogger("pynet")
        if self.args.pretrained_path and self.args.nb_folds > 1:
            self.logger.warning('Several folds found while a unique pretrained path is set!')

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        finetuned_folds = [0] if self.args.N_train_max is None else list(range(self.args.nb_folds))
        # Passes the training/testing set through the encoder and uses scikit-learn to predict
        # the target (either logistic regression or ridge regression

        if self.args.cv:
            self.logger.warning("CROSS-VALIDATION USED DURING TESTING, EVENTUAL TESTING SET IS OMIT")

        for fold in folds_to_test: ## pre-trained models fold
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                for i, net in enumerate(self.nets, start=4):
                    for finetuned_fold in finetuned_folds: ## fine-tuning training fold
                        outfile = self.args.outfile_name or ("Test_" + self.args.exp_name)
                        if len(finetuned_folds) > 1:
                            exp_name = outfile + "_block{}_ModelFold{}_fold{}_epoch{}.pkl".format(i, fold, finetuned_fold, epoch)
                        else:
                            exp_name = outfile + "_block{}_fold{}_epoch{}.pkl".format(i, fold, epoch)
                        model_cls = Base
                        if self.args.model == "SimCLR":
                            if len(self.args.modalities) > 1:
                                model_cls = MultiModalSimCLR
                            else:
                                model_cls = SimCLR
                        elif self.args.model == "Genesis":
                            model_cls = Genesis
                        # re-init net each time
                        net = BaseTrainer.build_network(self.args.net+'_block%i'%i, self.args.num_classes, self.args, in_channels=1)

                        model = model_cls(model=net, loss=self.loss,
                                       pretrained=pretrained_path,
                                       use_cuda=self.args.cuda)

                        x_enc, y_train = (None, None) # Training set encoded
                        x_enc_val, y_val = (None, None) # Validation set encoded
                        xt_enc, y_test = (None, None) # Test set encoded
                        xt_intra_enc, y_test_intra = (None, None) # Extra test in case of OpenBHB/ClinicalDataset...

                        if self.args.model == "SimCLR":
                            # 1st: passes all the training data through the model
                            x_enc, y_train = model.features_avg_test(self.manager.get_dataloader(train=True,
                                                                                                 fold_index=finetuned_fold).train,
                                                                     M=int(self.args.test_param or 1))
                            # 2nd: passes all the testing data through the model
                            if self.args.cv:
                                xt_enc, y_test = model.features_avg_test(self.manager.get_dataloader(validation=True,
                                                                                                     fold_index=finetuned_fold).validation,
                                                                         M=int(self.args.test_param or 1))
                            else:
                                xt_enc, y_test = model.features_avg_test(self.manager.get_dataloader(test=True).test,
                                                                         M=int(self.args.test_param or 1))
                                if isinstance(self.manager, OpenBHBDataManager): # Add one extra test
                                    xt_intra_enc, y_test_intra = model.features_avg_test(
                                        self.manager.get_dataloader(test_intra=True).test, M=int(self.args.test_param or 1))

                                x_enc_val, y_val = model.features_avg_test(self.manager.get_dataloader(validation=True,
                                                                                                       fold_index=finetuned_fold).validation,
                                                                           M=int(self.args.test_param or 1))

                        else: # idem with MC Test
                            x_enc, y_train = model.MC_test(self.manager.get_dataloader(train=True, fold_index=finetuned_fold).train,
                                                           MC=int(self.args.test_param or 1))
                            if self.args.cv:
                                xt_enc, y_test = model.MC_test(self.manager.get_dataloader(validation=True,
                                                                                           fold_index=finetuned_fold).validation,
                                                               MC=int(self.args.test_param or 1))
                            else:
                                xt_enc, y_test = model.MC_test(self.manager.get_dataloader(test=True).test,
                                                               MC=int(self.args.test_param or 1))
                                if isinstance(self.manager, OpenBHBDataManager): # Add one extra test
                                    xt_intra_enc, y_test_intra = model.MC_test(self.manager.get_dataloader(test_intra=True).test,
                                                                               MC=int(self.args.test_param or 1))
                                x_enc_val, y_val = model.MC_test(self.manager.get_dataloader(validation=True,
                                                                                             fold_index=finetuned_fold).validation,
                                                                 MC=int(self.args.test_param or 1))
                        y_train = y_train[:, 0]
                        y_test = y_test[:, 0]
                        x_enc = np.mean(x_enc, axis=1) # mean over the sampled features f_i from the same input x
                        xt_enc = np.mean(xt_enc, axis=1)
                        if x_enc_val is not None and len(x_enc_val) > 1:
                            x_enc_val = np.mean(x_enc_val, axis=1)
                            y_val = y_val[:, 0]
                        if xt_intra_enc is not None:
                            xt_intra_enc = np.mean(xt_intra_enc, axis=1)
                            y_test_intra = y_test_intra[:, 0]

                        # 3rd: train scikit-learn model and save the results
                        label = self.args.labels
                        if label is None or len(label) > 1 or len({'age', 'sex', 'diagnosis', 'digit',
                                                                   'mnist_digit', 'stl10_class', 'site', 'label'} & set(label)) == 0:
                            raise ValueError("Please correct the label to predict (got {})".format(label))
                        label = label[0]

                        if x_enc_val is None or len(x_enc_val) == 0:
                            cv = 5
                            X_train, Y_train = np.array(x_enc).reshape(len(x_enc), -1), np.array(y_train).ravel()
                        else:
                            n_train, n_val = len(x_enc), len(x_enc_val)
                            cv = [(np.arange(n_train), n_train + np.arange(n_val))]
                            X_train, Y_train = np.concatenate((x_enc, x_enc_val)), \
                                               np.concatenate((y_train, y_val)).ravel()

                        if label in ['sex', 'diagnosis', 'digit', 'mnist_digit', 'stl10_class', 'site', 'label']:
                            scoring = "balanced_accuracy" if label in ['digit', 'mnist_digit', 'stl10_class', 'site',
                                                                       "label"] else "roc_auc"
                            if len(X_train) > 2e4:
                                model = LinearClassifier(batch_size=self.args.batch_size)
                                param_grid = {"weight_decay": [0, 1e-3, 1e-2, 1e-1]}
                            else:
                                model = LogisticRegression(solver="saga", max_iter=150)
                                param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 1e1]}
                            # Do NOT refit, otherwise can issue when N_train is set
                            best_model = clone(model)
                            model = GridSearchCV(model, cv=cv, param_grid=param_grid, scoring=scoring, n_jobs=5, refit=False)
                            #model = LogisticRegressionCV(cv=5, scoring=scoring, solver='liblinear', n_jobs=5)
                            model.fit(X_train, Y_train)
                            # Refit manually with the best params
                            best_params = model.cv_results_['params'][model.best_index_]
                            best_model.set_params(**best_params)
                            best_model.fit(np.array(x_enc).reshape(len(x_enc), -1), np.array(y_train).ravel())
                            y_pred = best_model.predict_proba(np.array(xt_enc).reshape(len(xt_enc), -1))
                            if xt_intra_enc is not None:
                                y_pred_intra = best_model.predict_proba(np.array(xt_intra_enc).reshape(len(xt_intra_enc), -1))
                            # Perform label mapping y to [0..N_cls-1]
                            le = LabelEncoder()
                            le.fit(np.array(y_train).flatten())
                            if y_test_intra is not None:
                                try:
                                    y_test_intra = le.transform(y_test_intra)
                                except ValueError as e:
                                    print("ERROR: No transform of intra-test labels: %s"%e, flush=True)
                            try:
                                y_test = le.transform(y_test)
                            except ValueError as e:
                                print("ERROR: No transform of inter-test: %s"%e, flush=True)
                        if label == 'age':
                            best_model = Ridge()
                            model = GridSearchCV(Ridge(), param_grid={'alpha': [1e-1, 1, 1e1, 1e2]}, cv=cv,
                                                 scoring='r2', refit=False)
                            model.fit(X_train, Y_train)
                            # Refit manually with the best params
                            best_params = model.cv_results_['params'][model.best_index_]
                            best_model.set_params(**best_params)
                            best_model.fit(np.array(x_enc).reshape(len(x_enc), -1), np.array(y_train).ravel())
                            y_pred = best_model.predict(np.array(xt_enc).reshape(len(xt_enc), -1))
                            if xt_intra_enc is not None:
                                y_pred_intra = best_model.predict(np.array(xt_intra_enc).reshape(len(xt_intra_enc), -1))

                        with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                            pickle.dump({"y": y_pred, "y_true": y_test}, f)

                        if xt_intra_enc is not None:
                            with open(os.path.join(self.args.checkpoint_dir, "Intra_"+exp_name), 'wb') as f:
                                pickle.dump({"y": y_pred_intra, "y_true": y_test_intra}, f)


class LinearClassifier(BaseEstimator, ClassifierMixin):
    """
        Implements linear classifier in a scikit-learn fashion trained with SGD on CUDA (with PyTorch).
        It is scalable and faster than sklearn.SGDClassifier runt on CPU.
        It implements a .fit(), .predict() and .predict_proba() method
    """
    def __init__(self, lr=0.1, batch_size=128, epochs=300, momentum=0.9, weight_decay=0.0, val_fraction=0.1, tol=1e-4):
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.val_fraction = val_fraction
        self.tol = tol
        self.display_freq = self.epochs // 10
        self.classifier = None

    class ArrayDataset(Dataset):
        def __init__(self, X, y=None):
            if y is not None:
                assert len(X) == len(y), "Wrong shape"
            self.X, self.y = X, y
        def __getitem__(self, i):
            if self.y is not None:
                return (self.X[i], self.y[i])
            return self.X[i]
        def __len__(self):
            return len(self.X)

    class LinearModel(nn.Module):
        """Linear classifier"""
        def __init__(self, feat_dim, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(feat_dim, num_classes)

        def forward(self, features):
            return self.fc(features)

    def predict(self, X):
        check_is_fitted(self)

        self.classifier.eval()
        loader = DataLoader(self.ArrayDataset(X), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
                out = self.classifier(x).detach().cpu().numpy()
                outputs.extend(out.argmax(axis=1))
        return np.array(outputs)

    def predict_proba(self, X):
        check_is_fitted(self)

        self.classifier.eval()
        loader = DataLoader(self.ArrayDataset(X), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.classifier(x)
            outputs.extend(torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy())
        return np.array(outputs)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, test_size=self.val_fraction)

        self.classes_ = unique_labels(y_tr)
        self.num_features = X_tr.shape[1]

        # build data loaders
        train_loader = DataLoader(self.ArrayDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.ArrayDataset(X_val, y_val), batch_size=self.batch_size)

        # build model and criterion
        self.classifier = self.LinearModel(self.num_features, num_classes=len(self.classes_))
        if torch.cuda.is_available():
            self.classifier = self.classifier.to('cuda')
        criterion = nn.CrossEntropyLoss()

        # build optimizer
        optimizer = torch.optim.SGD(self.classifier.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        # training routine
        losses_val = []
        stopping_criterion = None
        patience = 10
        acc_val = 0.0
        for epoch in range(1, self.epochs + 1):
            # train for one epoch
            loss, acc = self.train(train_loader, self.classifier, criterion, optimizer)
            loss_val, acc_val = self.validate(val_loader, self.classifier, criterion)
            scheduler.step(loss_val)
            losses_val.append(loss_val)
            if len(losses_val) > 2 * patience:
                stopping_criterion = np.max(np.abs(np.mean(losses_val[-patience:]) - losses_val[-patience:]))
                if stopping_criterion < self.tol: # early-stopping
                    break
        print("Validation Accuracy: %.3f" % acc_val, flush=True)
        losses_val = np.array(losses_val)
        if (np.max(np.abs(np.mean(losses_val[-patience:]) - losses_val[-patience:])) > self.tol):
            print("Warning: max iter reached without clear convergence", flush=True)
        return self

    def train(self, train_loader, classifier, criterion, optimizer):
        from pynet.metrics import accuracy
        """one epoch training"""
        classifier.train()
        losses = []
        top1 = []
        for idx, (features, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            # compute loss
            output = classifier(features)
            loss = criterion(output, labels)
            # update metric
            losses.append(loss.detach().cpu().numpy())
            acc1 = accuracy(output, labels)
            top1.append(acc1)
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return np.mean(losses), np.mean(top1)


    def validate(self, val_loader, classifier, criterion):
        from pynet.metrics import accuracy
        """validation"""
        classifier.eval()
        losses, top1 = [], []
        with torch.no_grad():
            for idx, (features, labels) in enumerate(val_loader):
                if torch.cuda.is_available():
                    features = features.cuda()
                    labels = labels.cuda()
                # forward
                output = classifier(features)
                loss = criterion(output, labels)

                # update metric
                losses.append(loss.detach().cpu().numpy())
                acc1 = accuracy(output, labels)
                top1.append(acc1)
        return np.mean(losses), np.mean(top1)


class SimCLRTester(BaseTester):
# For each sample x_i in the test set, it passes M times t(x_i) through the network f pre-trained for t ~ T
# M == self.args.test_param or 1 by default.

    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("SimCLRTest_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}.pkl".format(fold, epoch)
                kwargs_test = dict()
                if len(self.args.modalities) > 1:
                    model_cls = MultiModalSimCLR
                    if isinstance(self.net, FusionDenseNet):
                        kwargs_test["fusion"] = True
                else:
                    model_cls = SimCLR
                model = model_cls(model=self.net, loss=self.loss,
                                  metrics=self.args.metrics,
                                  pretrained=pretrained_path,
                                  use_cuda=self.args.cuda)
                y, y_true = model.features_avg_test(self.manager.get_dataloader(test=True).test,
                                                    M=int(self.args.test_param or 1), **kwargs_test)
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": y, "y_true": y_true}, f)


class BayesianTester(BaseTester):

    def run(self):
        MC = int(self.args.test_param or 10)
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        if self.args.cv:
            self.logger.warning("CROSS-VALIDATION USED DURING TESTING, EVENTUAL TESTING SET IS OMIT")
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                pretrained_path = self.args.pretrained_path or \
                                  os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name, fold, epoch))
                outfile = self.args.outfile_name or ("MCTest_" + self.args.exp_name)
                exp_name = outfile + "_fold{}_epoch{}.pkl".format(fold, epoch)
                model = Base(model=self.net, loss=self.loss,
                             metrics=self.args.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)
                if self.args.cv:
                    y, y_true = model.MC_test(self.manager.get_dataloader(validation=True,
                                                                          fold_index=fold).validation, MC=MC)
                else:
                    if isinstance(self.manager, OpenBHBDataManager):
                        y_intra, y_true_intra = model.MC_test(self.manager.get_dataloader(test_intra=True).test, MC=MC)
                        with open(os.path.join(self.args.checkpoint_dir, "Intra_"+exp_name), 'wb') as f:
                            pickle.dump({"y": y_intra, "y_true": y_true_intra}, f)

                    y, y_true = model.MC_test(self.manager.get_dataloader(test=True).test, MC=MC)
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": y, "y_true": y_true}, f)


class EnsemblingTester(BaseTester):
    def run(self, nb_rep=10):
        if self.args.pretrained_path is not None:
            raise ValueError('Unset <pretrained_path> to use the EnsemblingTester')
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            for epoch in epochs_tested[fold]:
                Y, Y_true = [], []
                for i in range(nb_rep):
                    pretrained_path = os.path.join(self.args.checkpoint_dir, get_chk_name(self.args.exp_name+
                                                                                          '_ensemble_%i'%(i+1), fold, epoch))
                    outfile = self.args.outfile_name or ("EnsembleTest_" + self.args.exp_name)
                    exp_name = outfile + "_fold{}_epoch{}.pkl".format(fold, epoch)
                    model = Base(model=self.net, loss=self.loss,
                                 metrics=self.args.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    y, y_true,_,_,_ = model.test(self.manager.get_dataloader(test=True).test)
                    Y.append(y)
                    Y_true.append(y_true)
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": np.array(Y).swapaxes(0,1), "y_true": np.array(Y_true).swapaxes(0,1)}, f)
