import os, pickle, subprocess
from threading import Thread
import numpy as np
from datasets.open_bhb import OpenBHB
from sklearn.model_selection import GridSearchCV
from sklearn.base import is_classifier, is_regressor, clone

class OpenBHBMLTrainer(Thread):
    """
    A convenient worker specially adapted to perform ML on OpenBHB with scikit-learn. It can be executed in
    standalone fashion. The methods start()/join() must be favored to run the worker.
    """
    def __init__(self, model, hyperparams, training_dataset, testing_dataset, train_indices=None, mask=None,
                 exp_name=None, saving_dir=None, scoring=None, scaler=None, n_jobs=1, logger=None, **kwargs):
        """
        :param model: a scikit-learn model
        :param hyperparams: hyper-parameters over which Grid-Search 3-fold Cross-Validation is performed with
                scikit-learn
        :param training_dataset/testing_dataset: OpenBHB datasets used for Train/Test.
        :param train_indices (Optional): list of indices to give to <get_data> from OpenBHB (only for training)
        :param mask (Optional): a binary mask to give to <get_data> from OpenBHB
        :param exp_name: str, the results will be saved in <exp_name>
        :param saving_dir: str, path to the results (if it does not exist, it is created)
        :param scoring: scoring fn to give to scikit-learn <GridSearchCV> to perform grid-search
        :param scaler: a scikit-learn Scaler to transform train/test data
        :param n_jobs: number of jobs to perform grid-search over set of hyper-parameters
        :param logger: python Logger to use to write the training/test results (convenient for debugging)
        """
        super().__init__(**kwargs)

        assert isinstance(training_dataset, OpenBHB) and isinstance(testing_dataset, OpenBHB), \
            "Datasets must be OpenBHB"

        assert (is_classifier(model) or is_regressor(model)), "Model must be a scikit-learn classifier or regressor"

        self.model = model
        self.hyperparams = hyperparams
        self.training_dataset = training_dataset
        self.test_dataset = testing_dataset
        self.train_indices = train_indices
        self.mask = mask
        self.scoring = scoring
        self.scaler = scaler
        self.saving_dir = saving_dir
        self.exp_name = exp_name
        self.n_jobs = n_jobs
        self.logger = logger

    def run(self):
        # Loads the data in memory
        (X_train, y_train) = self.training_dataset.get_data(self.train_indices, mask=self.mask)
        if self.logger is not None: self.logger.info("Data loaded.")
        self.model_cv = GridSearchCV(self.model, self.hyperparams, n_jobs=self.n_jobs, scoring=self.scoring, cv=3)
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)

        # Performs Grid-Search with n_jobs workers
        self.model_cv.fit(X_train, y_train)

        # Reports the results on train
        if self.logger is not None:
            exp = os.path.join(self.saving_dir or '', self.exp_name or '{} {}'.format(self.model.__str__,
                                                                                      self.training_dataset.__str__))

            self.logger.info("{}: Best score/params on Train: {} / {}".format(exp, self.model_cv.best_score_,
                                                                              self.model_cv.best_params_))
        # Free the memory as soon as possible
        del (X_train)
        (X_test, y_test) = self.test_dataset.get_data(mask=self.mask)
        if self.scaler is not None:
            X_test = self.scaler.fit_transform(X_test)
        y_pred = self.model_cv.predict(X_test)

        # Reports the results on test
        if self.logger is not None:
            exp = os.path.join(self.saving_dir or '', self.exp_name or '{} {}'.format(self.model.__str__,
                                                                                      self.test_dataset.__str__))
            self.logger.info("{}: Best score on Test: {}".format(exp, self.model_cv.score(X_test, y_test)))

        ## Saves the results on disk
        file_name = self.exp_name or "Test_{}_{}.pkl".format(self.model.__str__, self.training_dataset.__str__)
        if self.saving_dir is not None:
            if not os.path.isdir(self.saving_dir):
                # create the directory
                subprocess.check_call(['mkdir', '-p', self.saving_dir])
            file_name = os.path.join(self.saving_dir, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump({'y_pred': y_pred, 'y_true': y_test}, f, protocol=4)
        # saves the model in a distinct file
        file_name = self.exp_name or "Test_{}_{}.pkl".format(self.model.__str__, self.training_dataset.__str__)
        file_name = os.path.join(self.saving_dir, "Model_"+file_name)
        with open(file_name, 'wb') as f:
            pickle.dump({'model': self.model_cv}, f, protocol=4)


class MLTester(Thread):
    """
       A convenient worker specially adapted to test ML on OpenBHB with scikit-learn. It can be executed in
       standalone fashion. The methods start()/join() must be favored to run the worker.
       """

    def __init__(self, model, X_test, y_test,  exp_name=None, saving_dir=None,
                 scaler=None, logger=None, **kwargs):
        """
        :param model: a scikit-learn model can implement predict()
        :param X_test, y_test
        :param mask (Optional): a binary mask to give to <get_data> from OpenBHB
        :param exp_name: str, the results will be saved in <exp_name>
        :param saving_dir: str, path to the results (if it does not exist, it is created)
        :param scoring: scoring fn to give to scikit-learn <GridSearchCV> to perform grid-search
        :param scaler: a scikit-learn Scaler to transform test data
        :param logger: python Logger to use to write the training/test results (convenient for debugging)
        """
        super().__init__(**kwargs)

        assert hasattr(model, "predict"), "Model must implement predict()"

        self.model = model
        self.X_test, self.y_test = X_test, y_test
        self.scaler = scaler
        self.saving_dir = saving_dir
        self.exp_name = exp_name
        self.logger = logger

    def run(self):
        X_test, y_test = self.X_test.copy(), self.y_test.copy()
        if self.scaler is not None:
            X_test = self.scaler.fit_transform(X_test)
        y_pred = self.model.predict(X_test)

        # Reports the results on test
        if self.logger is not None:
            exp = os.path.join(self.saving_dir or '', self.exp_name or '{}'.format(self.model.__str__))
            self.logger.info("{}: Best score on Test: {}".format(exp, self.model.score(X_test, y_test)))

        ## Saves the results on disk
        file_name = self.exp_name or "Test_{}.pkl".format(self.model.__str__)
        if self.saving_dir is not None:
            if not os.path.isdir(self.saving_dir):
                # create the directory
                subprocess.check_call(['mkdir', '-p', self.saving_dir])
            file_name = os.path.join(self.saving_dir, file_name)
        if os.path.isfile(file_name):
            raise ValueError("File %s already exists ! Aborting...")

        with open(file_name, 'wb') as f:
            pickle.dump({'y_pred': y_pred, 'y_true': y_test}, f, protocol=4)


class MLTrainer(Thread):
    """
    A convenient worker specially adapted to perform ML with scikit-learn. It can be executed in
    standalone fashion. The methods start()/join() must be favored to run the worker.
    """
    def __init__(self, model, hyperparams, X_train, y_train, X_val=None, y_val=None, X_tests=None, y_tests=None,
                 test_names=None, exp_name=None, saving_dir=None, save_model=True, scoring=None, n_jobs=1,
                 logger=None, **kwargs):
        """
        :param model: a scikit-learn model
        :param hyperparams: hyper-parameters over which Grid-Search 3-fold Cross-Validation is performed with
                scikit-learn
        :param X_train: np.array for training. If None, it will try to load the last checkpoint and eventually test the
                model.
        :param X_tests (optional): list of testing np.array
        :param y_tests (optional): list of testing np.array target labels
        :param test_names (Optional): list of str to be concatenated to <exp_name> for dumping testing results.
                We assume len(test_names) == len(X_tests)
        :param exp_name: str, the results will be saved in <exp_name>
        :param saving_dir: str, path to the results (if it does not exist, it is created)
        :param save_model: boolean, whether the sklearn model is saved after training or not
        :param scoring: scoring fn to give to scikit-learn <GridSearchCV> to perform grid-search
        :param n_jobs: number of jobs to perform grid-search over set of hyper-parameters
        :param logger: python Logger to use to write the training/test results (convenient for debugging)
        """
        super().__init__(**kwargs)
        self.logger = logger
        self.last_checkpoint = None # Flag to indicate if we directly load the last checkpoint
        self.exp_name = exp_name
        self.model = model
        self.saving_dir = saving_dir or ""
        self.hyperparams = hyperparams
        self.scoring = scoring
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_tests, self.y_tests = X_tests, y_tests
        self.test_names = test_names
        self.n_jobs = n_jobs
        self.save_model = save_model
        self.last_checkpoint = None

        if X_train is None:
            file_name = self.exp_name or "Test_{}.pkl".format(self.model.__str__)
            self.last_checkpoint = os.path.join(self.saving_dir, "Model_" + file_name)
            if self.logger is not None:
                self.logger.warning("No X_train given, the last checkpoint to be loaded will be at %s"%
                                    self.last_checkpoint)
        else:
            assert isinstance(X_train, np.ndarray)

        assert (is_classifier(model) or is_regressor(model)), "Model must be a scikit-learn classifier or regressor"

        if X_tests is not None:
            assert y_tests is not None and test_names is not None, "<y_test> and <test_names> must be filled !"
            assert len(y_tests) == len(X_tests) == len(test_names)
            for (X_test, y_test) in zip(X_tests, y_tests):
                assert len(X_test) == len(y_test), "Incorrect dimension for X_test or y_test ({} != {})".\
                    format(X_test.shape, np.array(y_test).shape)
        if X_val is not None:
            assert y_val is not None and len(y_val) == len(X_val)


    def run(self):
        # Performs Grid-Search with n_jobs workers
        if self.last_checkpoint is None:
            if self.X_val is not None:
                n_train, n_val = len(self.X_train), len(self.X_val)
                self.model_cv = GridSearchCV(self.model, self.hyperparams, n_jobs=self.n_jobs, scoring=self.scoring,
                                             cv=[(np.arange(n_train), n_train+np.arange(n_val))], refit=False)
                self.model_cv.fit(np.concatenate((self.X_train, self.X_val)),
                                  np.concatenate((self.y_train, self.y_val)))
                best_fold, cv_results = self.model_cv.best_index_, self.model_cv.cv_results_
                best_score, best_params = cv_results['split0_test_score'][best_fold], cv_results['params'][best_fold]
                self.model_cv = clone(self.model).set_params(**best_params)
                # Refit by hand the model with the best params found only on training set
                self.model_cv.fit(self.X_train, self.y_train)
            else:
                self.model_cv = GridSearchCV(self.model, self.hyperparams, n_jobs=self.n_jobs, scoring=self.scoring, cv=3)
                self.model_cv.fit(self.X_train, self.y_train)
                best_score, best_params = self.model_cv.best_score_, self.model_cv.best_params_
            # Reports the results on train
            if self.logger is not None:
                exp = os.path.join(self.saving_dir or '', self.exp_name or '{}'.format(self.model.__str__, ))
                self.logger.info("{}: Best score/params on Train: {} / {}".format(exp, best_score, best_params))
        else:
            try:
                self.model_cv = MLTrainer.get_pickle(self.last_checkpoint).get("model")
            except BaseException as e:
                self.logger.error("Impossible to load %s: %s"%(self.last_checkpoint, e))
                return

        file_name = self.exp_name or "Test_{}.pkl".format(self.model.__str__)
        file_name = os.path.join(self.saving_dir, "Model_" + file_name)
        if self.last_checkpoint is None and self.save_model:
            MLTrainer.save({'model': self.model_cv}, file_name)

        if self.X_tests is not None:
            for (X_test, y_test, test_name) in zip(self.X_tests, self.y_tests, self.test_names):
                y_pred = self.model_cv.predict(X_test)
                kwargs = dict()
                try:
                    if hasattr(self.model_cv, "predict_proba"):
                        kwargs["y_pred_proba"] = self.model_cv.predict_proba(X_test)
                    if hasattr(self.model_cv, "decision_function"):
                        kwargs["decision_function"] = self.model_cv.decision_function(X_test)
                except BaseException as e:
                    if self.logger is not None:
                        self.logger.error(str(e))
                # Reports the results on test
                if self.logger is not None:
                    exp = os.path.join(self.saving_dir or '', '{} {}'.format(test_name, self.exp_name or self.model.__str__))
                    self.logger.info("{}: Best score on {}: {}".format(exp, test_name, self.model_cv.score(X_test, y_test)))

                ## Saves the results on disk
                file_name =  "{}_{}".format(test_name, self.exp_name or (self.model.__str__+'.pkl'))
                file_name = os.path.join(self.saving_dir, file_name)
                MLTrainer.save({'y_pred': y_pred, 'y_true': y_test, **kwargs}, file_name)

    @staticmethod
    def save(obj, file):
        dir_path = os.path.dirname(file)
        if dir_path != '' and not os.path.isdir(dir_path):
            # create the directory
            subprocess.check_call(['mkdir', '-p', dir_path])
        with open(file, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

    @staticmethod
    def get_pickle(path):
        import pickle
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj

