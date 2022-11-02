from datasets.open_bhb import OpenBHB, SubOpenBHB, Callable, List
from datasets.bhb_10k import BHB
from datasets.clinical_multisites import SCZDataset, BipolarDataset, ASDDataset, SubSCZDataset, \
    SubBipolarDataset, SubASDDataset
from dl_training.self_supervision.sim_clr import SimCLROpenBHB, SimCLRSubOpenBHB
from dl_training.transforms import Padding, Crop, Normalize, Standardize
from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
import torch, logging
from collections import namedtuple
from dl_training.preprocessing.combat import CombatModel
import numpy as np
import pandas as pd

SetItem = namedtuple("SetItem", ["test", "train", "validation"], defaults=(None,) * 3)
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels"])

class Zscore(TransformerMixin, BaseEstimator):
    """
    Standardization per sample
    """
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit_transform(self, X, y=None, **kwargs):
        return self.std * (X - np.mean(X, axis=1)[:, np.newaxis]) / \
               (np.std(X, axis=1)[:, np.newaxis] + self.eps) + self.mean

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.fit_transform(X)


class StandardScalerBiased(StandardScaler):
    def transform(self, X, copy=None):
        self.fit(X)
        return super().transform(X, copy)


class OpenBHBDataManager:

    def __init__(self, root: str, preproc: str, labels: List[str]=None, sampler: str="random",
                 batch_size: int=1, number_of_folds: int=None, N_train_max: int=None,
                 residualize: str=None, mask = None, model: str=None, device:str="cuda", **dataloader_kwargs):
        assert model in [None, "SimCLR", "base"], "Unknown model: %s"%model
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'"%sampler
        assert residualize in [None, "linear", "combat"], "Unknown residualizer %s"%residualize
        self.logger = logging.getLogger("SMLvsDL")
        self.dataset = dict()
        self.labels = labels or []
        self.residualize = residualize
        self.mask = mask
        # attributes for ComBat residualization
        self.discrete_vars = ["sex"]
        self.continuous_vars = ["age"]
        # Linear Adj. (on site) residualization attributes
        self.formula_res, self.formula_full = "site", "site + age + sex"
        scheme = "train_val_test"
        self.scheme = scheme

        input_transforms = self.get_input_transforms(preproc)
                
        if N_train_max is not None:
            self.logger.info("Automatic stratification on Age+Sex+Site")
            dataset_cls = SubOpenBHB if model != "SimCLR" else SimCLRSubOpenBHB
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, scheme=scheme,
                                                 split="train", stratify=True,
                                                 transforms=input_transforms,
                                                 N_train_max=N_train_max, nb_folds=number_of_folds,
                                                 fold=fold, target=labels)
                                     for fold in range(number_of_folds)]
            self.dataset["validation"] = [dataset_cls(root, preproc=preproc, scheme=scheme, split="val",
                                                      transforms=input_transforms, target=labels)
                                          for _ in range(number_of_folds)]
            self.number_of_folds = number_of_folds
        else:
            kwargs_dataset = dict()
            dataset_cls = OpenBHB if model != "SimCLR" else SimCLROpenBHB
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, scheme=scheme, split="train",
                                                 transforms=input_transforms, target=labels,
                                                 **kwargs_dataset)]
            self.dataset["validation"] = [dataset_cls(root, preproc=preproc, scheme=scheme, split="val",
                                                      transforms=input_transforms, target=labels,
                                                      **kwargs_dataset)]
            self.number_of_folds = 1
        dataset_cls = OpenBHB if model != "SimCLR" else SimCLROpenBHB
        self.dataset["test"] = dataset_cls(root, preproc=preproc, scheme=scheme, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, scheme=scheme, split="test_intra",
                                                 transforms=input_transforms, target=labels)
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

    @staticmethod
    def collate_fn(list_samples):
        """ After fetching a list of samples using the indices from sampler,
              the function passed as the collate_fn argument is used to collate lists
              of samples into batches.

              A custom collate_fn is used here to apply the transformations.

              See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
              """
        data = dict(outputs=None) # compliant with DataManager <collate_fn>
        data["inputs"] = torch.stack([torch.from_numpy(sample[0]) for sample in list_samples], dim=0).float()
        data["labels"] = torch.stack([torch.tensor(sample[1]) for sample in list_samples], dim=0).squeeze().float()
        return DataItem(**data)

    def fit_residualizer(self, splits=None, fold_index=0):
        """
        :param splits: list of splits, must be a train, validation or tests split the Residualizer will be applied on
                (required only for linear residualization)
        :param fold_index: the training fold index used to fit the Residualizer
        :return: a tuple (Residualizer, dict(split: design matrix for this split)
        """
        assert self.residualize in ["linear", "combat"], "Unknown residualization: %s"%type
        if self.residualize == "linear":
            assert set(splits) <= {"train", "validation", "test", "test_intra"}
        ## Fits a Residualizer on the training data
        train_df = self.dataset["train"][fold_index].all_labels.copy()
        train_data, _ = self.dataset["train"][fold_index].get_data(mask=self.mask)
        train_data = train_data.reshape(len(train_data), -1)

        test_dfs = {
            k: self.dataset[k][fold_index].all_labels.copy() if isinstance(self.dataset[k], list) else self.dataset[
                k].all_labels.copy() for k in splits}
        train_index = np.arange(len(train_df))
        offsets = np.cumsum([len(test_dfs[k]) for k in splits])
        offsets = offsets - offsets[0] + len(train_df)
        test_indexes = {k: offset + np.arange(len(test_dfs[k])) for k, offset in zip(splits, offsets)}
        all_df = pd.concat([train_df] + [test_dfs[k] for k in splits], ignore_index=True)

        all_df[self.discrete_vars+["site"]] = all_df[self.discrete_vars+["site"]].astype(object)

        if self.residualize == "combat":
            residualizer = CombatModel()
            residualizer.fit(train_data, all_df[["site"]].values[train_index],
                             discrete_covariates=all_df[self.discrete_vars].values[train_index],
                             continuous_covariates=all_df[self.continuous_vars].values[train_index])
            return residualizer, None

        elif self.residualize == "linear":
            from mulm.residualizer import Residualizer
            residualizer = Residualizer(data=all_df, formula_res=self.formula_res, formula_full=self.formula_full)
            Zres = residualizer.get_design_mat(all_df)
            residualizer.fit(train_data, Zres[train_index])
            del(train_data)  # free memory asap
            return residualizer, {d: (Zres[test_indexes[d]] if d != "train" else Zres[train_index]) for d in splits }

        raise ValueError("Unknown type")

    def get_dataloader(self, train=False, validation=False,
                       test=False, test_intra=False, fold_index=None,
                       residualizer=None, Zres=None):

        assert test + test_intra <= 1, "Only one tests accepted"
        _test, _train, _validation, sampler = (None, None, None, None)

        tests_to_return = []
        if validation: tests_to_return.append("validation")
        if test: tests_to_return.append("test")
        if test_intra: tests_to_return.append("test_intra")

        if self.residualize is not None:
            if residualizer is None:
                (residualizer, Zres) = self.fit_residualizer(tests_to_return+['train'], fold_index)

        test_loaders = dict()
        for t in tests_to_return:
            dataset = self.dataset[t] if t != "validation" else self.dataset[t][fold_index]
            if self.residualize == "linear":
                dataset = dataset.transform(residualizer, Zres[t], mask=self.mask)
            elif self.residualize == "combat":
                labels = dataset.all_labels.copy()
                dataset = dataset.transform(residualizer, labels[["site"]].astype(object), mask=self.mask,
                                            discrete_covariates=labels[self.discrete_vars],
                                            continuous_covariates=labels[self.continuous_vars])
            test_loaders[t] = DataLoader(dataset, batch_size=self.batch_size,
                                         collate_fn=OpenBHBDataManager.collate_fn,
                                         **self.dataloader_kwargs)
        if "test_intra" in test_loaders:
            assert "test" not in test_loaders
            test_loaders["test"] = test_loaders.pop("test_intra")
        if train:
            if self.sampler == "random":
                sampler = RandomSampler(self.dataset["train"][fold_index])
            elif self.sampler == "sequential":
                sampler = SequentialSampler(self.dataset["train"][fold_index])
            dataset = self.dataset["train"][fold_index]
            if self.residualize == "linear":
                dataset = dataset.transform(residualizer, Zres["train"], mask=self.mask)
            elif self.residualize == "combat":
                labels = dataset.all_labels.copy()
                dataset = dataset.transform(residualizer, labels[["site"]].astype(object),
                                            discrete_covariates=labels[self.discrete_vars],
                                            continuous_covariates=labels[self.continuous_vars], mask=self.mask)
            _train = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler,
                collate_fn=OpenBHBDataManager.collate_fn,
                **self.dataloader_kwargs)

        return SetItem(train=_train, **test_loaders)

    @staticmethod
    def get_input_transforms(preproc, model = None):
        if preproc in ["vbm", "quasi_raw"]:
            # Input size 121 x 145 x 121
            input_transforms = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),  Normalize()])
            if model == "SimCLR":
                from dl_training.self_supervision.sim_clr import DA_Module
                input_transforms.transforms.append(DA_Module())
        else:
            raise ValueError("Unknown preproc: %s"%preproc)
        return input_transforms

    def get_nb_folds(self):
        return self.number_of_folds

class BHBDataManager(OpenBHBDataManager):

    def __init__(self, root: str, preproc: str, labels: List[str]=None, sampler: str="random",
                 batch_size: int=1, number_of_folds: int=None, N_train_max: int=None, residualize: bool=False,
                 mask = None, model:str=None, device:str="cuda", scheme: str="train_val_test",
                 **dataloader_kwargs):

        assert model in [None, "base"], "Unknown model: %s"%model
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'"%sampler
        assert scheme == "train_val_test", "Scheme %s not implemented yet"%scheme
        assert N_train_max is None, "Sub-sampling BHB not implemented yet"

        self.logger = logging.getLogger("SMLvsDL")
        self.dataset = dict()
        self.labels = labels or []
        self.residualize = residualize
        self.mask = mask
        self.number_of_folds = number_of_folds
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs
        self.scheme = scheme

        # attributes for ComBat residualization
        self.discrete_vars = ["sex"]
        self.continuous_vars = ["age"]
        # Linear Adj. (on site) residualization attributes
        self.formula_res, self.formula_full = "site", "site + age + sex"

        input_transforms = self.get_input_transforms(preproc, model)

        self.dataset["train"] = [BHB(root, preproc=preproc, scheme=self.scheme, split="train",
                                         transforms=input_transforms, target=labels)
                                 for _ in range(self.number_of_folds)]
        self.dataset["validation"] = [BHB(root, preproc=preproc, scheme=self.scheme, split="val",
                                              transforms=input_transforms, target=labels)
                                      for _ in range(self.number_of_folds)]
        self.dataset["test"] = BHB(root, preproc=preproc, scheme=self.scheme, split="test",
                                       transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = BHB(root, preproc=preproc, scheme=self.scheme, split="test_intra",
                                         transforms=input_transforms, target=labels)


class ClinicalDataManager(OpenBHBDataManager):

    def __init__(self, root: str, preproc: str, db: str, labels: List[str]=None, sampler: str="random",
                 batch_size: int=1, number_of_folds: int=None, N_train_max: int=None, residualize: str=None,
                 mask = None, device:str="cuda", **dataloader_kwargs):

        assert db in ["scz", "bipolar", "asd"], "Unknown db: %s"%db
        assert sampler in ["random", "sequential"], "Unknown sampler '%s'"%sampler
        assert residualize in [None, "linear", "combat"], "Unkown residulalizer %s"%residualize

        self.logger = logging.getLogger("SMLvsDL")
        self.dataset = dict()
        self.labels = labels or []
        self.residualize = residualize
        self.mask = mask
        self.number_of_folds = number_of_folds
        self.sampler = sampler
        self.batch_size = batch_size
        self.device = device
        self.dataloader_kwargs = dataloader_kwargs

        # ComBat residualization (on site) attributes
        self.discrete_vars = ["sex", "diagnosis"]
        self.continuous_vars = ["age"]
        # Linear Adj. (on site) residualization attributes
        self.formula_res, self.formula_full = "site + age + sex", "site + age + sex + diagnosis"

        input_transforms = self.get_input_transforms(preproc)

        dataset_cls = None
        if db == "scz":
            dataset_cls = SCZDataset if N_train_max is None else SubSCZDataset
        elif db == "bipolar":
            dataset_cls = BipolarDataset if N_train_max is None else SubBipolarDataset
        elif db == "asd":
            dataset_cls = ASDDataset if N_train_max is None else SubASDDataset

        if N_train_max is None:
            self.dataset["train"] = [dataset_cls(root, preproc=preproc, split="train",
                                                 transforms=input_transforms, target=labels)
                                     for _ in range(self.number_of_folds)]
        else:
            self.dataset["train"] = [dataset_cls(root, N_train_max=N_train_max, fold=f,
                                                 nb_folds=self.number_of_folds, preproc=preproc, split="train",
                                                 transforms=input_transforms, target=labels)
                                     for f in range(self.number_of_folds)]

        self.dataset["validation"] = [dataset_cls(root, preproc=preproc, split="val",
                                                  transforms=input_transforms, target=labels)
                                      for _ in range(self.number_of_folds)]
        self.dataset["test"] = dataset_cls(root, preproc=preproc, split="test",
                                           transforms=input_transforms, target=labels)
        self.dataset["test_intra"] = dataset_cls(root, preproc=preproc, split="test_intra",
                                                 transforms=input_transforms, target=labels)




