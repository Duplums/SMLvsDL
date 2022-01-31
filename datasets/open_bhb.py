from torch.utils.data.dataset import Dataset
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import os, pickle
import pandas as pd
import numpy as np
import bisect, scipy
from typing import Callable, Any, List, Type, Iterable

class OpenBHB(Dataset):
    """
        OpenBHB Dataset written in a torchvision-like manner. It is memory-efficient, taking advantage of
        memory-mapping implemented with NumPy.  It comes with 2 differents schemes:
            - Train/Validation/(Test Intra-Site+ Inter-Site) split
        ... 2 pre-processings:
            - Quasi-Raw
            - VBM
        ... And 2 differents tasks:
            - Age prediction (regression)
            - Sex prediction (classification)
        ... With meta-data:
            - unique identifier across pre-processing and split (participant_id, session, run, study)
            - TIV + ROI measures based on Neuromorphometrics atlas
    Attributes:
          * target, list[int]: labels to predict
          * target_mapping: dict(int: str): each site number is associated to its original name
          * all_labels, pd.DataFrame: all labels stored in a pandas DataFrame containing ["age", "sex", "site"]
          * shape, tuple: shape of the data
          * metadata: pd DataFrame: TIV + ROI measures extracted for each image
          * id: pandas DataFrame, each row contains a unique identifier for an image

    """
    def __init__(self, root: str, preproc: str='vbm', scheme: str='train_val_test', target: [str, List[str]]='age',
                 split: str='train', fold: int=None, transforms: Callable[[np.ndarray], np.ndarray]=None,
                 target_transforms: Callable[[int, float], Any]=None, load_data: bool=False):
        """
        :param root: str, path to the root directory containing the different .npy and .csv files
        :param preproc: str, must be either VBM ('vbm'), Quasi-Raw ('quasi_raw') or FreeSurfer ('fs')
        :param scheme: str, must be either 5-fold CV ('cv') or Train/Val/Test ('train_val_test')
        :param target: str or [str], either 'age', 'sex' or 'site'.
        :param split: str, either 'train', 'val', 'test' (inter) or 'test_intra' for 'train_val_test' scheme
        :param fold: int, specified the fold to use (only for CV scheme, it is ignored otherwise). If not given,
        take the fold 0.
        :param transforms (callable, optional): A function/transform that takes in
            a 3D MRI image and returns a transformed version.
        :param target_transforms (callable, optional): A function/transform that takes in
            a target and returns a transformed version.
        :param load_data (bool, optional): If True, loads all the data in memory
               --> WARNING: it can be VERY time/memory-consuming
        """
        if isinstance(target, str):
            target = [target]
        assert preproc in ['vbm', 'quasi_raw'], "Unknown preproc: %s"%preproc
        assert scheme in ['cv', 'train_val_test'], "Unknown scheme: %s"%scheme
        assert set(target) <= {'age', 'sex', 'site'}, "Unknown target: %s"%target
        assert split in ['train', 'val', 'test', 'test_intra', 'validation'], "Unknown split: %s"%split
        assert fold in [None, 0, 1, 2, 3, 4], "Fold should be 0<=f<5"

        if scheme == 'cv' and split == 'val':
            raise ValueError("No validation split for 5-fold CV scheme.")

        self.root = root
        self.preproc = preproc
        self.split = split
        self.fold = fold
        self.scheme_name = scheme
        self.target_name = target
        self.transforms = transforms
        self.target_transforms = target_transforms

        # Set all the attributes specific to OpenBHB (schemes & studies)
        self._set_dataset_attributes()
        assert hasattr(self, "_studies") & hasattr(self, "_cv_scheme") & \
               hasattr(self, "_train_val_test_scheme") & hasattr(self, "_mapping_sites"), \
            "Missing attributes for %s"%str(self)

        if self.split == "val": self.split = "validation"

        if not self._check_integrity():
            raise RuntimeError("Files not found. Check the the root directory %s"%root)

        if scheme == "train_val_test":
            self.scheme = self.load_pickle(os.path.join(
                root, self._train_val_test_scheme))[self.split]
        elif scheme == "cv":
            f = "fold%i"%(self.fold or 0)
            self.scheme = self.load_pickle(os.path.join(root, self._cv_scheme))[f][self.split]

        npy_files = {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                     "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy"}

        pd_files = {"vbm": "%s_t1mri_mwp1_participants.csv",
                    "quasi_raw": "%s_t1mri_quasi_raw_participants.csv"}


        ## 1) Loads globally all the data for a given pre-processing
        # TODO: change folder name according to preproc
        folder = self.preproc if preproc != "vbm" else "cat12vbm"
        _root = os.path.join(root, folder)
        df_open_bhb = pd.concat([pd.read_csv(os.path.join(_root, pd_files[self.preproc] % db)) for db in self._studies],
                                ignore_index=True, sort=False)
        data_open_bhb = [np.load(os.path.join(_root, npy_files[self.preproc] % db), mmap_mode='r') for db in self._studies]
        cumulative_sizes = np.cumsum([len(db) for db in data_open_bhb])

        ## 2) Selects the data to load in memory according to selected scheme
        unique_keys = ['participant_id', 'study', 'session', 'run']
        mask = self._extract_mask(df_open_bhb, unique_keys=unique_keys)

        # Get TIV and tissue volumes according to the Neuromorphometrics atlas
        self.metadata = self._extract_metadata(df_open_bhb[mask]).reset_index(drop=True)
        self.id = df_open_bhb[mask][unique_keys].reset_index(drop=True)

        # Get the labels to predict
        assert set(target) <= set(df_open_bhb.keys()), "Inconsistent files: missing %s in pandas DataFrame"%target
        self.all_labels = df_open_bhb[mask][["age", "sex", "site"]].reset_index(drop=True)
        self.target = df_open_bhb[mask][target]
        assert self.target.isna().sum().sum() == 0, "Missing values for '%s' label"%target

        # Map the sites to a number and preserve the mapping
        reverse_target_mapping = self.load_pickle(os.path.join(root, self._mapping_sites))
        self.all_labels["site"] = self.all_labels["site"].apply(lambda k: reverse_target_mapping[k])
        self.target_mapping = {i: k for (k, i) in reverse_target_mapping.items()}

        # Eventually, apply the same mapping to target
        if "site" in self.target.keys():
            self.target['site'] = self.target['site'].apply(lambda k: reverse_target_mapping[k]).values
            self.target = self.target.values
        else:
            self.target = self.target.values

        # Prepares private variables to build mapping target_idx -> img_idx
        self.shape = (mask.sum(), *data_open_bhb[0][0].shape)
        self._mask_indices = np.arange(len(df_open_bhb))[mask]
        self._cumulative_sizes = cumulative_sizes
        self._data = data_open_bhb
        self._data_loaded = None

        # Loads all in memory to retrieve it rapidly when needed
        if load_data:
            self._data_loaded = self.get_data()[0]

    def _set_dataset_attributes(self):
        self._studies = ['abide1', 'abide2', 'ixi', 'npc', 'rbp', 'gsp', 'localizer', 'mpi-leipzig', 'corr', 'nar']
        self._train_val_test_scheme = "train_val_test_test-intra_open_bhb_stratified.pkl"
        self._cv_scheme = "5-fold_cv_open_bhb_stratified.pkl"
        self._mapping_sites = "mapping_site_name-class.pkl"


    # TODO: change the formatted names
    def _check_integrity(self):
        """
        Check the integrity of root dir (including the directories/files required). It does NOT check their content.
        Should be formatted as:
        /root
            5-fold_cv_open_bhb_stratified.pkl
            train_val_test_open_bhb_stratified.pkl
            mapping_site_name-class.pkl
            /cat12vbm
                [cohort]_t1mri_mwp1_participants.csv
                [cohort]_t1mri_mwp1_gs-raw_data64.npy
            /quasi_raw
                [cohort]_t1mri_quasi_raw_participants.csv
                [cohort]_t1mri_quasi_raw_data32_1.5mm_skimage.npy
            /fs
                [cohort]_t1mri_free_surfer_participants.csv
                [cohort]_t1mri_free_surfer_data32.npy
        """
        is_complete = os.path.isdir(self.root)
        is_complete &= os.path.isfile(os.path.join(self.root, self._cv_scheme))
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))
        is_complete &= os.path.isfile(os.path.join(self.root, self._mapping_sites))

        dir_files = {
            "cat12vbm": ["%s_t1mri_mwp1_participants.csv", "%s_t1mri_mwp1_gs-raw_data64.npy"],
            "quasi_raw": ["%s_t1mri_quasi_raw_participants.csv", "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy"],
            "fs": []
        }

        for (dir, files) in dir_files.items():
            for file in files:
                for db in self._studies:
                    is_complete &= os.path.isfile(os.path.join(self.root, dir, file%db))
        return is_complete


    def _extract_mask(self, df, unique_keys):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        # TODO: correct this hack in the final version
        df = df.copy()
        df.loc[df['run'].isna(), 'run'] = 1
        if df['run'].dtype == np.float:
            df['run'] = df['run'].astype(int)
        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(np.bool)
        return mask

    def _extract_metadata(self, df):
        """
        :param df: pandas DataFrame
        :return: TIV and tissue volumes defined by the Neuromorphometrics atlas
        """
        metadata = ["tiv"] + [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        assert len(metadata) == 288, "Missing meta-data values (%i != %i)"%(len(metadata), 288)
        assert set(metadata) <= set(df.keys()), "Missing meta-data columns: {}".format(set(metadata) - set(df.keys))
        assert df[metadata].isna().sum().sum() == 0, "NaN values found in meta-data"
        return df[metadata]


    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl

    def _mapping_idx(self, idx):
        """
        :param idx: int ranging from 0 to len(dataset)-1
        :return: integer that corresponds to the original image index to load
        """
        idx = self._mask_indices[idx]
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        return (dataset_idx, sample_idx)

    def get_data(self, indices: Iterable[int]=None, mask: np.ndarray=None, dtype: Type=np.float32):
        """
        Loads all (or selected ones) data in memory and returns a big numpy array X_data with y_data
        The input/target transforms are ignored.
        Warning: this can be VERY memory-consuming (~40GB if all data are loaded)
        :param indices (Optional): list of indices to load
        :param mask (Optional binary mask): binary mask to apply to the data. Each 3D volume is transformed into a
        vector. Can be 3D mask or 4D (channel + img)
        :param dtype (Optional): the final type of data returned (e.g np.float32)
        :return (np.ndarray, np.ndarray), a tuple (X, y)
        """
        (tf, target_tf) = (self.transforms, self.target_transforms)
        self.transforms, self.target_transforms = None, None
        if mask is not None:
            assert len(mask.shape) in [3, 4], "Mask must be 3D or 4D (current shape is {})".format(mask.shape)
            if len(mask.shape) == 3:
                # adds the channel dimension
                mask = mask[np.newaxis, :]
        if indices is None:
            nbytes = np.product(self.shape) if mask is None else mask.sum() * len(self)
            print("Dataset size to load (shape {}): {:.2f} GB".format(self.shape, nbytes*np.dtype(dtype).itemsize/
                                                                      (1024*1024*1024)), flush=True)

            if self._data_loaded is not None:
                data = self._data_loaded[:, mask] if mask is not None else self._data_loaded.copy()
            else:
                if mask is None:
                    data = np.zeros(self.shape, dtype=dtype)
                else:
                    data = np.zeros((len(self), mask.sum()), dtype=dtype)
                for i in range(len(self)):
                    data[i] = self[i][0][mask] if mask is not None else self[i][0]
            self.transforms, self.target_transforms = (tf, target_tf)
            return data, np.copy(self.target)
        else:
            nbytes = np.product(self.shape[1:]) * len(indices) if mask is None else mask.sum() * len(indices)
            print("Dataset size to load (shape {}): {:.2f} GB".format((len(indices),) + self.shape[1:],
                                                                     nbytes*np.dtype(dtype).itemsize/
                                                                      (1024*1024*1024)), flush=True)

            if self._data_loaded is not None:
                data = self._data_loaded[indices, mask] if mask is not None else self._data_loaded[indices]
            else:
                if mask is None:
                    data = np.zeros((len(indices), *self.shape[1:]), dtype=dtype)
                else:
                    data = np.zeros((len(indices), mask.sum()), dtype=dtype)
                for i, idx in enumerate(indices):
                    data[i] = self[idx][0][mask] if mask is not None else self[idx][0]
            self.transforms, self.target_transforms = (tf, target_tf)
            return data.astype(dtype), self.target[indices]


    def transform(self, tf, *args, mask: np.ndarray=None, dtype: Type=np.float32, copy: bool=True, **kwargs):
        """
        :param tf: a Transformer object that implements transform() to by apply on the data
        NB: the data shape must be preserved after transformation
        :param *args, **kwargs: arguments to give to the Transformer object
        :param mask: a 3D or 4D mask given to self.get_data()
        :param copy: if True, returns a copy of self whose data have been transformed
        :return: an OpenBHB dataset whose data have been transformed and stored directly in _data_loaded
        """
        this = self
        if copy: this = self.copy()
        # Preserves the data shape
        data_shape = this.shape
        this_data, _ = this.get_data(mask=mask, dtype=dtype)
        this._data_loaded = np.zeros(data_shape, dtype=dtype)
        if mask is None:
            this._data_loaded = tf.transform(this_data, *args, **kwargs)
        else:
            if len(mask.shape) == 3: mask = mask[np.newaxis, :]
            this._data_loaded[:, mask] = tf.transform(this_data, *args, **kwargs)
        return this

    def copy(self):
        """
        :return: a deep copy of this
        """

        this = self.__class__(self.root, self.preproc, self.scheme_name, self.target_name,
                       self.split, self.fold, self.transforms, self.target_transforms)
        return this


    @staticmethod
    def get_mask(root: str, preproc: str, masking_fn: Callable[[np.ndarray], np.ndarray]=(lambda x: (x>0)),
                 operator: str='|'):
        """
        For a given pre-processing, it computes a global mask on the selected dataset according
        to a given masking function (computed on each image) and an operator (computed between masks).
        NB: the final mask contains the channel dimension, having shape (C, H, W, D)

        :param root, path to OpenBHB data
        :param preproc: "vbm", "quasi_raw" or "fs"
        :param masking_fn: a callable fn that takes an array as param and returns a binary mask
        :param operator: str, must be '|' or '&'. Operator applied between all masks obtained from images
        :return: binary mask
        """

        assert operator in ['|', '&'], "Unknown operator: %s"%operator
        assert preproc in ['vbm', 'quasi_raw'], "Unknown preprocessing: %s"%preproc

        d_train = OpenBHB(root, preproc=preproc, scheme="cv", split="train", fold=0)
        d_test = OpenBHB(root, preproc=preproc, scheme="cv", split="test", fold=0)

        mask_shape = d_test.shape[1:]
        global_mask = np.ones(mask_shape, dtype=np.bool) if operator=="&" else np.zeros(mask_shape, dtype=np.bool)
        op = np.logical_or if operator == "|" else np.logical_and
        for d in [d_train, d_test]:
            for i in range(len(d)):
                global_mask = op(global_mask, masking_fn(d[i][0]))
        return global_mask


    def __getitem__(self, idx: int):
        if self._data_loaded is not None:
            sample, target = self._data_loaded[idx], self.target[idx]
        else:
            (dataset_idx, sample_idx) = self._mapping_idx(idx)
            sample, target = self._data[dataset_idx][sample_idx], self.target[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return sample, target

    def __len__(self):
        return len(self.target)


    def __str__(self):
        if self.fold is not None:
            return "OpenBHB-%s-%s-%s-%s"%(self.preproc, self.scheme_name, self.split, self.fold)
        return "OpenBHB-%s-%s-%s"%(self.preproc, self.scheme_name, self.split)

class SubOpenBHB(OpenBHB):
    """
    This class is a subset of OpenBHB. It is defined only for Train/Val/Test split. It allows to perform
    (Stratified) Shuffle Split inside the training set for a given N_train. The stratification is
    always performed according to the target. The random seed is fixed so that the code is fully reproducible.
    """
    def __init__(self, *args, N_train_max: int=None, stratify: [bool, str, List[str]]=True, nb_folds: int=3,
                 load_data: bool=False, **kwargs):
        """
        :param args: args to give to OpenBHB
        :param N_train_max: number of training samples to sub-sample from OpenBHB
        :param stratify: stratify according to the given column names. It can stratify in a multi-label fashion.
                         If set to True, stratify according to Age+Sex+Site.
        :param nb_folds: number of folds in the Monte-Carlo sub-sampling
        :param load_data: If True, loads all the data in memory
        :param kwargs: passed to OpenBHB
        """
        super().__init__(*args, **kwargs)
        self.args, self.kwargs = args, kwargs
        self.stratify = stratify
        if isinstance(stratify, str):
            self.stratify = [stratify]
        if isinstance(stratify, bool):
            self.stratify = list(self.all_labels.keys())
        if isinstance(self.stratify, list):
            assert (set(self.stratify) <= set(self.all_labels)) and len(self.stratify) > 0

        self.nb_folds = nb_folds or 1
        self.N_train_max = N_train_max or len(self)

        if self.scheme_name != "train_val_test":
            raise RuntimeError("Scheme must be 'train_val_test'")

        if self.split == "train":
            self.fold = self.fold or 0
            assert 0 <= self.fold < self.nb_folds, "Incorrect fold index: %i"%self.fold
            assert self.N_train_max <= len(self), "Inconsistent N_train (got >%i)"%len(self)
            if self.stratify:
                if len(self.stratify) > 1:
                    splitter = MultilabelStratifiedShuffleSplit(n_splits=nb_folds, train_size=self.N_train_max,
                                                                test_size=len(self)-self.N_train_max,
                                                                random_state=0)
                else:
                    splitter = StratifiedShuffleSplit(n_splits=nb_folds, train_size=self.N_train_max, random_state=0)
            else:
                splitter = ShuffleSplit(n_splits=self.nb_folds, train_size=self.N_train_max, random_state=0)
            dummy_x = np.zeros(len(self))
            if isinstance(self.stratify, list):
                y = self.all_labels[self.stratify].copy(deep=True).values
                if "age" in self.stratify:
                    i_age = self.stratify.index("age")
                    y[:, i_age] = SubOpenBHB.discretize_continous_label(y[:, i_age])
            else:
                raise ValueError("Unknown stratifier: {}".format(self.stratify))
            gen = splitter.split(dummy_x, y)
            for _ in range(self.fold+1):
                train_index, _ = next(gen)
            self._train_index = train_index
            self.all_labels = self.all_labels.iloc[train_index].reset_index(drop=True)
            self.target = self.target[train_index]
            self.metadata = self.metadata.iloc[self._train_index].reset_index(drop=True)
            self.id = self.id.iloc[self._train_index].reset_index(drop=True)
            self.shape = (len(self._train_index), *self.shape[1:])

        if load_data:
            self._data_loaded = self.get_data()[0]

    @staticmethod
    def discretize_continous_label(labels, bins: [str, int]="sturges"):
        # Get an estimation of the best bin edges. 'Sturges' is conservative for pretty large datasets (N>1000).
        bin_edges = np.histogram_bin_edges(labels, bins=bins)
        # Discretizes the values according to these bins
        discretization = np.digitize(labels, bin_edges[1:], right=True)
        return discretization


    def copy(self):
        this = SubOpenBHB(*self.args, N_train_max=self.N_train_max, stratify=self.stratify, nb_folds=self.nb_folds,
                          **self.kwargs)
        return this

    def __getitem__(self, idx:int):
        if self.split == "train":
            if self._data_loaded is not None:
                sample, target = self._data_loaded[idx], self.target[idx]
            else:
                (dataset_idx, sample_idx) = self._mapping_idx(self._train_index[idx])
                sample, target = self._data[dataset_idx][sample_idx], self.target[idx]
            if self.transforms is not None:
                sample = self.transforms(sample)
            if self.target_transforms is not None:
                target = self.target_transforms(target)
            return sample, target
        return super().__getitem__(idx)