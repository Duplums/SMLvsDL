from torch.utils.data.dataset import Dataset
from abc import ABC, abstractmethod
import os, pickle
import pandas as pd
import numpy as np
import bisect, logging
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from typing import Callable, List, Type, Sequence, Dict

class ClinicalBase(ABC, Dataset):
    """
        A generic clinical Dataset written in a torchvision-like manner. It parses a .pkl file defining the different
        splits based on a <unique_key>. All clinical datasets must have:
        - a training set
        - a validation set
        - a test set
        - (eventually) an other intra-test set

        This generic dataset is memory-efficient, taking advantage of memory-mapping implemented with NumPy.
        It always come with:
        ... 3 pre-processings:
            - Quasi-Raw
            - VBM
            - FreeSurfer
        ... And 2 differents tasks:
            - Diagnosis prediction (classification)
            - Site prediction (classification)
        ... With meta-data:
            - user-defined unique identifier across pre-processing and split
            - TIV + ROI measures based on Neuromorphometrics atlas
    Attributes:
          * target, list[int]: labels to predict
          * all_labels, pd.DataFrame: all labels stored in a pandas DataFrame containing ["diagnosis", "site", "age", "sex]
          * shape, tuple: shape of the data
          * metadata: pd DataFrame: Age + Sex + TIV + ROI measures extracted for each image
          * id: pandas DataFrame, each row contains a unique identifier for an image

    """
    def __init__(self, root: str, preproc: str='vbm', target: [str, List[str]]='diagnosis',
                 split: str='train', transforms: Callable[[np.ndarray], np.ndarray]=None,
                 load_data: bool=False):
        """
        :param root: str, path to the root directory containing the different .npy and .csv files
        :param preproc: str, must be either VBM ('vbm'), Quasi-Raw ('quasi_raw') or FreeSurfer ('fs')
        :param target: str or [str], either 'dx' or 'site'.
        :param split: str, either 'train', 'val', 'test' (inter) or (eventually) 'test_intra'
        :param transforms (callable, optional): A function/transform that takes in
            a 3D MRI image and returns a transformed version.
        :param load_data (bool, optional): If True, loads all the data in memory
               --> WARNING: it can be time/memory-consuming
        """
        if isinstance(target, str):
            target = [target]
        assert preproc in ['vbm', 'quasi_raw', 'fs'], "Unknown preproc: %s"%preproc
        assert set(target) <= {'diagnosis', 'site'}, "Unknown target: %s"%target
        assert split in ['train', 'val', 'test', 'test_intra', 'validation'], "Unknown split: %s"%split

        self.root = root
        self.preproc = preproc
        self.split = split
        self.target_name = target
        self.transforms = transforms
        self.logger = logging.getLogger("pynet")

        if self.split == "val": self.split = "validation"

        if not self._check_integrity():
            raise RuntimeError("Files not found. Check the the root directory %s"%root)

        self.scheme = self.load_pickle(os.path.join(
            root, self._train_val_test_scheme))[self.split]

        npy_files = {"vbm": "%s_t1mri_mwp1_gs-raw_data64.npy",
                     "quasi_raw": "%s_t1mri_quasi_raw_data32_1.5mm_skimage.npy"}

        pd_files = {"vbm": "%s_t1mri_mwp1_participants.csv",
                    "quasi_raw": "%s_t1mri_quasi_raw_participants.csv"}


        ## 1) Loads globally all the data for a given pre-processing
        folder = self.preproc if preproc != "vbm" else "cat12vbm"
        _root = os.path.join(root, folder)
        df = pd.concat([pd.read_csv(os.path.join(_root, pd_files[self.preproc] % db)) for db in self._studies],
                       ignore_index=True, sort=False)
        data = [np.load(os.path.join(_root, npy_files[self.preproc] % db), mmap_mode='r')
                         for db in self._studies]
        cumulative_sizes = np.cumsum([len(db) for db in data])

        ## 2) Selects the data to load in memory according to selected scheme
        mask = self._extract_mask(df, unique_keys=self._unique_keys, check_uniqueness=self._check_uniqueness)

        # Get TIV and tissue volumes according to the Neuromorphometrics atlas
        self.metadata = self._extract_metadata(df[mask]).reset_index(drop=True)
        self.id = df[mask][self._unique_keys].reset_index(drop=True)

        # Get the labels to predict
        assert set(self.target_name) <= set(df.keys()), \
            "Inconsistent files: missing %s in pandas DataFrame"%self.target_name
        self.target = df[mask][self.target_name]
        assert self.target.isna().sum().sum() == 0, "Missing values for '%s' label"%self.target_name
        self.target = self.target.apply(self.target_transform_fn, axis=1, raw=True).values.ravel().astype(np.float32)

        all_keys = ["age", "sex", "diagnosis", "site"]
        self.all_labels = df[mask][all_keys].reset_index(drop=True)
        # Transforms (dx, site) according to _dx_site_mappings
        self.all_labels = self.all_labels.apply(lambda row: [row[0], row[1],
                                                             self._dx_site_mappings["diagnosis"][row[2]],
                                                             self._dx_site_mappings["site"][row[3]]],
                                                axis=1, raw=True, result_type="broadcast")
        # Prepares private variables to build mapping target_idx -> img_idx
        self.shape = (mask.sum(), *data[0][0].shape)
        self._mask_indices = np.arange(len(df))[mask]
        self._cumulative_sizes = cumulative_sizes
        self._data = data
        self._data_loaded = None

        # Loads all in memory to retrieve it rapidly when needed
        if load_data:
            self._data_loaded = self.get_data()[0]

    @property
    @abstractmethod
    def _studies(self) -> List[str]:
        ...
    @property
    @abstractmethod
    def _train_val_test_scheme(self) -> str:
        ...
    @property
    @abstractmethod
    def _unique_keys(self) -> List[str]:
        ...
    @property
    @abstractmethod
    def _dx_site_mappings(self) -> Dict[str, Dict[str, int]]:
        ...
    @property
    def _check_uniqueness(self) -> bool:
        return True

    def _check_integrity(self):
        """
        Check the integrity of root dir (including the directories/files required). It does NOT check their content.
        Should be formatted as:
        /root
            <train_val_test_split.pkl>
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
        is_complete &= os.path.isfile(os.path.join(self.root, self._train_val_test_scheme))

        # TODO: change the formatted names
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


    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        """
        :param df: pandas DataFrame
        :param unique_keys: list of str
        :param check_uniqueness: if True, check the unique_keys identified uniquely an image in the dataset
        :return: a binary mask indicating, for each row, if the participant belongs to the current scheme or not.
        """
        _source_keys = df[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        if check_uniqueness:
            assert len(set(_source_keys)) == len(_source_keys), "Multiple identique identifiers found"
        _target_keys = self.scheme[unique_keys].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        mask = _source_keys.isin(_target_keys).values.astype(np.bool)
        return mask

    def _extract_metadata(self, df: pd.DataFrame):
        """
        :param df: pandas DataFrame
        :return: TIV and tissue volumes defined by the Neuromorphometrics atlas
        """
        metadata = ["age", "sex", "tiv"] + [k for k in df.keys() if "GM_Vol" in k or "WM_Vol" in k or "CSF_Vol" in k]
        assert len(metadata) == 290, "Missing meta-data values (%i != %i)"%(len(metadata), 290)
        assert set(metadata) <= set(df.keys()), "Missing meta-data columns: {}".format(set(metadata) - set(df.keys))
        if df[metadata].isna().sum().sum() > 0:
            self.logger.warning("NaN values found in meta-data")
        return df[metadata]

    def target_transform_fn(self, target):
        ## Transforms the target according to mapping site <-> int and dx <-> int
        target = target.copy()
        for i, name in enumerate(self.target_name):
            target[i] = self._dx_site_mappings[name][target[i]]
        return target

    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        return pkl

    def get_data(self, indices: Sequence[int]=None, mask: np.ndarray=None, dtype: Type=np.float32):
        """
        Loads all (or selected ones) data in memory and returns a big numpy array X_data with y_data
        The input/target transforms are ignored.
        Warning: this can be memory-consuming (~10GB if all data are loaded)
        :param indices (Optional): list of indices to load
        :param mask (Optional binary mask): binary mask to apply to the data. Each 3D volume is transformed into a
        vector. Can be 3D mask or 4D (channel + img)
        :param dtype (Optional): the final type of data returned (e.g np.float32)
        :return (np.ndarray, np.ndarray), a tuple (X, y)
        """
        tf = self.transforms
        self.transforms = None
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
            self.transforms = tf
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
            self.transforms = tf
            return data.astype(dtype), self.target[indices]

    def _mapping_idx(self, idx: int):
        """
        :param idx: int ranging from 0 to len(dataset)-1
        :return: integer that corresponds to the original image index to load
        """
        idx = self._mask_indices[idx]
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        sample_idx = idx - self._cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else idx
        return (dataset_idx, sample_idx)


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

        this = self.__class__(self.root, self.preproc, self.target_name,
                              self.split, self.transforms)
        return this

    def __getitem__(self, idx: int):
        if self._data_loaded is not None:
            sample, target = self._data_loaded[idx], self.target[idx]
        else:
            (dataset_idx, sample_idx) = self._mapping_idx(idx)
            sample, target = self._data[dataset_idx][sample_idx], self.target[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, target.astype(np.float32)

    def __len__(self):
        return len(self.target)


    def __str__(self):
        return "%s-%s-%s"%(type(self).__name__, self.preproc, self.split)


class SubClinicalDataset(ClinicalBase):
    """
        This class is a subset of ClinicalBase. It allows to perform (Stratified) Shuffle Split (i.e Monte-Carlo CV)
        inside the training set for a given N_train and # folds. The stratification can be multi-label and is
        defined by the user. The random seed is fixed so that the code is fully reproducible.
        """

    def __init__(self, *args, N_train_max: int = None, stratify: [bool, str, List[str]] = True, fold: int = 0,
                 nb_folds: int = 3, load_data: bool = False, **kwargs):
        """
        :param args: args to give to ClinicalBase
        :param N_train_max: number of training samples to sub-sample from OpenBHB
        :param stratify: stratify according to the given column names. It can stratify in a multi-label fashion.
                         If set to True, stratify according to Age+Sex+Site+Diagnosis.
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

        if self.split == "train":
            self.fold = fold
            self.nb_folds = nb_folds
            self.N_train_max = N_train_max or len(self)
            assert 0 <= self.fold < self.nb_folds, "Incorrect fold index: %i" % self.fold
            assert self.N_train_max <= len(self), "Inconsistent N_train (got >%i)" % len(self)
            if self.stratify:
                if len(self.stratify) > 1:
                    splitter = MultilabelStratifiedShuffleSplit(n_splits=nb_folds, train_size=self.N_train_max,
                                                                test_size=len(self) - self.N_train_max,
                                                                random_state=0)
                else:
                    splitter = StratifiedShuffleSplit(n_splits=nb_folds, train_size=self.N_train_max,
                                                      random_state=0)
            else:
                splitter = ShuffleSplit(n_splits=self.nb_folds, train_size=self.N_train_max, random_state=0)
            dummy_x = np.zeros(len(self))
            if isinstance(self.stratify, list):
                y = self.all_labels[self.stratify].copy(deep=True).values
                if "age" in self.stratify:
                    i_age = self.stratify.index("age")
                    y[:, i_age] = SubClinicalDataset.discretize_continous_label(y[:, i_age].astype(np.float32))
            else:
                raise ValueError("Unknown stratifier: {}".format(self.stratify))
            gen = splitter.split(dummy_x, y)
            for _ in range(self.fold + 1):
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
    def discretize_continous_label(labels, bins: [str, int] = "sturges"):
        # Get an estimation of the best bin edges. 'Sturges' is conservative for pretty large datasets (N>1000).
        bin_edges = np.histogram_bin_edges(labels, bins=bins)
        # Discretizes the values according to these bins
        discretization = np.digitize(labels, bin_edges[1:], right=True)
        return discretization

    def copy(self):
        this = self.__class__(*self.args, N_train_max=self.N_train_max, stratify=self.stratify,
                              nb_folds=self.nb_folds, **self.kwargs)
        return this

    def __getitem__(self, idx: int):
        if self.split == "train":
            if self._data_loaded is not None:
                sample, target = self._data_loaded[idx], self.target[idx]
            else:
                (dataset_idx, sample_idx) = self._mapping_idx(self._train_index[idx])
                sample, target = self._data[dataset_idx][sample_idx], self.target[idx]
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample, target.astype(np.float32)
        return super().__getitem__(idx)


class SCZDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["schizconnect-vip", "bsnip", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_scz_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "schizophrenia": 1},
                    site=self._site_mapping)

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        # Little hack
        df = df.copy()
        df.loc[df['session'].isna(), 'session'] = 1
        df.loc[df['session'].isin(['v1', 'V1']), 'session'] = 1
        df["session"] = df["session"].astype(int)
        self.scheme['session'] = self.scheme['session'].astype(int)
        return super()._extract_mask(df, unique_keys, check_uniqueness=check_uniqueness)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_scz.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_scz.pkl"))
        super().__init__(root, *args, **kwargs)


class SubSCZDataset(SubClinicalDataset, SCZDataset):
    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_scz.pkl"))
        super().__init__(root, *args, **kwargs) # Call to SubClinicalDataset


class BipolarDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["biobd", "bsnip", "cnp", "candi"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_bip_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "bipolar": 1, "bipolar disorder": 1, "psychotic bipolar disorder": 1},
                    site=self._site_mapping)

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        # Little hack
        df = df.copy()
        df.loc[df['session'].isna(), 'session'] = 1
        df.loc[df['session'].isin(['v1', 'V1']), 'session'] = 1
        df["session"] = df["session"].astype(int)
        self.scheme['session'] = self.scheme['session'].astype(int)
        return super()._extract_mask(df, unique_keys, check_uniqueness=check_uniqueness)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_bip.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_bip.pkl"))
        super().__init__(root, *args, **kwargs)


class SubBipolarDataset(SubClinicalDataset, BipolarDataset):
    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_bip.pkl"))
        super().__init__(root, *args, **kwargs)


class ASDDataset(ClinicalBase):

    @property
    def _studies(self):
        return ["abide1", "abide2"]

    @property
    def _train_val_test_scheme(self):
        return "train_val_test_test-intra_asd_stratified.pkl"

    @property
    def _unique_keys(self):
        return ["participant_id", "session", "study", "run"]

    @property
    def _dx_site_mappings(self):
        return dict(diagnosis={"control": 0, "autism": 1},
                    site=self._site_mapping)

    def _extract_mask(self, df: pd.DataFrame, unique_keys: Sequence[str], check_uniqueness: bool=True):
        # Little hack
        df = df.copy()
        df.loc[df['session'].isna(), 'session'] = 1
        df.loc[df['session'].isin(['v1', 'V1']), 'session'] = 1
        df["session"] = df["session"].astype(int)
        df.loc[df['run'].isna(), 'run'] = 1
        if df['run'].dtype == np.float:
            df['run'] = df['run'].astype(int)
        self.scheme['session'] = self.scheme['session'].astype(int)

        return super()._extract_mask(df, unique_keys, check_uniqueness=check_uniqueness)

    def _check_integrity(self):
        return super()._check_integrity() & os.path.isfile(os.path.join(self.root, "mapping_site_name-class_asd.pkl"))

    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_asd.pkl"))
        super().__init__(root, *args, **kwargs)


class SubASDDataset(SubClinicalDataset, ASDDataset):
    def __init__(self, root: str, *args, **kwargs):
        self._site_mapping = self.load_pickle(os.path.join(root, "mapping_site_name-class_asd.pkl"))
        super().__init__(root, *args, **kwargs)