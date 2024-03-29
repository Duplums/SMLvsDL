## Objective 

This library aims at reproducing results described in "Deep Learning Improvement over Standard Machine Learning in Anatomical Neuroimaging comes from Transfer Learning", Under Review

It contains the main scripts to run the different experiments with 1) Standard Machine Learning (SML) models including kernel-SVM and 
regularized linear models (Logistic Regression with l1 and ElasticNet); 2) CNN models including 3D-AlexNet, 3D-ResNet and 3D-DenseNet.

The scripts to run sensitivity analysis and model occlusion are also given along with dimensionality reduction and data harmonization.

PyTorch is used to run Deep Learning experiments while scikit-learn is used for SML.  

## Software Requirements
### OS and Hardware Requirements

This library has been tested only on Linux 18.04. The experiments have been executed for the most part
on [Jean-Zay](http://www.idris.fr/jean-zay/jean-zay-presentation.html) cluster equipped with 4 NVIDIA Tesla V100 per node 
with 32 Go GPU each. 

### Required Packages and Installation 

A [conda](https://docs.conda.io/en/latest/) v4.10.1 environment has been used to run this library in a standalone mode. 
All the dependencies can be found in requirement.txt with the exact version for each package used. The conda environement can 
be easily reproduced with:

`conda env create -f environment.yml`

Installing all packages with dependencies can take up to several hours, depending on the internet connexion and less than 10GB on disk.

*Important Note:* in order to run data harmonization with linear adjusted residualization, the package MULM is necessary. For now, 
 it is not accessible through conda. It can be clone from the GitHub repository: https://github.com/neurospin/pylearn-mulm

*Brain masks:* Throughout the experiments with SML and DL, we generally applied a brain mask to remove noisy voxels outside the brain tissues.
These masks are available in `/masks` to ease reproducibility.  

## Demo

### Expected data
Currently, 5 torch Dataset objects have been written for this library: `OpenBHB`, `BHB`, `SCZDataset`, `BipolarDataset` and 
`ASDDataset`. They assume an underlying data structure that is defined in `_check_integrity` function. For now, only OpenBHB
is publicly available on [IEEE Dataport](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing).
The others can be downloaded on the dedicated web platforms (see below).

#### OpenBHB Dataset 

`OpenBHB` aggregates 10 brain MRI datasets of healthy controls (HC) both pre-processed with VBM and Quasi-Raw.
Pre-processed data are hosted [here](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing).

**Source**  | **# Subjects**  | **# Sessions** | **Age** | **Sex (\%F)** | **# Sites**
|:---: | :---: | :---: | :---: | :---: | :---: | 
[IXI](http://brain-development.org/ixi-dataset) | 559 | 559 | 48 ± 16 | 55 | 3 
[CoRR](https://www.nitrc.org/projects/fcon_1000) | 1366 | 2873 | 26 ± 16 | 50 | 19
[NPC](https://openneuro.org/datasets/ds002330/versions/1.1.0) | 65 | 65 | 26 ± 4 | 55 | 1
[NAR](https://openneuro.org/datasets/ds002345/versions/1.0.1) | 303 | 323 | 22 ± 5 | 58 | 1
[RBP](https://openneuro.org/datasets/ds002247/versions/1.0.0) | 40 | 40 | 22 ± 5 | 52 | 1
[GSP](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/25833) | 1570 | 1639 | 21 ± 3 | 58 | 5
[ABIDE 1](http://fcon_1000.projects.nitrc.org/indi/abide) (HC) | 566 | 566 | 17 ± 8 | 17 | 20
[ABIDE 2](http://fcon_1000.projects.nitrc.org/indi/abide) (HC) | 542 | 555 | 15 ± 9 | 30 | 19
[Localizer](http://brainomics.cea.fr/localizer/localizer) | 82 | 82 | 25 ± 7 | 56 | 2
[MPI-Leipzig](https://openneuro.org/datasets/ds000221/versions/00002) | 316 | 317 | 37 ± 19 | 40 | 2

#### BHB Dataset

The BHB Dataset includes `OpenBHB` along with 3 additional cohorts detailed hereafter that must be downloaded on the 
dedicated web platforms, and healthy controls from 3 clinical cohorts (BIOBD, SCHIZCONNECT and BSNIP, see below).

**Source**  | **# Subjects**  | **# Sessions** | **Age** | **Sex (\%F)** | **# Sites**
|:---: | :---: | :---: | :---: | :---: | :---: | 
[HCP](https://www.humanconnectome.org/study/hcp-young-adult)  | 1113 | 1113 | 29 ± 4 | 45 | 1
[OASIS 3](https://www.oasis-brains.org) | 578 | 1166 | 68 ± 9 | 62 | 4
[ICBM](https://ida.loni.usc.edu) | 606 | 939 | 30 ± 12 | 45 | 3


#### Clinical Datasets

The 3 clinical datasets `SCZDataset`, `BipolarDataset` and `ASDDataset` are derived mostly from public cohorts excepted for 
BIOBD, BSNIP and PRAGUE, that are private for clinical research. These 3 datasets are based on the following sources.

**Source**  | **Disease** | **# Subjects** | **Age** | **Sex (\%F)** | **# Sites**
| :---:| :---: | :---: | :---: | :---: | :---: |
[BSNIP](http://b-snip.org)  | Control<br>Schizophrenia<br>Bipolard Disorder | 198<br>190<br>116 | 32 ± 12<br>34 ± 12<br>37 ± 12 | 58<br>30<br>66 | 5
[SCHIZCONNECT](http://schizconnect.org)  | Control<br>Schizophrenia | 275<br>329 | 34 ± 12<br>32 ± 13 | 28<br>47 | 4
PRAGUE  | Control | 90 | 26 ± 7 | 55 | 1
[BIOBD](https://pubmed.ncbi.nlm.nih.gov/29981196/) | Control<br>Bipolar Disorder | 306<br>356 | 40 ± 12<br>40 ± 13 | 55 | 8
[CANDI](https://www.nitrc.org/projects/candi_share) | Control<br>Schizophrenia | 25<br>20 | 10 ± 3<br>13 ± 3 | 41<br>45 | 1
[CNP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5664981/) | Control<br>Schizophrenia<br>Bipolar Disorder | 123<br>50<br>49 | 31 ± 9<br>36 ± 9<br>35 ± 9| 47<br>24<br>43 | 1 

### Running SML 

The scripts `clinical_sml.py` and `age_sex_sml.py` in `sml_training` directory allow to run SML models on clinical and healthy subjects datasets respectively 
for clinical classification (patient vs control) and phenotype prediction (age regression or sex classification). 

#### SML for classification and regression

Both `clinical_sml.py` and `age_sex_sml.py` can be executed as followed from a bash terminal:

```
python3 <script>.py --root <ROOT_DIR> --saving_dir <SAVE_DIR> --pb <PB> --preproc <PREPROC> --model <MODEL> --nb_folds 3 --N_train <N>
```

where all parameters are described with a helper. 

*Problem definition:* it can be set through `--pb`. For `age_sex_sml.py`, there are two available problems: age regression (age) 
and sex classification (sex). As for `clinical_sml.py`, there are 3 clinical binary classification problems (patient vs control): 
schizophrenia (scz), autism spectrum disorders (asd) and bipolar disorder (bipolar). 

*Pre-processing:* we assume to have access to 2 different pre-processing: VBM (vbm) and Quasi-Raw (quasi_raw). It can be set through
`--preproc`.    

*Dimensionality Reduction:* by default, 3 different reduction methods are tested (GRP, UFS, RFE) but 
they can be chosen with `--red_meth` parameter. In that case, the number of selected features can be chosen with `--nfeatures`.

*Data Harmonization:* data can be residualized with linear adjusted regression or ComBat with 
the parameter `--residualize`. The Linear Adjusted Regression needs MULM package that can be found on [GitHub](https://github.com/neurospin/pylearn-mulm).


*Changing training size:* the training set can be further downsampled with `--N_train` parameter through stratified random split. 
For a given pair `(N_train, nb_folds)`, a unique train split is built that is reproducible across machines (the random seed is fixed).

### Running DL

The script `dl_training/main.py` is the main entry point to run the experiments with DL models. It can be executed from a 
terminal console with:

`python3 dl_training/main.py [--OPT]`

The options are documented through a helper. The main command lines can be found below.


#### Learning curves for age and sex prediction

Here is the command line to train a DenseNet121 on age prediction with N=10K training samples and VBM pre-processing.
Network, task, number of training samples and pre-processing can be easily adapted.  
```
ROOT="."
CHK="."
PREPROC="vbm"
NET="densenet121" # can be also "resnet18" or "alexnet"
PB="age" # can be "sex"
# For N>5K, switch to BHB with N=9253 samples.
N=9253 # in [100, 500, 1000, 3000, 5000, 9253]
# Age prediction
python3 dl_training/main.py --root $ROOT --checkpoint_dir $CHK --preproc $PREPROC \
--exp_name ${NET}_${PREPROC}_${PB}_N$N --pb $PB --N_train_max $N --nb_folds 3 --net $NET \
--batch_size 32 --lr 1e-4 --gamma_scheduler 0.8 --sampler random --train --test
```

On BHB, this should give MAE=2.58 and MAE=3.53 respectively on internal and external test set. On sex prediction,
with the same architecture and training size, it should give AUC=97% and AUC=98% on internal and external test 
respectively. 

#### Diagnosis prediction for schizophrenia, bipolar disorder and ASD

To train a classifier (e.g DenseNet121) on clinical datasets, the following command line can be executed: 
```
ROOT="."
CHK="."
PREPROC="vbm" # can also be "quasi_raw"
NET="densenet121" # can be also "resnet18" or "alexnet"
PB="scz" # can be aslo "asd" or "bipolar"
# Schizophrenia classification
python3 dl_training/main.py --root $ROOT --checkpoint_dir $CHK --preproc $PREPROC \
--exp_name ${NET}_${PREPROC}_${PB} --pb $PB --nb_folds 3 --net $NET \
--batch_size 32 --lr 1e-4 --gamma_scheduler 0.8 --sampler random --nb_epochs 100
```

On schizophrenia classification, this should give AUC=85%/75% on internal/external test respectively. For 
bipolar classification, AUC=76%/68% and for ASD classification, AUC=66%/63%. 

#### Deep Ensembling

The easiest way to perform Deep Ensemble learning is to run `p` times the same command line as before by specifying
a different seed each time with `--manual_seed`. Then each network can output a prediction independently and they can
be averaged.

#### Self-supervised experiments

To perform contrastive learning (a self-supervised algorithm) with Age-Aware InfoNCE loss (introduced 
[here](https://arxiv.org/pdf/2106.08808.pdf)), the following command pre-train a DenseNet on OpenBHB.
```
ROOT="."
CHK="."
PREPROC="vbm" # can also be "quasi_raw"
NET="densenet121"
PB="self_supervised"
SIGMA=5
python3 dl_training/main.py --root $ROOT --checkpoint_dir $CHK --preproc $PREPROC \
--exp_name ${NET}_${PREPROC}_${PB} --pb $PB --nb_folds 3 --net $NET --sigma $SIGMA \
--batch_size 64 --lr 1e-4 --gamma_scheduler 0.8 --sampler random --nb_epochs 100  
```
*Important Remark:* since OpenBHB contains part of ABIDE, the pre-trained DenseNet cannot be fine-tuned directly
on ASD dataset (containing also this dataset). Special care must be taken by removing ABIDE from the pre-training dataset.  

#### Fine-tuning experiments

Assuming the previous network has been pre-trained with self-supervision, it can be fine-tuned through:
 ```
ROOT="."
CHK="."
PREPROC="vbm" # can also be "quasi_raw"
NET="densenet121"
PB="scz" # can also be "asd" or "bipolar"
$PRETRAINING="${NET}_${PREPROC}_self_supervised_0_epoch_99.pth"
python3 dl_training/main.py --root $ROOT --checkpoint_dir $CHK --preproc $PREPROC \
--exp_name ${NET}_${PREPROC}_${PB}_finetuned --pb $PB --nb_folds 3 --net $NET \
--sigma $SIGMA --batch_size 64 --lr 1e-4 --gamma_scheduler 0.8 --sampler random \
--nb_epochs 100 --pretrained_path ${PRETRAINING}$ --train --test
```

### Sensitivity Analysis and Model Occlusion

In this library, sensitivity analysis and model occlusion are perform with the AAL atlas that can be found in `/atlas`.
They can be runt both on scikit-learn models and Torch models with:
 ```
ROOT="."
DIR="."
PREPROC="vbm" # can also be "quasi_raw"
NET="densenet121"
METH="gradient" # can be "occ"
PB="age" # can also be "sex", "scz", "asd" or "bipolar"
CHK="${NET}_${PREPROC}_${PB}_0_epoch_299.pth"
python3 sml_training/run_saliency_maps.py --root $ROOT --saving_dir $DIR --preproc $PREPROC \
--saliency_meth ${METH} --pb $PB --chkpt $CHK
```

In the end, this dumps a pickle file containing a dictionary with normalized relevance score computed for each brain region and 
each testing sample. 
