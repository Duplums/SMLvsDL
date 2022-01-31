import argparse, nibabel, os, sys
sys.path.extend(['.', '..'])
import numpy as np
import torch
import logging
from datasets.open_bhb import OpenBHB
from torch.utils.data import SequentialSampler
from sml_training.sk_trainer import MLTrainer
from dl_training.utils import get_pickle_obj
from dl_training.models.alexnet import AlexNet3D_Dropout
from dl_training.models.resnet import resnet18
from dl_training.models.densenet import densenet121
from torchvision.transforms.transforms import Compose
from torch.utils.data import DataLoader
from dl_training.transforms import Crop, Padding, Normalize
from sklearn.preprocessing import StandardScaler
from dl_training.datamanager import Zscore, OpenBHBDataManager
from sml_training.saliency_maps import area_occlusion, sensitivity_analysis, \
    get_brain_area_masks, get_relevance_per_area
from sklearn.base import is_regressor, is_classifier
from tqdm import tqdm


class ModelEnsemble(torch.nn.Module):
    def __init__(self, models, apply_softmax=True):
        super().__init__()
        self.models = models
        self.apply_softmax = apply_softmax

    def forward(self, x):
        out = [model(x) for model in self.models]
        if self.apply_softmax:
            if len(out)>0 and (len(out[0].shape) == 1 or out[0].shape[1] == 1):
                out = [torch.sigmoid(o) for o in out]
            else:
                out = [torch.nn.functional.softmax(o, dim=1) for o in out]
        out = self.aggregate(out)
        return out

    def aggregate(self, x):
        return sum(x)/len(x)


def load_dl_model(pths, apply_softmax=True):
    if type(pths) == "str":
        pths = [pths]
    nets = []
    for pth in pths:
        checkpoint = torch.load(pth, map_location=torch.device('cuda' if torch.cuda.is_available()
                                                               else 'cpu')).get("model")
        if "AlexNet" in pth:
            net = AlexNet3D_Dropout(num_classes=1)
        elif "DenseNet" in pth:
            net = densenet121(num_classes=1)
        elif "ResNet" in pth:
            net = resnet18(in_channels=1, num_classes=1)
        else:
            raise ValueError("Unknown net: %s"%pth)
        net.load_state_dict(checkpoint)
        if torch.cuda.is_available():
            net = net.to('cuda')
        # Put network in eval mode
        net.eval()
        nets.append(net)
    return ModelEnsemble(nets, apply_softmax=apply_softmax)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="Root directory to data")
    parser.add_argument("--saving_dir", required=True, type=str, help="Saving directory")
    parser.add_argument("--preproc", choices=["vbm", "quasi_raw"], default="vbm")
    parser.add_argument("--pb", required=True, choices=["age", "sex", "scz", "bipolar", "asd"])
    parser.add_argument("--saliency_meth", default="occ", choices=["occ", "gradient"])
    parser.add_argument("--mask", default="reduced", choices=["none", "std", "reduced"])
    parser.add_argument("--scaler", default="standard", choices=["standard", "zscore"])
    parser.add_argument("--dl", action="store_true", help="Whether the testing model is Pytorch or sklearn-Like")
    parser.add_argument("--chkpt", type=str, nargs='+', required=True, help="Checkpoint to load (as a .pkl or .pth)")
    args = parser.parse_args()

    logger = logging.getLogger("SMLvsDL")

    root = args.root
    brain_atlas = (os.path.join(root, "AAL3v1_1mm.nii.gz"),
                   os.path.join(root, "AAL3v1_1mm.nii.txt"))
    preproc = args.preproc
    pb = args.pb
    scaler = StandardScaler if args.scaler == "standard" else Zscore

    # Loads Mask (only for sklearn model)
    mask = nibabel.load(os.path.join(root, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
    mask = (mask.get_fdata() != 0)

    # Loads data
    if args.dl:
        area_masks = get_brain_area_masks((121, 145, 121), brain_atlas[0], brain_atlas[1],
                                          transforms=(Compose([Crop((121, 128, 121)),
                                                               Padding([128, 128, 128])])))
        input_transforms = Compose([Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'),
                                    Normalize()])
        test_dataset, test_intra_dataset = (None, None)
        if pb in ["age", "sex"]:
            test_dataset = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="test", target=pb,
                                      transforms=input_transforms)
            test_intra_dataset = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="test_intra", target=pb,
                                            transforms=input_transforms)
        elif pb in ["scz", "bipolar", "asd"]:
            dataset_cls = eval("%sDataset"%(pb.upper() if pb in ["scz", "asd"] else pb.capitalize()))
            test_dataset = dataset_cls(root, preproc=preproc, split="test", target="diagnosis",
                                      transforms=input_transforms)
            test_intra_dataset = dataset_cls(root, preproc=preproc, split="test_intra", target="diagnosis",
                                            transforms=input_transforms)

        test = DataLoader(test_dataset, batch_size=2, num_workers=5, sampler=SequentialSampler(test_dataset),
                          collate_fn=OpenBHBDataManager.collate_fn)
        test_intra = DataLoader(test_intra_dataset, batch_size=16, num_workers=5,
                                sampler=SequentialSampler(test_intra_dataset),
                                collate_fn=OpenBHBDataManager.collate_fn)
    else:
        area_masks = get_brain_area_masks((121, 145, 121), brain_atlas[0], brain_atlas[1],
                                          transforms=lambda x: x[mask])

        if pb in ["age", "sex"]:
            test = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="test", target=pb)
            test_intra = OpenBHB(root, preproc=preproc, scheme="train_val_test", split="test_intra", target=pb)
        elif pb in ["scz", "bipolar", "asd"]:
            dataset_cls = eval("%sDataset" % (pb.upper() if pb in ["scz", "asd"] else pb.capitalize()))
            test = dataset_cls(root, preproc=preproc, split="test", target="diagnosis")
            test_intra = dataset_cls(root, preproc=preproc, split="test_intra", target="diagnosis")

        test_data, target_test = test.get_data(mask=mask, dtype=np.float32)
        test_intra_data, target_test_intra = test_intra.get_data(mask=mask, dtype=np.float32)
        # Just for saliency maps, computes the statistics of test set (no prediction)
        test_data = scaler().fit_transform(test_data)
        test_intra_data = scaler().fit_transform(test_intra_data)
        test, test_intra = [test_data], [test_intra_data]

    # Loads the model
    model = None
    if args.dl:
        model = load_dl_model(args.chkpt, apply_softmax=(pb!="age" and args.saliency_meth != "gradient"))
    else:
        model = get_pickle_obj(args.chkpt).get("model")
    if model is None:
        raise ValueError("Model at %s is None"%args.chkpt)

    if args.dl: assert isinstance(model, torch.nn.Module), "Unknown Torch model %s"%type(model).__name__
    else: assert is_regressor(model) or is_classifier(model), "Unknown sklearn model %s"%type(model).__name__
    # Puts the model in evaluation mode (no dropout, batchnorm, etc.)
    model.eval()

    saliency_maps = dict(inter=[], intra=[])
    test_names = ["inter", "intra"]
    for t, t_name in zip([test, test_intra], test_names):
        bar = tqdm(desc="Test %s"%t_name, total=len(test))
        for i, X in enumerate(t):
            if args.dl:
                if args.saliency_meth == "occ":
                    saliency_imgs = area_occlusion(model, X.inputs, area_masks,
                                                   apply_softmax=False,
                                                   is_classif=(args.pb != "age"),
                                                   cuda=torch.cuda.is_available())
                    saliency_maps[t_name].extend(saliency_imgs.astype(np.float32))

                elif args.saliency_meth == "gradient":
                    try:
                        saliency_imgs = sensitivity_analysis(model, X.inputs,
                                                             postprocess="abs",
                                                             is_classif=(args.pb != "age"),
                                                             apply_softmax=False,
                                                             cuda=torch.cuda.is_available())
                    except ValueError:
                        print("All gradients are zeros for batch %i (test %s)"%(i, t_name))
                        exit(0)
                    saliency_maps[t_name].extend(saliency_imgs.astype(np.float32))

            else:
                if args.saliency_meth == "occ":
                    saliency_imgs = area_occlusion(model, X, area_masks,
                                                   sklearn=True,
                                                   is_classif=(args.pb != "age"),
                                                   occlusion_value=0)
                    # Maps back the saliency imgs
                    saliency_imgs_ = np.zeros((len(saliency_imgs), 121, 145, 121), dtype=np.float32)
                    saliency_imgs_[:, mask] = saliency_imgs
                    saliency_maps[t_name].extend(saliency_imgs_)

                elif args.saliency_meth == "gradient":
                    # Only one saliency map corresponding to model's weights
                    saliency_img = sensitivity_analysis(model, X, postprocess="abs", sklearn=True,
                                                        is_classif=(args.pb != "age"))
                    saliency_img_ = np.zeros((1, 121, 145, 121), dtype=np.float32)
                    saliency_img_[:, mask] = saliency_img
                    saliency_maps[t_name] = saliency_img_
            bar.update()

    # Maps back the area masks
    if (not args.dl) and mask is not None:
        for (area, m_) in area_masks.items():
            area_masks[area] = np.zeros((121, 145, 121), dtype=np.bool)
            area_masks[area][mask] = m_
    for t_name in test_names:
        logger.info("Most Important Regions: {}".
                    format(get_relevance_per_area(area_masks, np.mean(saliency_maps[t_name], axis=0))[:10]))
    chk = args.chkpt
    if type(chk) == list: chk = chk[0]
    saving_file = os.path.join(os.path.dirname(chk), "saliency_maps_%s_%s.pkl"%(args.saliency_meth,
                               os.path.basename(os.path.splitext(chk)[0])))

    MLTrainer.save(dict(saliency_map={t_name: {'avg': np.mean(saliency_maps[t_name], axis=0),
                                               'std': np.std(saliency_maps[t_name], axis=0),
                                               'relevances': [get_relevance_per_area(area_masks, s_map)
                                                              for s_map in saliency_maps[t_name]]}
                                      for t_name in test_names},
                        area_masks=area_masks), saving_file)
