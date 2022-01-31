from dl_training.core import Base
from dl_training.datamanager import OpenBHBDataManager, BHBDataManager, ClinicalDataManager
from dl_training.self_supervision.sim_clr import SimCLR
from dl_training.models.resnet import *
from dl_training.models.densenet import *
from dl_training.losses import *
from dl_training.models.sfcn import SFCN
from dl_training.models.alexnet import AlexNet3D_Dropout
import nibabel, os


class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.pb, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(self.args)
        self.metrics = BaseTrainer.build_metrics(self.args.pb)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=args.gamma_scheduler,
                                                         step_size=args.step_size_scheduler)
        model_cls = SimCLR if args.pb == "self_supervised" else Base
        self.kwargs_train = dict()

        self.model = model_cls(model=self.net,
                               metrics=self.metrics,
                               pretrained=args.pretrained_path,
                               load_optimizer=args.load_optimizer,
                               use_cuda=args.cuda,
                               loss=self.loss,
                               optimizer=self.optimizer)

    def run(self):
        with_validation = True
        train_history, valid_history = self.model.training(self.manager,
                                                           nb_epochs=self.args.nb_epochs,
                                                           scheduler=self.scheduler,
                                                           with_validation=with_validation,
                                                           checkpointdir=self.args.checkpoint_dir,
                                                           nb_epochs_per_saving=self.args.nb_epochs_per_saving,
                                                           exp_name=self.args.exp_name,
                                                           fold_index=self.args.folds,
                                                           **self.kwargs_train)

        return train_history, valid_history

    @staticmethod
    def build_metrics(pb):
        if pb in ["scz", "bipolar", "asd", "sex"]:
            metrics = ["balanced_accuracy", "roc_auc"]
        elif pb == "age":
            metrics = ["RMSE"]
        elif pb == "self_supervised":
            metrics = ["accuracy"] # for SimCLR, accuracy to retrieve the original views from same image
        else:
            raise NotImplementedError("Unknown pb: %s"%pb)
        return metrics

    @staticmethod
    def build_loss(args):
        # Binary classification tasks
        if args.pb in ["scz", "bipolar", "asd", "sex"]:
            # Balanced BCE loss
            pos_weights = {"scz": 1.131, "asd": 1.584, "bipolar": 1.584, "sex": 1.0}
            loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights[args.pb], dtype=torch.float32,
                                                                device=('cuda' if args.cuda else 'cpu')))
        # Regression task
        elif args.pb == "age":
            loss = nn.L1Loss()
        # Self-supervised task
        elif args.pb == "self_supervised":
            ## Default value for sigma == 5
            loss = WeaklySupervisedNTXenLoss(temperature=0.1, kernel="rbf", sigma=args.sigma, return_logits=True)
        else:
            raise ValueError("Unknown problem: %s"%args.pb)
        return loss

    @staticmethod
    def build_network(name, pb, **kwargs):
        num_classes = 1 # one output for BCE loss and L1 loss. Last layers removed for self-supervision
        if name == "resnet18":
            if pb == "self_supervised":
                net = resnet18(out_block="simCLR")
            else:
                net = resnet18(num_classes=num_classes, **kwargs)
        elif name == "sfcn":
            if pb == "self_supervised":
                raise NotImplementedError()
            else:
                net = SFCN(output_dim=num_classes, dropout=True, **kwargs)
                logger.warning('By default, dropout=True for SFCN.')
        elif name == "densenet121":
            if pb == "self_supervised":
                net = densenet121(num_classes=num_classes, out_block="simCLR", **kwargs)
            else:
                net = densenet121(num_classes=num_classes, **kwargs)
        elif name == "alexnet": # AlexNet 3D version derived from Abrol et al., 2021
            if pb == "self_supervised":
                raise NotImplementedError()
            else:
                net = AlexNet3D_Dropout(num_classes=num_classes)
        else:
            raise ValueError('Unknown network %s' % name)

        return net

    @staticmethod
    def build_data_manager(args):
        if args.pb in ["scz", "bipolar", "asd"]:
            labels = ["diagnosis"]
        elif args.pb == "self_supervised": # introduce age to improve the representation
            labels = ["age"]
        else:
            labels = [args.pb] # either "age" or "sex"

        preproc = args.preproc
        try:
            if preproc == "vbm":
                mask = nibabel.load(os.path.join(args.root, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
            else:
                mask = nibabel.load(os.path.join(args.root, "mni_raw_brain-mask_1.5mm.nii.gz"))
        except FileNotFoundError:
            raise FileNotFoundError("Brain masks not found. You can find them in /masks directory "
                                    "and mv them to this directory: %s"%args.root)

        mask = (mask.get_data() != 0)
        _manager_cls = None
        if args.pb in ["age", "sex"]:
            if args.N_train_max <= 5000:
                _manager_cls = OpenBHBDataManager
            else:
                args.N_train_max = None
                _manager_cls = BHBDataManager
        elif args.pb == "self_supervised":
            _manager_cls = OpenBHBDataManager
        elif args.pb in ["scz", "bipolar", "asd"]:
            _manager_cls = ClinicalDataManager

        kwargs_manager = dict(labels=labels, sampler=args.sampler, batch_size=args.batch_size,
                              residualize=args.residualize, mask=mask, number_of_folds=args.nb_folds,
                              N_train_max=args.N_train_max, device=('cuda' if args.cuda else 'cpu'),
                              num_workers=args.num_cpu_workers, pin_memory=True, drop_last=False)

        if args.pb in ["age", "sex", "self_supervised"]:
            kwargs_manager["model"] = "SimCLR" if args.pb == "self_supervised" else "base"
        elif args.pb in ["scz", "bipolar", "asd"]:
            kwargs_manager["db"] = args.pb

        manager = _manager_cls(args.root, preproc, **kwargs_manager)

        return manager
