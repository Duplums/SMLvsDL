from dl_training.core import Base
from dl_training.datamanager import OpenBHBDataManager, BHBDataManager, ClinicalDataManager
from dl_training.self_supervision.sim_clr import SimCLR
from dl_training.models.resnet import *
from dl_training.models.densenet import *
from dl_training.losses import *
from dl_training.models.sfcn import SFCN
from dl_training.models.alexnet import AlexNet3D_Dropout
import re, nibabel, os


class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.num_classes, args, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(args.loss, net=self.net, args=self.args)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, gamma=args.gamma_scheduler,
                                                         step_size=args.step_size_scheduler)
        model_cls = Base
        self.kwargs_train = dict()


        if args.model == "SimCLR":
            model_cls = SimCLR

        self.model = model_cls(model=self.net,
                               metrics=args.metrics,
                               pretrained=args.pretrained_path,
                               freeze_until_layer=args.freeze_until_layer,
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
                                                           epoch_index=self.args.start_from,
                                                           standard_optim=getattr(self.net, 'std_optim', True),
                                                           gpu_time_profiling=self.args.profile_gpu,
                                                           **self.kwargs_train)

        return train_history, valid_history

    @staticmethod
    def build_loss(name, args=None):
        if name == 'l1':
            loss = nn.L1Loss()
        elif name == 'BCE':
            pos_weight = None
            if args.loss_param is not None:
                pos_weight = torch.tensor(args.loss_param, dtype=torch.float32, device=('cuda' if args.cuda else 'cpu'))
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif name == "GeneralizedSupervisedNTXenLoss": ## Default value for sigma == 5
            if args.loss_param == 0:
                loss = GeneralizedSupervisedNTXenLoss(temperature=0.1, kernel='discrete', sigma=args.loss_param,
                                                      return_logits=True)
            else:
                kernel = args.loss_kwargs or "rbf"
                if isinstance(kernel, list): kernel = kernel[0]
                loss = GeneralizedSupervisedNTXenLoss(temperature=0.1, kernel=kernel, sigma=args.loss_param or 5,
                                                      return_logits=True)
        else:
            raise ValueError("Loss not yet implemented")
        return loss

    @staticmethod
    def build_network(name, num_classes, args, **kwargs):
        if name == "resnet18":
            net = resnet18(pretrained=False, num_classes=num_classes, concrete_dropout=args.concrete_dropout,
                           dropout_rate=args.dropout, **kwargs)
        elif name == "resnet18_block4":
            net = resnet18(pretrained=False, num_classes=num_classes, concrete_dropout=args.concrete_dropout,
                           dropout_rate=args.dropout, out_block="block4", **kwargs)
        elif name == "sfcn":
            net = SFCN(output_dim=num_classes, dropout=True, **kwargs)
            logger.warning('By default, dropout=True for SFCN.')
        elif name == "densenet121":
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, **kwargs)
        elif name in ['densenet121_block4', 'densenet121_simCLR', 'densenet121_sup_simCLR']:
            block = re.search('densenet121_(\w+)', name)[1]
            net = densenet121(progress=False, num_classes=num_classes, drop_rate=args.dropout, bayesian=args.bayesian,
                              concrete_dropout=args.concrete_dropout, out_block=block, **kwargs)
        elif name == "dl1_abrol":
            net = AlexNet3D_Dropout(num_classes=num_classes)
        elif name == "dl1_abrol_block4":
            net = AlexNet3D_Dropout(num_classes=num_classes, return_features=True)
        else:
            raise ValueError('Unknown network %s' % name)

        return net

    @staticmethod
    def build_data_manager(args):
        labels = args.labels or []
        scheme = "cv" if args.cv else "train_val_test"
        preproc = "vbm" if args.preproc == "cat12" else args.preproc
        if preproc == "vbm":
            mask = nibabel.load(os.path.join(args.root_data, "mni_cerebrum-gm-mask_1.5mm.nii.gz"))
        else:
            mask = nibabel.load(os.path.join(args.root_data, "mni_raw_brain-mask_1.5mm.nii.gz"))
        mask = (mask.get_data() != 0)
        _manager_cls = None
        if args.db =="open_bhb":
            _manager_cls = OpenBHBDataManager
        elif args.db == "bhb":
            _manager_cls = BHBDataManager
        elif re.match(r"clinical_\w*", args.db):
            _manager_cls = ClinicalDataManager

        kwargs_manager = dict(labels=labels, sampler=args.sampler, batch_size=args.batch_size,
                              residualize=args.residualize, mask=mask, number_of_folds=args.nb_folds,
                              N_train_max=args.N_train_max, device=('cuda' if args.cuda else 'cpu'),
                              num_workers=args.num_cpu_workers, pin_memory=args.pin_mem, drop_last=args.drop_last)

        if args.db in ["open_bhb", "bhb"]:
            kwargs_manager["scheme"] = scheme
            kwargs_manager["model"] = args.model
        else:
            kwargs_manager["db"] = re.match(r"clinical_(\w*)", args.db)[1]

        manager = _manager_cls(args.root_data, preproc, **kwargs_manager)

        return manager
