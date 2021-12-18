import argparse
from json_config import CONFIG
from pynet.metrics import METRICS
from training import BaseTrainer
from testing import BaseTester, AlphaWGANTester, RobustnessTester, BayesianTester, EnsemblingTester, \
    NNRepresentationTester, SimCLRTester, CVBaseTester, OpenBHBTester
import torch
import logging

if __name__=="__main__":

    logger = logging.getLogger("pynet")

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, required=True, help="Path to data root directory")
    parser.add_argument("--preproc", type=str, default='cat12', choices=['cat12', 'quasi_raw', 'vision',
                                                                         "vbm_roi", "fsl_desikan_roi",
                                                                         "fsl_destrieux_roi"])
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--freeze_until_layer", type=str)
    parser.add_argument("--checkpoint_dir", type=str, default="/neurospin/psy_sbox/bd261576/checkpoints")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--outfile_name", type=str, help="The output file name used to save the results in testing mode.")
    parser.add_argument("--N_train_max", type=int, default=None, help="Maximum number of training samples "
                                                                      "to be used per fold")
    parser.add_argument("--nb_epochs_per_saving", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--input_dim", type=int, help="Input data dimension given to model")
    parser.add_argument("--no_val", action="store_true", help="No validation metrics computed")
    parser.add_argument("--da", type=str, nargs='+', default=[], choices=['flip', 'blur', 'noise', 'resized_crop',
                                                                          'affine', 'ghosting', 'motion', 'spike',
                                                                          'biasfield', 'swap', 'cutout', 'cifar'])
    parser.add_argument("--manual_seed", type=int, help="The manual seed to give to pytorch.")
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate to use thoughout the network. "
                                                                 "Careful, all networks do not have this option.")
    parser.add_argument('--concrete_dropout', action='store_true')
    parser.add_argument("--bayesian", action='store_true', help="Whether to use dropout during test time or not")
    parser.add_argument("--nb_folds", type=int, default=5)
    parser.add_argument("--gamma_scheduler", type=float, required=True)
    parser.add_argument("--step_size_scheduler", type=int, default=10)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--pin_mem", type=bool, default=True)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument("--db", choices=list(CONFIG['db'].keys()), required=True)
    parser.add_argument("--self_supervision", choices=["crop", "flip", "jittering", "gray", "cifar"], type=str, nargs='+')
    parser.add_argument("--cv", action="store_true", help="Cross-Validation mode when test == nn_repr (test set ignored)")
    parser.add_argument("--switch_to_copy", action='store_true', help='If set, switch to the copy of the initial data '
                                                                      '(prevents from concurrent access)')
    parser.add_argument("--modalities", nargs="+", choices=["t1", "dwi", "mnist", "svhn", "colorful_mnist", "cifar",
                                                            "cifar100"], default=["t1"])
    parser.add_argument("--no_missing_mod", action="store_true", help="Flag given to DataManager to avoid images with "
                                                                      "missing modality.")
    parser.add_argument("--residualize", type=str, help="Currently only used for (Open)BHB/Clinical Dataset.")
    parser.add_argument("--sampler", choices=["random", "weighted_random", "sequential"], required=True)
    parser.add_argument("--model", choices=['base', 'SimCLR', 'Genesis', 'BYOL', 'MoCo'], default='base')
    parser.add_argument("--load_data", action="store_true")
    parser.add_argument("--labels", nargs='+', type=str, help="Label(s) to be predicted")
    parser.add_argument("--loss", type=str, required=True)
    parser.add_argument("--loss_param", type=float, nargs='+', help="The hyper-parameter given to the loss. Must be a float.")
    parser.add_argument("--loss_kwargs", type=str, nargs='+', help="The string hyper-parameters given to the loss.")
    parser.add_argument("--folds", nargs='+', type=int, help="Fold indexes to run during the training")
    parser.add_argument("--stratify_label", type=str, help="Label used for the stratification of the train/val split")
    parser.add_argument("--with_visualization", action="store_true")
    parser.add_argument("--metrics", nargs='+', type=str, choices=list(METRICS.keys()), help="Metrics to be computed at each epoch")
    parser.add_argument("--add_input", action="store_true", help="Whether to add the input data to the output "
                                                                 "(e.g for a reconstruction task)")
    parser.add_argument("--num_cpu_workers", type=int, default=0, help="Number of workers assigned to do the "
                                                                       "preprocessing step (used by DataLoader of Pytorch)")
    parser.add_argument("--test_all_epochs", action="store_true")
    parser.add_argument("--test_param", type=float, help="The hyper-parameter given to the test function. Must be a float.")
    parser.add_argument("--test_best_epoch", type=str, choices=list(METRICS.keys()),
                        help="If set, it must be a metric or 'loss' in order to select the best epoch to test")
    parser.add_argument("--net", type=str, help="Network to use")
    parser.add_argument("--lr", type=float, required=True, help="Initial learning rate")
    parser.add_argument("--nb_epochs", type=int)
    parser.add_argument("--load_optimizer", action="store_true", help="If <pretrained_path> is set, loads also the "
                                                                      "optimizer's weigth")
    parser.add_argument("--start_from", type=int, help="Iteration where to restart the counting from during training "
                                                       "(useful when re-starting a training)")
    parser.add_argument("--with_logit", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--profile_gpu", action="store_true", help="Profile the GPU time taken by a model during the training")
    parser.add_argument("--cuda", type=bool, default=True)

    # Kind of tests
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", choices=['basic', 'cv', 'open_bhb', 'robustness', 'ssim', 'MC',
                                           'ensemble', 'nn_repr', 'simclr'],
                        help="What kind of test it will perform.")

    args = parser.parse_args()

    if args.weight_decay is not None:
        CONFIG['optimizer']['Adam']['weight_decay'] = args.weight_decay

    if args.loss_param is not None and len(args.loss_param) == 1:
        args.loss_param = args.loss_param[0]

    if args.test_best_epoch is not None:
        assert args.test_best_epoch in (args.metrics or []), \
            "--test_best_epoch must be chosen in {}".format((args.metrics or []))
        logger.warning("!!WARNING: For {}, it is assumed that the highest score is the best !!".
                       format(args.test_best_epoch))

    if not torch.cuda.is_available():
        args.cuda = False
        logger.warning("cuda is not available and has been disabled.")

    if args.manual_seed:
        torch.manual_seed(args.manual_seed)

    if not args.train and not args.test:
        args.train = True
        logger.info("No mode specify: training mode is set automatically")

    if args.preproc in CONFIG:
        if args.switch_to_copy:
            args.input_path, args.metadata_path = CONFIG[args.preproc]['input_path_copy'], \
                                                  CONFIG[args.preproc]['metadata_path_copy']
        else:
            args.input_path, args.metadata_path = CONFIG[args.preproc]['input_path'], CONFIG[args.preproc][
                'metadata_path']
        logger.info('Path to data: %s\nPath to annotations: %s' % (args.input_path, args.metadata_path))

    if args.train:
        trainer = BaseTrainer(args)
        trainer.run()
        # do not consider the pretrained path anymore since it will be eventually computed automatically
        args.pretrained_path = None

    if args.test == 'basic':
        tester = BaseTester(args)
        tester.run()

    if args.test == 'cv':
        tester = CVBaseTester(args)
        tester.run()

    if args.test == "open_bhb":
        tester = OpenBHBTester(args)
        tester.run()

    if args.test == 'MC':
        tester = BayesianTester(args)
        tester.run()

    if args.test == "ensemble":
        tester = EnsemblingTester(args)
        tester.run()

    if args.test == 'ssim':
        tester = AlphaWGANTester(args)
        tester.run()

    if args.test == 'robustness':
        tester = RobustnessTester(args)
        tester.run()

    if args.test =='nn_repr':
        tester = NNRepresentationTester(args)
        tester.run()

    if args.test == "simclr":
        tester = SimCLRTester(args)
        tester.run()




