import os
import pickle
import logging
from dl_training.utils import get_chk_name
from dl_training.core import Base
from dl_training.training import BaseTrainer
from dl_training.transforms import *


class BaseTester():

    def __init__(self, args):
        self.args = args
        self.net = BaseTrainer.build_network(args.net, args.pb, in_channels=1)
        self.manager = BaseTrainer.build_data_manager(args)
        self.loss = BaseTrainer.build_loss(args)
        self.logger = logging.getLogger("SMLvsDL")
        self.metrics = BaseTrainer.build_metrics(args.pb)
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
                             metrics=self.metrics,
                             pretrained=pretrained_path,
                             use_cuda=self.args.cuda)
                model.testing(self.manager.get_dataloader(test=True, fold_index=fold).test,
                              saving_dir=self.args.checkpoint_dir, exp_name=exp_name, **self.kwargs_test)
    
    def get_folds_to_test(self):
        if self.args.folds is not None and len(self.args.folds) > 0:
            folds = self.args.folds
        else:
            folds = list(range(self.args.nb_folds))
        return folds

    def get_epochs_to_test(self):
        # Get the last point and tests it, for each fold
        epochs_tested = [[self.args.nb_epochs - 1] for _ in range(self.args.nb_folds)]

        return epochs_tested


class OpenBHBTester(BaseTester):
    """
    We perform 2 kind of tests:
    * Test-Intra where we tests on the left-out tests set with intra-site images
    * Test-Inter where we tests on inter-site images (never-seen site)
    """
    def run(self):
        epochs_tested = self.get_epochs_to_test()
        folds_to_test = self.get_folds_to_test()
        for fold in folds_to_test:
            tests = ["", "Intra_"]
            (residualizer, Zres) = (None, None)
            if self.manager.residualize is not None:
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
                                 metrics=self.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    model.testing(loader.test, saving_dir=self.args.checkpoint_dir, exp_name=exp_name,
                                  **self.kwargs_test)


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
                                 metrics=self.metrics,
                                 pretrained=pretrained_path,
                                 use_cuda=self.args.cuda)
                    y, y_true,_,_,_ = model.test(self.manager.get_dataloader(test=True).test)
                    Y.append(y)
                    Y_true.append(y_true)
                with open(os.path.join(self.args.checkpoint_dir, exp_name), 'wb') as f:
                    pickle.dump({"y": np.array(Y).swapaxes(0,1), "y_true": np.array(Y_true).swapaxes(0,1)}, f)
