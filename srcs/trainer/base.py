import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from abc import abstractmethod, ABCMeta
from pathlib import Path
from shutil import copyfile
from numpy import inf
import time
from datetime import datetime
from srcs.utils.util import write_conf, is_master, get_logger
from srcs.logger import TensorboardWriter, EpochMetrics
import os
from os.path import join as opj

class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = get_logger('trainer')

        self.device = config.local_rank if config.n_gpu > 1 else 0
        self.model = model.to(self.device)

        if config.n_gpu > 1:
            # multi GPU
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = DistributedDataParallel(
                model, device_ids=[self.device], output_device=self.device)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer.get('epochs', int(1e10))
        if self.epochs is None:
            self.epochs = int(1e10)
        self.logging_step = cfg_trainer.get('logging_step', 100)

        # setup metric monitoring for monitoring model performance and saving best-checkpoint
        self.monitor = cfg_trainer.get('monitor', 'off')

        metric_names = ['loss'] + [met.__name__ for met in self.metric_ftns]
        self.ep_metrics = EpochMetrics(metric_names, phases=(
            'train', 'valid'), monitoring=self.monitor)

        self.saving_top_k = cfg_trainer.get('saving_top_k', -1)
        self.landmark_list = cfg_trainer.get('landmark_list', [])
        self.early_stop = cfg_trainer.get('early_stop', inf)
        if self.early_stop is None:
            self.early_stop = inf
        self.final_test = cfg_trainer.get('final_test', False)

        write_conf(self.config, 'config.yaml')

        self.start_epoch = 1
        self.checkpt_dir = Path(self.config.checkpoint_dir)
        log_dir = Path(self.config.log_dir)
        if self.final_test:
            self.final_test_dir = Path(self.config.final_test_dir)
        if is_master():
            self.checkpt_dir.mkdir(exist_ok=True)
            # setup visualization writer instance
            log_dir.mkdir(exist_ok=True)
            if self.final_test:
                self.final_test_dir.mkdir(exist_ok=True)
            self.writer = TensorboardWriter(
                log_dir, cfg_trainer['tensorboard'])
        else:
            self.writer = TensorboardWriter(log_dir, False)

        if config.resume is not None:
            resume_conf = config.get(
                'resume_conf', None)
            if resume_conf is None:
                resume_conf = ['epoch', 'optimizer']
            self._resume_checkpoint(config.resume, resume_conf)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    def _test_epoch(self):
        """
        Final test logic after the training (! test the latest checkpoint)

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.logger.info(f"\n⏩⏩ Start Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ⏩⏩\n")
        not_improved_count = 0
        train_start = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            result = self._train_epoch(epoch)
            self.ep_metrics.update(epoch, result)

            # print result metrics of this epoch
            max_line_width = max(len(line)
                                 for line in str(self.ep_metrics).splitlines())
            # divider ---
            self.logger.info('-' * max_line_width)
            self.logger.info('\n' + str(self.ep_metrics.latest()) + '\n')

            if is_master():
                # check if model performance improved or not, for early stopping and topk saving
                is_best = False
                improved = self.ep_metrics.is_improved()
                if improved:
                    not_improved_count = 0
                    is_best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    if self.final_test:
                        self.logger.info(
                            f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🎉🎉\n\n == = > Start Testing(Using Latest Checkpoint): \n")
                        self._test_epoch()
                    else:
                        self.logger.info(
                            f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}🎉🎉\n\n")
                    exit(1)

                using_topk_save = self.saving_top_k > 0
                self._save_checkpoint(
                    epoch, save_best=is_best, save_latest=using_topk_save, landmark_list=self.landmark_list)
                # keep top-k checkpoints only, using monitoring metrics
                if using_topk_save:
                    self.ep_metrics.keep_topk_checkpt(
                        self.checkpt_dir, self.saving_top_k)

                self.ep_metrics.to_csv('epoch-results.csv')

            epoch_end = time.time()
            self.logger.info(
                f'🕒 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Epoch Time Cost: {epoch_end-epoch_start:.2f}s, Total Time Cost: {(epoch_end-train_start)/3600:.2f}h\n')
            self.logger.info('=' * max_line_width)
            if self.config.n_gpu > 1:
                dist.barrier()
        if self.final_test:
            self.logger.info(
                f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 🎉🎉\n\n == = > Start Testing(Using Latest Checkpoint): \n")
            self._test_epoch()
        else:
            self.logger.info(
                f"\n🎉🎉 Finish Training! | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}🎉🎉\n\n")

    def _save_checkpoint(self, epoch, save_best=False, save_latest=True, landmark_list=[]):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, save a copy of current checkpoint file as 'model_best.pth'
        :param save_latest: if True, save a copy of current checkpoint file as 'model_latest.pth'
        :param landmark_list: save and keep current checkpoints if current epoch is in this list 
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'epoch_metrics': self.ep_metrics, # may cause can't pickle error in torch.save
            'config': self.config
        }

        filename = str(self.checkpt_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(
            f"💾 Model checkpoint saved at: \n    {os.getcwd()}/{filename}")
        if save_latest:
            latest_path = str(self.checkpt_dir / 'model_latest.pth')
            copyfile(filename, latest_path)
        if save_best:
            best_path = str(self.checkpt_dir / 'model_best.pth')
            copyfile(filename, best_path)
            self.logger.info(
                f"🔄 Renewing best checkpoint!")
        if landmark_list and epoch in landmark_list:
            landmark_path = str(
                self.checkpt_dir / f'model_epoch{epoch}.pth')
            copyfile(filename, landmark_path)
            self.logger.info(
                f"💡 Saving landmark checkpoint at epoch {epoch}!")

    def _resume_checkpoint(self, resume_path, resume_conf=['epoch','optimizer']):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        :param resume_conf: resume config that controls what to resume
        """

        resume_path = opj(os.getcwd(), self.config['resume'])
        self.logger.info(f"💡 Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['config'].get('arch', None) != self.config.get('arch', None):
            self.logger.warning("⚠️ Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint
        if 'optimizer' in resume_conf:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(
                f'📣 Optimizer resumed from the loaded checkpoint!')

        # epoch start point
        if 'epoch' in resume_conf:
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(
                f"📣 Epoch index resumed to epoch ({checkpoint['epoch']}).")
        else:
            self.start_epoch = 1
            self.logger.info(
                f"📣 Epoch index renumbered from epoch (1).")

