import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import platform
from omegaconf import OmegaConf
from .base import BaseTrainer
from srcs.utils.util import collect, instantiate, get_logger
from srcs.logger import BatchMetrics

#======================================
# Trainer: modify '_train_epoch'
#======================================


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.limit_train_iters = config['trainer'].get(
            'limit_train_iters', len(self.data_loader))
        if not self.limit_train_iters or self.limit_train_iters > len(self.data_loader):
            self.limit_train_iters = len(self.data_loader)
        self.limit_valid_iters = config['trainer'].get(
            'limit_valid_iters', len(self.valid_data_loader))
        if not self.limit_valid_iters or self.limit_valid_iters > len(self.valid_data_loader):
            self.limit_valid_iters = len(self.valid_data_loader)
        self.log_weight = config['trainer'].get('log_weight', False)
        args = ['loss', *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = BatchMetrics(
            *args, postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics(
            *args, postfix='/valid', writer=self.writer)
        self.losses = self.config['loss']

    def _ce_reblur(self, output):
        # frame_n should equal to ce_code_n cases
        ce_weight = self.model.BlurNet.ce_weight.detach().squeeze()
        ce_code = ((torch.sign(ce_weight)+1)/2)
        ce_code_ = torch.tensor(ce_code).view(1, -1, 1, 1, 1)
        ce_output = torch.sum(torch.mul(output, ce_code_), dim=1)/len(ce_code)
        return ce_output
    
    def _after_iter(self, epoch, batch_idx, phase, loss, metrics, image_tensors: dict):
        # hook after iter
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_iters') + batch_idx, speed_chk=f'{phase}')

        loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
        getattr(self, f'{phase}_metrics').update('loss', loss_v)

        for k, v in metrics.items():
            getattr(self, f'{phase}_metrics').update(k, v.item()) # `v` is a torch tensor

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        interp_scale = self.model.frame_n//self.model.ce_code_n

        for batch_idx, vid in enumerate(self.data_loader):  # video_dataloader
            
            vid = vid.to(self.device).float()/255 
            target = vid[:,::interp_scale]

            output, data, data_noisy = self.model(vid)
            output_ = torch.flatten(output, end_dim=1)
            target_ = torch.flatten(target, end_dim=1)

            # main loss
            loss = self.losses['main_loss'] * \
                self.criterion['main_loss'](output_, target_)
            
            # reblur loss: frame_n should equal to ce_code_n cases
            if 'reblur_loss' in self.losses:
                ce_output = self._ce_reblur(output)
                loss = loss + self.losses['reblur_loss'] * \
                    self.criterion['reblur_loss'](ce_output, data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # iter record
            if batch_idx % self.logging_step == 0 or (batch_idx+1) == self.limit_train_iters:
                # loss
                loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
                self.writer.set_step(
                    (epoch - 1) * self.limit_train_iters + batch_idx, speed_chk='train')
                self.train_metrics.update('loss', loss_v)
                # iter metrics
                iter_metrics = {}
                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output_, target_))
                    else:
                        # print(output.shape, target.shape)
                        metric_v = met(output_, target_)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                frame_num = output.shape[1]
                image_tensors = {
                    'input': data_noisy[0, ...], 'output': output[0,0::frame_num//4, ...], 'target': target[0,0::frame_num//4, ...]}
                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'train',
                                 loss, iter_metrics, {})
                # iter log
                self.logger.info(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer.param_groups[0]["lr"]:.3e}')

            if (batch_idx+1) == self.limit_train_iters:
                # save demo images to tensorboard after trainig epoch
                self.writer.set_step(epoch)
                for k, v in image_tensors.items():
                    self.writer.add_image(
                        f'train/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                break
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        interp_scale = self.model.frame_n//self.model.ce_code_n
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, vid in enumerate(self.valid_data_loader):
                vid = vid.to(self.device).float()/255
                target = vid[:,::interp_scale]

                # forward
                output, data, data_noisy = self.model(vid)
                output_ = torch.flatten(output, end_dim=1)
                target_ = torch.flatten(target, end_dim=1)

                # main loss
                loss = self.losses['main_loss'] * \
                    self.criterion['main_loss'](output_, target_)
                # reblur loss: frame_n should equal to ce_code_n cases
                if 'reblur_loss' in self.losses:
                    ce_output = self._ce_reblur(output)
                    loss = loss + self.losses['reblur_loss'] *self.criterion['reblur_loss'](ce_output, data)

                # iter metrics
                iter_metrics = {}
                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output_, target_))
                    else:
                        # print(output.shape, target.shape)
                        metric_v = met(output_, target_)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                frame_num = output.shape[1]
                image_tensors = {
                    'input': data_noisy[0, ...], 'output': output[0, 0::frame_num//4, ...], 'target': target[0, 0::frame_num//4, ...]}

                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'valid',
                                 loss, iter_metrics, {})

                if (batch_idx+1) == self.limit_valid_iters:
                    # save demo images to tensorboard after valid epoch
                    self.writer.set_step(epoch)
                    for k, v in image_tensors.items():
                        self.writer.add_image(
                            f'valid/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                    break

        # add histogram of model parameters to the tensorboard
        if self.log_weight:
            for name, p in self.model.BlurNet.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            total = self.data_loader.batch_size * self.limit_train_iters
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.limit_train_iters
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)


#======================================
# Trainning: run Trainer for trainning
#======================================


def trainning(gpus, config):
    # enable access to non-existing keys
    OmegaConf.set_struct(config, False)
    n_gpu = len(gpus)
    config.n_gpu = n_gpu
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    if n_gpu > 1:
        torch.multiprocessing.spawn(
            multi_gpu_train_worker, nprocs=n_gpu, args=(gpus, config))
    else:
        train_worker(config)


def train_worker(config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    logger = get_logger('train')
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure
    model = instantiate(config.arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = {}
    if 'main_loss' in config.loss:
        criterion['main_loss'] = instantiate(config.main_loss)
    if 'reblur_loss' in config.loss:
        criterion['reblur_loss'] = instantiate(
            config.reblur_loss)
    metrics = [instantiate(met) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


def multi_gpu_train_worker(rank, gpus, config):
    """
    Training with multiple GPUs

    Args:
        rank ([type]): [description]
        gpus ([type]): [description]
        config ([type]): [description]

    Raises:
        RuntimeError: [description]
    """
    # initialize training config
    config.local_rank = rank
    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=len(gpus),
        rank=rank)
    torch.cuda.set_device(gpus[rank])

    # start training processes
    train_worker(config)
