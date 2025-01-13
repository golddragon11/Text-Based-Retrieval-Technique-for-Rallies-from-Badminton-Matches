import collections
import numpy as np
import torch
import tqdm
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.utils.utils import sim_matrix


class MyTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.writer = None
        self.step = 0

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.train_metric = metric_ftns
        self.valid_metric = metric_ftns

        self.min_loss = float('inf')
        self.min_loss_epoch = 0
        self.min_val_loss = float('inf')
        self.min_val_loss_epoch = 0
        self.max_accuracy = 0
        self.max_accuracy_epoch = 0
        self.max_precision = 0
        self.max_precision_epoch = 0

        self.softmax = torch.nn.Softmax(dim=1)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.cuda.empty_cache()

        self.model.train()
        for met in self.train_metric:
            met.reset()
        total_loss = 0
        total_iter = 0

        for batch_idx, data in enumerate(self.data_loader):
            for field in ['video_token', 'tracknet_token', 'video_mask', 'tracknet_mask', 'label']:
                if field in data:
                    data[field] = data[field].to(self.device)

            output = self.model(data)
            target = data['label']
            loss = self.loss(output, target)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.detach().item()
            total_iter += 1

            for met in self.train_metric:
                met.update(output['log_prob'], torch.argmax(target, dim=1))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                if self.writer is not None:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            self.step += 1
            del data, output, loss
            if batch_idx == self.len_epoch:
                break

        log = {
            'training': {'loss': total_loss / total_iter},
            'validation': {}
        }
        for met in self.train_metric:
            log['training'].update(met.compute())

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log['validation'].update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if log['training']['loss'] < self.min_loss:
            self.min_loss = log['training']['loss']
            self.min_loss_epoch = epoch
        if log['validation']['loss'] < self.min_val_loss:
            self.min_val_loss = log['validation']['loss']
            self.min_val_loss_epoch = epoch
        if log['validation']['accuracy_avg'] > self.max_accuracy:
            self.max_accuracy = log['validation']['accuracy_avg']
            self.max_accuracy_epoch = epoch
        if log['validation']['precision_avg'] > self.max_precision:
            self.max_precision = log['validation']['precision_avg']
            self.max_precision_epoch = epoch

        log['best'] = {
            'loss': {'epoch': self.min_loss_epoch, 'value': self.min_loss},
            'val_loss': {'epoch': self.min_val_loss_epoch, 'value': self.min_val_loss},
            'max_accuracy': {'epoch': self.max_accuracy_epoch, 'value': self.max_accuracy},
            'max_precision': {'epoch': self.max_precision_epoch, 'value': self.max_precision}
        }

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        torch.cuda.empty_cache()
        self.model.eval()
        for met in self.valid_metric:
            met.reset()

        val_loss = 0
        val_loss_detailed = collections.defaultdict(lambda: 0)

        with torch.no_grad():
            for data in tqdm.tqdm(self.valid_data_loader, desc='Validation', leave=False):
                for field in ['video_token', 'tracknet_token', 'video_mask', 'tracknet_mask', 'label']:
                    if field in data:
                        data[field] = data[field].to(self.device)

                output = self.model(data)
                target = data['label']

                for met in self.valid_metric:
                    met.update(output['log_prob'], torch.argmax(target, dim=1))

                if self.loss is not None:
                    loss = self.loss(output, target)
                    val_loss += loss.item()
                    del loss

        val_loss = val_loss / len(self.valid_data_loader)
        val_loss_detailed = {loss_name: loss_value / len(self.valid_data_loader) for loss_name, loss_value in val_loss_detailed.items()}

        log = {}
        log.update({'loss': val_loss})
        for met in self.valid_metric:
            log.update(met.compute())


        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _move_to_device(self, data, device):
        if torch.is_tensor(data):
            return data.to(device)
        else:
            return {key: val.to(device) for key, val in data.items()}


class MyTrainer2(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, loss, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.writer = None
        self.step = 0

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.train_metric = metric_ftns
        self.valid_metric = metric_ftns

        self.min_loss = float('inf')
        self.min_loss_epoch = 0
        self.min_val_loss = float('inf')
        self.min_val_loss_epoch = 0
        self.max_accuracy = 0
        self.max_accuracy_epoch = 0
        self.max_precision = 0
        self.max_precision_epoch = 0

        self.softmax = torch.nn.Softmax(dim=1)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        torch.cuda.empty_cache()

        self.model.train()
        total_loss = 0
        total_iter = 0

        for batch_idx, data in enumerate(self.data_loader):
            for field in ['data_token', 'text_token', 'text_mask', 'description_token', 'description_mask', 'class', 'eval_class']:
                if field in data:
                    data[field] = data[field].to(self.device)

            output = self.model(data)
            loss = self.loss(output)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.detach().item()
            total_iter += 1

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                if self.writer is not None:
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            self.step += 1
            del data, output, loss
            if batch_idx == self.len_epoch:
                break

        log = {
            'training': {'loss': total_loss / total_iter},
            'validation': {}
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log['validation'].update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        torch.cuda.empty_cache()
        self.model.eval()
        self.valid_metrics.reset()

        metrics = {}
        val_loss = 0
        val_loss_detailed = collections.defaultdict(lambda: 0)
        embed_arr = collections.defaultdict(lambda: [])
        class_arr = []
        eval_class_arr = []

        with torch.no_grad():
            for data in tqdm.tqdm(self.valid_data_loader, desc='Validation', leave=False):
                class_arr.append(data['meta']['class'])
                eval_class_arr.append(data['meta']['eval_class'])

                for field in ['text_token', 'text_mask', 'description_token', 'description_mask', 'data_token', 'class', 'eval_class']:
                    if field in data:
                        data[field] = data[field].to(self.device)

                embeds = self.model(data)
                for name, embed in embeds.items():
                    if '_embed' in name:
                        embed_arr[name].append(embed)

                if self.loss is not None:
                    loss = self.loss(embeds)
                    val_loss += loss.item()
                    del loss
                del data, embeds

        for name, embed in embed_arr.items():
            embed_arr[name] = torch.cat(embed, dim=0)

        val_loss = val_loss / len(self.valid_data_loader)
        val_loss_detailed = {loss_name: loss_value / len(self.valid_data_loader) for loss_name, loss_value in val_loss_detailed.items()}

        sims = {}
        sims['t2d'] = sim_matrix(embed_arr['text_embed'], embed_arr['data_embed']).detach().cpu().numpy()

        eval_class_arr = torch.cat(eval_class_arr)
        class_arr = torch.cat(class_arr)
        for metric in self.metric_ftns:
            metric_name = metric.__name__
            res = metric(sims, class_arr, eval_class_arr)
            metrics[metric_name] = res

        log = {}
        log.update({'loss': val_loss})
        log.update({'metrics': metrics})

        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _move_to_device(self, data, device):
        if torch.is_tensor(data):
            return data.to(device)
        else:
            return {key: val.to(device) for key, val in data.items()}


def stage1_inference(model, device, batch):
    torch.cuda.empty_cache()
    model.eval()

    for field in ['video_token', 'tracknet_token', 'video_mask', 'tracknet_mask']:
        if field in batch:
            batch[field] = torch.stack(batch[field], dim=0).to(device)

    with torch.no_grad():
        output = model(batch)

    return torch.argmax(output['prob'], dim=1)


def stage2_inference(model, device, batch):
    torch.cuda.empty_cache()
    model.eval()

    for field in ['text_token', 'text_mask', 'description_token', 'description_mask', 'data_token']:
        if field in batch and batch[field] is not None:
            batch[field] = torch.stack(batch[field], dim=0).to(device)

    with torch.no_grad():
        embeds = model(batch)

    return embeds
