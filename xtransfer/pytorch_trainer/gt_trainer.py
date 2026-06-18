import random
import  numpy as np
import torch
from tqdm import tqdm
import os
from .model_summary import ModelSummary
from utils.meters import AverageMeter, accuracy
from utils.tools import set_random_seed


try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class gt_Trainer():
    def __init__(
            self,
            seed=0,
            gpu_id=0,
            epochs=2,
            checkpoint_callback=None,
            early_stop_callback=None,
            logger=None,
            use_amp=False,
            val_percent=1.0,
            test_percent=1.0,
    ):
        self.seed = seed
        self.gpu_id = gpu_id
        self.epochs = epochs
        self.checkpoint_callback = checkpoint_callback
        self.early_stop_callback = early_stop_callback
        self.logger = logger
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.current_epoch = 0
        self.scheduler = None

        self.use_amp = False
        if use_amp:
            if not APEX_AVAILABLE:
                self.use_amp = False
                print("apex is not installed")
            else:
                self.use_amp = True

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{gpu_id}" if self.use_gpu else "cpu")

    def fit(self, model, checkpoint=None):
        print(ModelSummary(model, mode='top'))

        self.model = model
        self.model.trainer = self
        optimizer = self.model.configure_optimizers()
        if isinstance(optimizer, tuple):
            self.optimizer = optimizer[0]
            self.scheduler = optimizer[1]
        else:
            self.optimizer = optimizer
            self.scheduler = None

        self.model.to(self.device)
        self.__load_checkpoint(checkpoint, self.model, self.optimizer)
        if self.use_amp:
            self.model, self.optimizer = self.model.configure_apex(amp, self.model, self.optimizer, "O1")
        self.model.train()
        dataloader = model.train_dataloader()
        samples = len(dataloader.dataset)
        batch_size = dataloader.batch_size

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.model.on_epoch_start(epoch)
            with tqdm(total=samples) as pbar:
                pbar.set_description(f"Epoch [{epoch + 1}/{self.epochs}]")

                loss_meter = AverageMeter()
                acc_meter = AverageMeter()
                for i, batch in enumerate(dataloader):
                    if self.use_gpu:
                        batch = self.transfer_batch_to_gpu(batch)
                    output = self.model.training_step(batch, i)
                    loss_meter.update(output['loss'].item())
                    acc_meter.update(output['accuracy'].item())
                    self.model.backward(output['loss'], self.optimizer, self.use_amp)
                    self.model.optimizer_step(self.optimizer)

                    processed = min((i + 1) * batch_size, samples)
                    pbar.n = processed


                self.logger.write('train_loss',loss_meter.avg)
                self.logger.write('train_accuracy', acc_meter.avg)
                print('train loss',loss_meter.avg)
                print('train acc', acc_meter.avg)
            if self.val_percent > 0.0:
                val_acc = self.validate(self.model)
                print('val_acc', val_acc)
            else:
                print("Skipping validation")
            self.model.on_epoch_end(epoch)

    @torch.no_grad()
    def validate(self, model, fast_validate=False, checkpoint=None):
        model.trainer = self
        model.to(self.device)
        self.__load_checkpoint(checkpoint, model)
        model.eval()
        dataloader = model.val_dataloader()
        batch_size = dataloader.batch_size
        if fast_validate:
            samples = min(2 * batch_size, int(len(dataloader.dataset)))
            max_batches = 2
        else:
            samples = int(len(dataloader.dataset) * self.val_percent)
            max_batches = int(len(dataloader) * self.val_percent)

        description = 'Check validation step' if fast_validate else 'Validation'
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        with tqdm(total=samples) as pbar:

            for i, batch in enumerate(dataloader):
                pbar.set_description(description)

                if self.use_gpu:
                    batch = self.transfer_batch_to_gpu(batch)
                output = model.validation_step(batch, i)
                loss_meter.update(output['val_loss'].item())
                acc_meter.update(output['val_accuracy'].item())
                processed = min((i + 1) * batch_size, samples)
                pbar.n = processed

                if i >= max_batches:
                    break
        self.logger.write('val_loss', loss_meter.avg)
        self.logger.write('val_accuracy', acc_meter.avg)
        model.train()
        return acc_meter.avg

    @torch.no_grad()
    def test(self, model, checkpoint=None):
        model.trainer = self
        model.to(self.device)
        self.__load_checkpoint(checkpoint, model)
        model.eval()
        dataloader = model.test_dataloader()
        samples = int(len(dataloader.dataset) * self.test_percent)
        batch_size = dataloader.batch_size
        max_batches = int(len(dataloader) * self.test_percent)

        outputs = []
        with tqdm(total=samples) as pbar:
            for i, batch in enumerate(dataloader):
                pbar.set_description("Test")
                if self.use_gpu:
                    batch = self.transfer_batch_to_gpu(batch)
                output = model.test_step(batch, i)
                outputs.append(output)
                processed = min((i + 1) * batch_size, samples)
                pbar.n = processed

                if i >= max_batches:
                    break

        model.train()
        results = model.test_end(outputs)
        return results




    def test_baseline(self, model ,target_dataloader, output_root, dataset_name, n_query,all_users):
        acc_list = []

        for i , (x, y) in enumerate(target_dataloader):
            set_random_seed(5)

            t_u = target_dataloader.batch_sampler.sampled_users[-1][-1]
            train_users = []
            for u in all_users:
                if u != t_u:
                    train_users.append(u)

            if dataset_name != 'Face':
                checkpoint_dir = os.path.join(output_root, 'groundTruth', '%s_train_users_%s_test_users_%s' % (dataset_name, ''.join(train_users), t_u) , 'model.tar')
            else:
                u_map ={'0': '0','1': '2', '2': '3', '3': '6', '4': '9', }
                a_u = ['0','1','2','3','4','5','6','7','8','9']
                s_us = []
                for u in a_u:
                    if u != u_map[t_u]:
                        s_us.append(u)
                checkpoint_dir = os.path.join(output_root, 'groundTruth', '%s_train_users_%s_test_users_%s' % (dataset_name, ''.join(s_us), u_map[t_u]) , 'model.tar')

            checkpoint = torch.load(checkpoint_dir, map_location='cuda:0')
            model.load_state_dict(checkpoint)
            model.eval()

            x = x[:,n_query:]
            y = y[:,n_query:]
            x = x.contiguous().view(-1, *(x.shape[2:]))
            y = y.contiguous().view(-1, *(y.shape[2:]))

            l_a = model.validation_step([x,y] , 0)
            print( 'accuracy',l_a['val_accuracy'].item())
            acc_list.append(l_a['val_accuracy'].item() )

        acc_all = np.asarray(acc_list)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('Test Acc std = %4.2f%% +- %4.2f%%'%(acc_mean, acc_std))
        return

    def __create_checkpoint(self, logs=None):
        logs = logs or {}
        if self.checkpoint_callback != None:
            self.checkpoint_callback.on_epoch_end(self.current_epoch, save_func=self.save_checkpoint, seed=self.seed,
                                                  logs=logs)

    def __load_checkpoint(self, checkpoint, model, optimizer=None):
        if checkpoint is not None:
            print(f"Loading checkpoint: {checkpoint}")
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            if optimizer is not None and 'state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def transfer_batch_to_gpu(self, batch):
        if callable(getattr(batch, 'to', None)):
            return batch.to(self.device)
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x)
            return batch
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x)
            return tuple(batch)
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v)

            return batch

        return batch

    def save_checkpoint(self, filepath):
        checkpoint = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)

