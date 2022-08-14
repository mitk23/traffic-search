import time

import torch

import config
from utils.helper import fix_seed


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device=None,
        logger=None,
        model_name=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.model_name = model_name

        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs,
        log_steps=None,
        max_first_log_steps=3,
        max_time=None,
        save_epoch_steps=None,
        random_seed=config.RANDOM_SEED,
    ):
        fix_seed(random_seed)
        start = time.time()

        best_loss = 1e20
        for epoch in range(1, n_epochs + 1):
            ## increment epoch
            self.current_epoch += 1

            ## train
            self.model.train()
            epoch_start = time.time()

            train_loss = self.__train_epoch(train_loader)
            self.train_losses.append(train_loss)

            train_time = time.time() - epoch_start

            ## validate
            self.model.eval()
            epoch_start = time.time()

            val_loss = self.__valid_epoch(val_loader)
            self.val_losses.append(val_loss)

            val_time = time.time() - epoch_start

            ## logging
            log_flag = (self.logger is not None) and (log_steps is not None)
            if log_flag and (
                (self.current_epoch <= max_first_log_steps)
                or (self.current_epoch % log_steps == 0)
            ):
                message = f"Epoch: {self.current_epoch} | Train Loss: {train_loss:.3f}, Train Time: {train_time:.2f} [sec] | Valid Loss: {val_loss:.3f}, Valid Time: {val_time:.2f} [sec]"
                self.__logging(message)

            ## model saving
            # can_save = (self.model_path is not None) and (
            #     save_epoch_steps is not None
            # )
            # if can_save and (self.current_epoch % save_epoch_steps == 0):
            #     self.save(self.model_path)
            if (self.model_name is not None) and (val_loss < best_loss):
                best_loss = val_loss
                self.save(self.model_name)

            ## early stopping
            if (max_time is not None) and (time.time() - start >= max_time):
                break

        return self.train_losses, self.val_losses

    def validate(self, loader):
        """1epochだけ回して性能を評価"""
        self.model.eval()

        total_loss = self.__valid_epoch(loader)
        return total_loss

    def predict(self):
        return

    def save(self, model_name=None):
        assert isinstance(model_name, str), "model name must be passed"
        model_path = (
            f"{config.MODEL_DIR}/{model_name}_{self.current_epoch}.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        return

    def __train_epoch(self, loader):
        train_loss = 0.0

        for i_batch, (data, target) in enumerate(loader):
            if isinstance(data, (list, tuple)):
                data = map(lambda x: x.to(device=self.device), data)
            else:
                data = data.to(device=self.device)
            target = target.to(device=self.device)

            out = self.model(data)
            loss = self.loss_fn(out, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(loader)
        return train_loss

    def __valid_epoch(self, loader):
        valid_loss = 0.0

        with torch.no_grad():
            for i_batch, (data, target) in enumerate(loader):
                if isinstance(data, (list, tuple)):
                    data = map(lambda x: x.to(device=self.device), data)
                else:
                    data = data.to(device=self.device)
                target = target.to(device=self.device)

                out = self.model(data)
                loss = self.loss_fn(out, target)

                valid_loss += loss.item()

            valid_loss /= len(loader)
        return valid_loss

    def __logging(self, message):
        self.logger.log(message)
        return
