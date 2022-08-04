import datetime

import torch


class Trainer:
    def __init__(self, model, optimizer, loss_fn, device=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger

        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs,
        log_steps=None,
        max_first_log_steps=None,
        max_time=None,
    ):
        start = datetime.datetime.now()

        for epoch in range(1, n_epochs + 1):
            ## train
            self.model.train()
            train_loss = 0.0
            epoch_start = datetime.datetime.now()

            for i_batch, (data, target) in enumerate(train_loader):
                data, target = data.to(device=self.device), target.to(
                    device=self.device
                )

                out = self.model(data)
                loss = self.loss_fn(out, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            train_time = datetime.datetime.now() - epoch_start

            ## validate
            self.model.eval()
            val_loss = 0.0
            epoch_start = datetime.datetime.now()

            with torch.no_grad():
                for i_batch, (data, target) in enumerate(val_loader):
                    data, target = data.to(device=self.device), target.to(
                        device=self.device
                    )

                    out = self.model(data)
                    loss = self.loss_fn(out, target)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss / len(val_loader))

            val_time = datetime.datetime.now() - epoch_start

            ## logging
            log_flag = (self.logger is not None) and (log_steps is not None)
            if log_flag and (
                (epoch <= max_first_log_steps)
                or (self.current_epoch % log_steps == 0)
            ):
                message = f"Epoch: {self.current_epoch} | Train Loss: {train_loss}, Train Time: {train_time} | Valid Loss: {val_loss}, Valid Time: {val_time}"
                self.__logging(message)

            ## increment epoch
            self.current_epoch += 1

            ## early stopping
            if (max_time is not None) and (
                datetime.datetime.now() - start >= max_time
            ):
                break

        return self.train_losses, self.val_losses

    def validate(self, data_loader):
        self.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            for i_batch, (data, target) in enumerate(data_loader):
                data, target = data.to(device=self.device), target.to(
                    device=self.device
                )

                out = self.model(data)
                loss = self.loss_fn(out, target)

                total_loss += loss.item()

        total_loss /= len(data_loader)
        return total_loss

    def predict(self):
        return

    def __logging(self, message):
        self.logger.log(message)
        return
