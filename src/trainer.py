import time

import torch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device=None,
        logger=None,
        log_steps=None,
        max_first_log_steps=3,
        model_name=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.log_steps = log_steps
        self.max_first_log_steps = max_first_log_steps
        self.model_name = model_name

        self.train_losses = []
        self.val_losses = []
        self.best_loss = 1e20
        self.current_epoch = 0

    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs,
        max_time=None,
    ):
        start = time.time()

        for epoch in range(1, n_epochs + 1):
            # increment epoch
            self.current_epoch += 1

            # train
            self.model.train()
            epoch_start = time.time()

            train_loss = self.__train_epoch(train_loader)
            self.train_losses.append(train_loss)

            train_time = time.time() - epoch_start

            # validate
            self.model.eval()
            epoch_start = time.time()

            val_loss = self.__valid_epoch(val_loader)
            self.val_losses.append(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss

            val_time = time.time() - epoch_start

            # logging
            if (self.logger is not None) and (self.log_steps is not None):
                self.__logging(train_loss, train_time, val_loss, val_time)

            # model saving
            if self.model_name is not None:
                self.save(self.model_name)

            # early stopping
            if (max_time is not None) and (time.time() - start >= max_time):
                break

        return self.train_losses, self.val_losses

    def validate(self, loader):
        """1epochだけ回して性能を評価"""
        self.model.eval()

        total_loss = self.__valid_epoch(loader)
        return total_loss

    def predict(self, dataset):
        self.model.eval()

        with torch.no_grad():
            data, _ = dataset[:]
            if isinstance(data, (list, tuple)):
                data = map(lambda x: x.to(device=self.device), data)
            else:
                data = data.to(device=self.device)
            out = self.model.generate(data)

        return out

    def save(self, model_name=None):
        assert isinstance(model_name, str), "model name must be passed"

        model_path = f"{model_name}_best.pth"
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

            decoder_xs = torch.full_like(target, -1)
            decoder_xs[..., 1:] = target[..., :-1]

            out = self.model(data, decoder_xs)
            loss = self.loss_fn(out, target.view(target.shape[0], -1))

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

                predicted = self.model.generate(data)
                loss = self.loss_fn(
                    predicted, target.view(target.shape[0], -1)
                )

                valid_loss += loss.item()

            valid_loss /= len(loader)
        return valid_loss

    def __logging(self, train_loss, train_time, valid_loss, valid_time):
        if (self.current_epoch % self.log_steps > 0) and (
            self.current_epoch > self.max_first_log_steps
        ):
            return

        message = f"Epoch: {self.current_epoch} | Train Loss: {train_loss:.3f}, Train Time: {train_time:.2f} [sec] | Valid Loss: {valid_loss:.3f}, Valid Time: {valid_time:.2f} [sec]"

        self.logger.log(message)
        return
