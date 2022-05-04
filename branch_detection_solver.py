import time
import torch
import datetime
import torch.nn as nn
from typing import Union, Tuple
from torch.utils.data import DataLoader

# Class which handles training, validation and testing of branch detection network
class SolverBranchDetection:
    def __init__(
        self,
        train_loader: Union[DataLoader, None],
        val_loader: Union[DataLoader, None],
        test_loader: Union[DataLoader, None],
        loss_function: nn.Module,
        device: torch.device,
        id_string: str,
        log_file: str,
        checkpoint_file: str,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_function = loss_function
        self.device = device
        self.id_string = id_string
        self.log_file = log_file
        self.checkpoint_file = checkpoint_file

        self.early_stopping_counter = 0
        self.train_loss_arr = []
        self.val_loss_arr = []

    def train_network(
        self,
        model: nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        epochs: int,
        start_epoch: int,
    ) -> None:
        assert self.train_loader is not None and self.val_loader is not None
        early_stopping_flag = False
        print("Start training -", self.id_string)
        s_time = time.perf_counter()

        for i in range(epochs - start_epoch):
            if not early_stopping_flag:
                start_time = datetime.datetime.now()
                start_time = start_time.strftime("%Y/%m/%d:%H:%M:%S.%f")
                start_epoch_time = time.time()
                print(
                    "Start time: {} - Epoch: {}".format(
                        start_time, start_epoch + (i + 1)
                    )
                )

                model.train()
                train_loss, train_acc = self.step(
                    model=model, optimiser=optimiser, data_loader=self.train_loader
                )
                print(
                    f"Training loss:   {train_loss:>20,.4f} -     Accuracy: {train_acc:>3,.2f}"
                )
                self.save_loss_to_file(eval_type="training", loss=train_loss)

                print("Start validation")
                model.eval()
                with torch.no_grad():
                    val_loss, val_acc = self.step(
                        model=model, data_loader=self.val_loader
                    )
                print(
                    f"Validation loss:   {val_loss:>20,.4f} -     Accuracy: {val_acc:>3,.2f}"
                )
                self.save_loss_to_file(eval_type="validation loss", loss=val_loss)
                self.save_loss_to_file(eval_type="validation acc ", loss=val_acc)

                self.save_checkpoint_every_epoch(
                    epoch=(start_epoch + i + 1),
                    model=model,
                    optimiser=optimiser,
                    loss=val_loss,
                )

                if scheduler is not None:
                    scheduler.step()

                end_epoch_time = time.time()
                time_epoch = end_epoch_time - start_epoch_time
                print(f"Time needed for this epoch: {time_epoch}")

                early_stopping_flag = self.early_stopping(
                    train_loss=train_loss, val_loss=val_loss
                )

        e_time = time.perf_counter()
        time_diff = e_time - s_time
        print(time_diff)
        self.save_loss_to_file(eval_type="time", loss=time_diff)

    def test_network(self, model: nn.Module) -> None:
        print("Start testing -", self.id_string)

        model.eval()
        with torch.no_grad():
            test_loss, test_acc = self.step(model=model, data_loader=self.test_loader)
        print(f"Testing loss:   {test_loss:>20,.4f} -   Accuracy: {test_acc:>3,.4f}")
        self.save_loss_to_file(eval_type="testing loss", loss=test_loss)
        self.save_loss_to_file(eval_type="testing accuracy", loss=test_acc)

    # Train, validation and test step
    def step(
        self, model: nn.Module, data_loader, optimiser=None
    ) -> Tuple[float, float]:
        loss_value = 0.0
        accuracy = 0.0
        for j, (inputs, labels) in enumerate(data_loader):
            batch_loss = 0.0
            batch_acc = 0.0
            total = 0
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if optimiser is not None:
                optimiser.zero_grad()
                for param_group in optimiser.param_groups:
                    lr = param_group["lr"]

            outputs = model(inputs)
            batch_loss = self.loss_function(outputs, labels)
            loss_value += batch_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            batch_acc += predicted.eq(labels).sum().item()
            batch_acc = batch_acc / total
            accuracy += batch_acc

            if optimiser is not None:
                batch_loss.backward()
                optimiser.step()
            if not j % 10:
                print(
                    f"Batch {j+1} loss:   {batch_loss.item():>20,.4f}  -   Accuracy: {batch_acc:>3,.2f}"
                )
        loss_value = loss_value / (j + 1)  # normalize loss by number of batches
        accuracy = accuracy / (j + 1)

        return loss_value, accuracy

    # Save loss to log file
    def save_loss_to_file(self, eval_type: str, loss: float) -> None:
        fw = open(self.log_file, "a+")

        line = "\n{} - {:<12} - {:>20,.4f}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), eval_type, loss
        )
        fw.write(line)
        fw.close()

    # Save checkpoint every epoch
    def save_checkpoint_every_epoch(
        self, epoch: int, model: nn.Module, optimiser: torch.optim, loss: float
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "loss": loss,
            },
            self.checkpoint_file,
        )

    def early_stopping(self, train_loss: float, val_loss: float) -> bool:
        if self.val_loss_arr != [] and self.val_loss_arr[-1] < val_loss and self.train_loss_arr[-1] >= train_loss:
            self.early_stopping_counter += 1
        self.val_loss_arr.append(val_loss)
        self.train_loss_arr.append(train_loss)
        if self.early_stopping_counter >= 5:
            return True
        else:
            return False
