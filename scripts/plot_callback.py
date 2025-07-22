from transformers import TrainerCallback
import matplotlib.pyplot as plt
import numpy as np

class PlotMetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_loss.append(logs["loss"])
            if "eval_loss" in logs:
                self.eval_loss.append(logs["eval_loss"])
            if "eval_accuracy" in logs:
                self.eval_accuracy.append(logs["eval_accuracy"])
            if "epoch" in logs:
                self.epochs.append(logs["epoch"])

    def moving_average(self, data, window_size=20):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure()
        ax1 = plt.gca()
        epochs = self.epochs[:len(self.train_loss)]
        # Train Loss (medie mobilă)
        if self.train_loss:
            ma_train_loss = self.moving_average(self.train_loss, window_size=20)
            ma_epochs = epochs[len(epochs)-len(ma_train_loss):]
            ax1.plot(ma_epochs, ma_train_loss, label="Train Loss (MA)", color="tab:blue", linewidth=2)
        # Eval Loss
        if self.eval_loss:
            ax1.plot(self.epochs[:len(self.eval_loss)], self.eval_loss, label="Eval Loss", color="tab:orange", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        # Eval Accuracy pe axă secundară
        if self.eval_accuracy:
            ax2 = ax1.twinx()
            ax2.plot(self.epochs[:len(self.eval_accuracy)], self.eval_accuracy, label="Eval Accuracy", color="tab:green", linewidth=2, linestyle="--")
            ax2.set_ylabel("Accuracy")
            ax2.legend(loc="lower right")
        ax1.legend(loc="upper right")
        plt.title("Training Metrics")
        plt.savefig("../model/training_metrics.png")
        plt.close() 