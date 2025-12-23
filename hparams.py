import optuna


class Hparams:
    def __init__(self):
        # model
        self.lr = 3e-6
        self.vat_lambda = 1.0  # The weight of adversarial training loss
        self.vat_learning_rate = 3e-6
        self.vat_loss_fn = "symmetric-kl"
        self.vat_init_perturbation = 1e-2
        self.use_adversarial = False

    def set_tune_config(self, trial: optuna.trial.Trial):
        self.lr = trial.suggest_float("lr", 1e-6, 1e-3)
