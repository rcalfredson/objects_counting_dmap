import numpy as np

MODES, PERIOD = "modes", "period"
DEFAULTS = {
    PERIOD: 1,
    MODES: {
        "constant": {"split": 0.5},
        "flip": {"epoch": 150},
        "randomChoice": {"prob": 0.5},
    },
}
DUAL_MODE = "dualMode"
DUAL_OPTIONS = "dualOptions"


def set_dual_loss_defaults(config):
    if DUAL_MODE not in config or config[DUAL_MODE] not in DEFAULTS[MODES]:
        config[DUAL_MODE] = "constant"
    if DUAL_OPTIONS not in config or type(config[DUAL_OPTIONS]) != dict:
        config[DUAL_OPTIONS] = dict(period=DEFAULTS[PERIOD])
    for key in DEFAULTS[MODES][config[DUAL_MODE]]:
        if key not in config[DUAL_OPTIONS]:
            config[DUAL_OPTIONS][key] = DEFAULTS[MODES][config[DUAL_MODE]][key]
    return config


def _constant_loss_weight_function(split):
    def constant_loss_weight_function(*args):
        return split, 1 - split

    return constant_loss_weight_function


def _flip_loss_weight_function(epoch_of_flip):
    def flip_loss_weight_function(epoch):
        if epoch + 1 >= epoch_of_flip:
            return 0, 1
        else:
            return 1, 0

    return flip_loss_weight_function


def _random_choice_loss_weight_function(prob):
    def random_choice_loss_weight_function(*args):
        if np.random.rand() <= prob:
            return 1, 0
        else:
            return 0, 1

    return random_choice_loss_weight_function


def get_loss_weight_function(config):
    mode = config[DUAL_MODE]
    if mode == "constant":
        return _constant_loss_weight_function(config[DUAL_OPTIONS]["split"])
    elif mode == "flip":
        return _flip_loss_weight_function(config[DUAL_OPTIONS]["epoch"])
    elif mode == 'randomChoice':
        return _random_choice_loss_weight_function(config[DUAL_OPTIONS]["prob"])