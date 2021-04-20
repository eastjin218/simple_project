from copy import deepcopy

import numpy as numpy

import torch

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import PregressBar

from utils import get_gred_norm, get_parameter_norm

VERBOSE_SILENT=0
VERBOSE_EPOCH_WISE =1
VERBOSE_BATCH_WISE =2

class MyEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        self.model =model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss =np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def 

class Trainer():

    def __init__(self, config):
        self.config = config

    def train(
        self, model, optimizer,
        train_loader, valid_loader,
    )

        train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )

        validation_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer, self.config
        )

        MyEngine.attach(
            train_engine, validation_engine,
            verbose =self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine, valid_loader,
        )

        validation_engine.add_event_handler(
            Event.EPOCH_COMPLETED,
            MyEngine.check_best,
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model