#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import logging
import argparse
import pprint
import matplotlib.pyplot as plt

import torch
from torch.utils.data import ConcatDataset
from torch.nn import MSELoss, CrossEntropyLoss, Sigmoid

import ignite
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import (
    Accuracy, RunningAverage, Precision, Recall,
    ClassificationReport, ConfusionMatrix)
from ignite.engine import Events, Engine
from blinkdetect.dataset import BlinkDataset1C, BlinkDataset2C, BlinkDataset4C
from blinkdetect.models.blinkdetection import BlinkDetector

torch.multiprocessing.set_sharing_strategy('file_system')

def_anns_file = os.path.join(os.path.dirname(
    __file__), "..", "dataset", "augmented_signals", "annotations.json")
checkpoints_folder = os.path.join(
    os.path.dirname(__file__), "..", "checkpoint")
best_checkpoints_folder = os.path.join(
    os.path.dirname(__file__), "..", "best_model")


def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--annotation_file",
        required=True, metavar='N', nargs='+')
    argparser.add_argument("--dataset_path",
        required=True)
    argparser.add_argument("--model",
        default="",
        help="if a correct path to a model is present this model will be retrained")
    argparser.add_argument("--prefix",
        required=True)
    argparser.add_argument("--channels",
        required=True, choices=['1C', '2C', '4C'])
    argparser.add_argument("--batch",
        type=int, default=4)
    argparser.add_argument("--epoch",
        type=int, default=50)

    argparser.add_argument('--normalized',
        dest='normalized', action='store_true')
    argparser.add_argument('--no-normalized',
        dest='normalized', action='store_false')
    argparser.set_defaults(normalized=True)

    argparser.add_argument('--earlystopping',
        dest='earlystopping', action='store_true')
    argparser.add_argument('--no-earlystopping',
        dest='earlystopping', action='store_false')
    argparser.set_defaults(earlystopping=True)

    return argparser.parse_args()


if __name__ == '__main__':
    args = parser()

    _retrain = False
    if args.model != "":
        if len(os.path.basename(args.model).split("_")) == 3:
            fileprefix, name, step_number = os.path.basename(
                args.model).split("_")
            _, prefix, _normalized, _chan, _batch = fileprefix.split("-")
            args.batch = int(_batch)
            args.channels = _chan
            args.normalized = False if _normalized == "False" else True
            _retrain = True
        # else:
        #     prefix, _normalized, _chan, _batch_t = os.path.basename(args.model).split("-")
        #     _batch, ext = _batch_t.split(".") # epoch
        #     _batch = "".join(["E", _batch])
        # normalized = False if _normalized=='False' else True
    BATCH_SIZE = args.batch
    EPOCH = args.epoch

    # dataset_name = os.path.basename(os.path.dirname(args.dataset_path))

    _p = os.path.normpath(args.dataset_path)
    dataset_name = _p.split(os.sep)[-3]

    if args.dataset_path == "":
        _name, _ = os.path.basename(args.annotation_file).split(".")
        _, _version = _name.split("-")
        dataset_path = os.path.join(os.path.dirname(
            __file__), "..", "dataset", "augmented_signals", "versions", _version, "training-softmax")
    else:
        dataset_path = args.dataset_path

    os.makedirs(dataset_path, exist_ok=True)
    log_file = os.path.join(
        dataset_path, f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{args.earlystopping}-softmax.txt")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(filename=log_file,  level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed
    torch.manual_seed(192020)
    ignite.utils.manual_seed(192020)

    """
    #Random data for testing can be ignored/removed
    y = torch.rand((100, 4, 30))
    duration = 20.0 * torch.rand((100,1), dtype=torch.float32)
    label = torch.randint(1, (100, 1), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(y, label, duration)
    num_samples = len(dataset)
    """
    datasets = []
    for _ann_file in args.annotation_file:
        if args.channels == '1C':
            num_chan = 1
            datasets.append(BlinkDataset1C(
                _ann_file, args.normalized))  # Dataset class
        if args.channels == '2C':
            num_chan = 2
            datasets.append(BlinkDataset2C(
                _ann_file, args.normalized))  # Dataset class
        if args.channels == '4C':
            num_chan = 4
            datasets.append(BlinkDataset4C(
                _ann_file, args.normalized))  # Dataset class

    dataset = ConcatDataset(datasets)
    num_samples = len(dataset)

    train_count = int(0.7 * num_samples)
    valid_count = int(0.2 * num_samples)
    test_count = num_samples - train_count - valid_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_count, valid_count, test_count))

    print(
        f"dataset: {len(dataset)}, train: {len(train_dataset)}, val: {len(valid_dataset)}, test: {len(test_dataset)}")
    logging.info(
        f"dataset: {len(dataset)}, train: {len(train_dataset)}, val: {len(valid_dataset)}, test: {len(test_dataset)}")

    # Dataloaders
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=1)
    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True,
                                                       num_workers=1)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=1)
    all_dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     num_workers=1)

    dataloaders = {'train': train_dataset_loader, 'val': valid_dataset_loader,
                   'test': test_dataset_loader, "dataset": all_dataset_loader}

    # Neural Network Initialization
    network = BlinkDetector(num_chan)
    if _retrain:
        network.load_state_dict(torch.load(args.model))
    network.to(device)

    pytorch_total_params = sum(p.numel() for p in network.parameters())
    print("number of model parameters:", pytorch_total_params)
    logging.info(f"number of model parameters: {pytorch_total_params}")
    logging.info("network architecture")
    logging.info(network)

    # Define the loss function and optimizer
    cls_loss = CrossEntropyLoss()
    reg_loss = MSELoss()
    sig = Sigmoid()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    logging.info(optimizer)

    training_losses = []
    validation_losses = []

    # Training Process function
    def train_step(trainer, batch):
        network.train()
        optimizer.zero_grad()
        data, target, duration, _pid, _rng, _yaw, _pitch = batch
        data, target, duration = data.to(
            device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)
        _loss1 = cls_loss(pred_target, target.long())
        _loss2 = reg_loss(pred_duration, duration.type_as(pred_duration))
        loss = _loss1 + _loss2
        loss.backward()
        optimizer.step()
        return loss.item()

    # Evaluation Process function
    def eval_step(trainer, batch):
        network.eval()
        with torch.no_grad():
            data, target, duration, _pid, _rng, _yaw, _pitch = batch
            data, target, duration = data.to(
                device), target.to(device), duration.to(device)
            pred_target, pred_duration = network(data)
            _loss1 = cls_loss(pred_target, target.long())
            _loss2 = reg_loss(pred_duration, duration.type_as(pred_duration))
            loss = _loss1 + _loss2
            return {
                "combined": loss.item(),
                "cls_loss": _loss1.item(),
                "reg_loss": _loss2.mul(3.1355).exp_().item(),
                "preds": [pred_target, pred_duration],
                "GT": [target, duration]
            }

    # Engines
    trainer = Engine(train_step)
    train_evaluator = Engine(eval_step)
    evaluator = Engine(eval_step)
    tester = Engine(eval_step)

    # Trainer Metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    def class_transform(output):
        y_pred, _ = output["preds"]
        y, _ = output["GT"]
        y_pred = torch.max(y_pred.data, 1)
        return y_pred[1], y

    def binary_one_hot_output_transform(output):
        y_pred, _ = output["preds"]
        y, _ = output["GT"]
        y_pred = torch.max(y_pred.data, 1)
        y_pred = ignite.utils.to_onehot(y_pred[1], 2)
        y = y.long()
        return y_pred, y

    #  Evaluator metrics
    RunningAverage(output_transform=lambda x: x['combined']).attach(
        evaluator, 'loss')
    ClassificationReport(output_transform=binary_one_hot_output_transform,
                         output_dict=True).attach(evaluator, "cr")
    ConfusionMatrix(2, output_transform=binary_one_hot_output_transform).attach(
        evaluator, "cr")
    Accuracy(output_transform=class_transform).attach(evaluator, 'accuracy')

    precision = Precision(output_transform=class_transform, average=False)
    recall = Recall(output_transform=class_transform, average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()

    precision.attach(evaluator, 'precision')
    recall.attach(evaluator, 'recall')
    F1.attach(evaluator, 'F1')

    #  Trainer evaluator metrics
    ClassificationReport(output_transform=binary_one_hot_output_transform,
                         output_dict=True).attach(train_evaluator, "cr")
    ConfusionMatrix(2, output_transform=binary_one_hot_output_transform).attach(
        train_evaluator, "cr")

    Accuracy(output_transform=class_transform).attach(
        train_evaluator, 'accuracy')

    precision = Precision(output_transform=class_transform, average=False)
    recall = Recall(output_transform=class_transform, average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()

    precision.attach(train_evaluator, 'precision')
    recall.attach(train_evaluator, 'recall')
    F1.attach(train_evaluator, 'F1')

    # Tester Metrics
    RunningAverage(output_transform=lambda x: x['reg_loss']).attach(
        tester, 'duration error')
    ClassificationReport(output_transform=binary_one_hot_output_transform,
                         output_dict=True).attach(tester, "cr")
    ConfusionMatrix(2, output_transform=binary_one_hot_output_transform).attach(
        tester, "cr")

    Accuracy(output_transform=class_transform).attach(tester, 'accuracy')

    precision = Precision(output_transform=class_transform, average=False)
    recall = Recall(output_transform=class_transform, average=False)
    F1 = (precision * recall * 2 / (precision + recall)).mean()

    precision.attach(tester, 'precision')
    recall.attach(tester, 'recall')
    F1.attach(tester, 'F1')

    #  ProgressBar
    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ['loss'])

    #  Training Results
    @trainer.on(Events.COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(dataloaders['train'])
        metrics = train_evaluator.state.metrics
        pbar.log_message("----------------------------------")
        pbar.log_message(
            "Training Results - Epoch: {} \nMetrics\n{}"
            .format(trainer.state.epoch, pprint.pformat(metrics)))
        logging.info("----------------------------------")
        logging.info(
            "Training Results - Epoch: {} \nMetrics\n{}"
            .format(trainer.state.epoch, pprint.pformat(metrics)))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        metrics = trainer.state.metrics
        training_losses.append(metrics['loss'])

    #  Validation Results
    def log_validation_results(engine):
        evaluator.run(dataloaders['val'])
        metrics = evaluator.state.metrics
        validation_losses.append(metrics['loss'])
        # pbar.log_message("----------------------------------")
        # pbar.log_message(
        #     "Validation Results - Epoch: {} \nMetrics\n{}"
        #     .format(engine.state.epoch, pprint.pformat(metrics)))
        # pbar.n = pbar.last_print_n = 0
        logging.info("----------------------------------")
        logging.info(
            "Validation Results - Epoch: {} \nMetrics\n{}"
            .format(engine.state.epoch, pprint.pformat(metrics)))

    #  Testing Results
    def log_testing_results(engine):
        tester.run(dataloaders['test'])
        metrics = tester.state.metrics
        pbar.log_message("----------------------------------")
        pbar.log_message(
            "Testing Results - Epoch: {} \nMetrics\n{}"
            .format(engine.state.epoch, pprint.pformat(metrics)))
        pbar.n = pbar.last_print_n = 0
        logging.info("----------------------------------")
        logging.info(
            "Testing Results - Epoch: {} \nMetrics\n{}"
            .format(engine.state.epoch, pprint.pformat(metrics)))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
    trainer.add_event_handler(Events.COMPLETED, log_testing_results)

    # EarlyStopping
    def score_function(engine):
        val_loss = engine.state.metrics['F1']
        return val_loss
    if args.earlystopping:
        handler = EarlyStopping(
            patience=8, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    # ModelCheckpoint
    checkpointer = ModelCheckpoint(
        checkpoints_folder,
        f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}",
        n_saved=2,
        create_dir=True,
        require_empty=False)
    best_model_save = ModelCheckpoint(
        best_checkpoints_folder,
        f"best-{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}", n_saved=1,
        create_dir=True,
        score_function=score_function, require_empty=False)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpointer, {
                              f'{dataset_name}-softmax': network})
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, best_model_save, {
                                f'{dataset_name}-softmax': network})

    #  RUN
    trainer.run(dataloaders['train'], max_epochs=EPOCH)

    if args.earlystopping:
        logging.info("early stopping:")
        logging.info(handler.state_dict())

    # save learning curve
    plt.plot(training_losses, 'r', label="training loss")
    plt.plot(validation_losses, 'b',  label="validation loss")
    plt.legend()
    fig_out_file = os.path.join(
        dataset_path, f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{args.earlystopping}-softmax.png")
    plt.savefig(fig_out_file, dpi=300, bbox_inches='tight')
    plt.close()
