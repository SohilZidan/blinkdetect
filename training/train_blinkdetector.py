#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import logging
from tqdm import tqdm

import torch
from torch.nn import MSELoss, BCEWithLogitsLoss, Sigmoid

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from blinkdetect.models.blinkdetection import BlinkDetector
from blinkdetect.dataset import BlinkDataset1C, BlinkDataset2C, BlinkDataset4C

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt


def_anns_file = os.path.join(os.path.dirname(__file__), "dataset","augmented_signals", "annotations.json")

def parser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--annotation_file", default=def_anns_file)
    argparser.add_argument("--dataset_path", default="")
    argparser.add_argument("--prefix", required=True)
    argparser.add_argument("--channels", required=True, choices=['1C', '2C', '4C'])
    argparser.add_argument("--batch", type=int ,default=4)
    argparser.add_argument("--epoch", type=int ,default=50)
    argparser.add_argument("--normalized", action="store_true")
    
    return argparser.parse_args()

if __name__ == '__main__':
          
    args = parser()

    # 
    BATCH_SIZE = args.batch
    EPOCH = args.epoch
    
    
    if args.dataset_path == "":
          _name, _ = os.path.basename(args.annotation_file).split(".")
          _, _version = _name.split("-")
          dataset_path = os.path.join(os.path.dirname(__file__), ".." , "dataset", "augmented_signals", "versions", _version, "training")
    else:
          dataset_path = args.dataset_path
    
    os.makedirs(dataset_path, exist_ok=True)
    
    logging.basicConfig(filename=os.path.join(dataset_path, f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{EPOCH}.txt"),  level=logging.INFO)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix random seed
    torch.manual_seed(192020)
    

    """
    #Random data for testing can be ignored/removed
    y = torch.rand((100, 4, 30))
    duration = 20.0 * torch.rand((100,1), dtype=torch.float32)
    label = torch.randint(1, (100, 1), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(y, label, duration)
    num_samples = len(dataset)
    """
    if args.channels == '1C':
      num_chan = 1
      dataset = BlinkDataset1C(args.annotation_file, args.normalized) #Dataset class
    if args.channels == '2C':
      num_chan = 2
      dataset = BlinkDataset2C(args.annotation_file, args.normalized) #Dataset class
    if args.channels == '4C':
      num_chan = 4
      dataset = BlinkDataset4C(args.annotation_file, args.normalized) #Dataset class
    
    num_samples = len(dataset)

    train_count = int(0.7 * num_samples)
    valid_count = int(0.2 * num_samples)
    test_count = num_samples - train_count - valid_count
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count))

    print(f"dataset: {len(dataset)}, train: {len(train_dataset)}, val: {len(valid_dataset)}, test: {len(test_dataset)}")
    logging.info(f"dataset: {len(dataset)}, train: {len(train_dataset)}, val: {len(valid_dataset)}, test: {len(test_dataset)}")
    # exit()

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


    
    # Initialize the MLP
    network = BlinkDetector(num_chan).to(device)
    pytorch_total_params = sum(p.numel() for p in network.parameters())
    print("number of model parameters:", pytorch_total_params)
    logging.info(f"number of model parameters: {pytorch_total_params}")

    logging.info("network architecture")
    logging.info(network)
    # exit()
    # 
    # Define task dependent log_variance
    # 
    # log_var_a = torch.zeros((1,), requires_grad=True)
    # log_var_b = torch.zeros((1,), requires_grad=True)
    # Initialized standard deviations (ground truth is 10 and 1):
    # std_1 = torch.exp(log_var_a)**0.5
    # std_2 = torch.exp(log_var_b)**0.5
    # get all parameters (model parameters + task dependent log variances)
    # params = ([p for p in network.parameters()] + [log_var_a] + [log_var_b])

    # Define the loss function and optimizer
    cls_loss = BCEWithLogitsLoss()
    reg_loss = MSELoss()
    sig = Sigmoid()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # optimizer = torch.optim.Adam(params, lr=1e-4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    epochs = tqdm(range(EPOCH), desc="Epochs")
    training_progress = tqdm(total=len(train_dataset_loader),
                             desc="Training progress")
    validation_progress = tqdm(total=len(valid_dataset_loader),
                               desc="Validation progress")
    
    training_losses = []
    validation_losses = []
    # Training loop
    for epoch in epochs:
      training_progress.reset()
      validation_progress.reset()

      current_loss = 0.0
      avg_loss = 0.0

      for data, target, duration in dataloaders['train']:

        data, target, duration = data.to(device), target.to(device), duration.to(device)

        optimizer.zero_grad()
        pred_target, pred_duration = network(data)

        # Compute loss
        loss = cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(pred_duration, duration.type_as(pred_duration))
        # loss = cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(sig(pred_duration), duration.type_as(pred_duration))

        # loss_c = cls_loss(pred_target, target.type_as(pred_target))
        # loss_r = reg_loss(sig(pred_duration), duration.type_as(pred_duration))
        # loss = torch.sqrt(loss_c**2+ loss_r**2)

        # loss = criterion_0([cls_loss(pred_target, target.type_as(pred_target)), reg_loss(sig(pred_duration), duration.type_as(pred_duration))], [log_var_a, log_var_b])

          
        loss.backward()

        optimizer.step()
        # Print statistic
        current_loss += loss.item()
        training_progress.update()
      
  

      avg_loss = current_loss*BATCH_SIZE / len(train_dataset)
      training_losses.append(avg_loss)
      training_progress.set_postfix(average_loss=avg_loss)
        

      with torch.no_grad():
        val_loss = 0.0
        avg_val_loss = 0.0
        for data, target, duration in dataloaders['val']:
          data, target, duration = data.to(device), target.to(device), duration.to(device)
          pred_target, pred_duration = network(data)
          val_loss += cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(pred_duration, duration.type_as(pred_duration))
          # val_loss += cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(sig(pred_duration), duration.type_as(pred_duration))
          # loss_c = cls_loss(pred_target, target.type_as(pred_target))
          # loss_r = reg_loss(sig(pred_duration), duration.type_as(pred_duration))

          # val_loss += torch.sqrt(loss_c**2+ loss_r**2)

          # val_loss += criterion([sig(1/(log_var_a.to(device)**2)*pred_target), pred_duration], [target, duration], [log_var_a, log_var_b])
          # val_loss += criterion_0([cls_loss(pred_target, target.type_as(pred_target)), reg_loss(sig(pred_duration), duration.type_as(pred_duration))], [log_var_a, log_var_b])
          # val_loss += criterion_1([cls_loss(pred_target, target.type_as(pred_target)), reg_loss(pred_duration, duration.type_as(pred_duration))], [log_var_a, log_var_b])
          validation_progress.update()
      avg_val_loss = val_loss.item()*BATCH_SIZE / len(valid_dataset)
      validation_losses.append(avg_val_loss)
      validation_progress.set_postfix(valid_loss=avg_val_loss)
        
    
      
    


    # testing
    y_pred_list = []
    y_test = []
    duration_MSE = 0
    classification_cls = 0
    # 
    testing_progress = tqdm(total=len(test_dataset_loader),
                               desc="Testing progress")
    # testing_progress.reset()
    with torch.no_grad():
      
      test_loss = 0.0
      avg_test_loss = 0.0
      for data, target, duration in dataloaders['test']:
        data, target, duration = data.to(device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)
        # 
        y_test_pred = torch.sigmoid(pred_target)
        y_pred_tag = torch.round(y_test_pred).type_as(target)
        y_pred_list.extend(y_pred_tag.cpu().numpy())
        y_test.extend(target.cpu().numpy())
        # 
        duration_MSE += reg_loss(sig(pred_duration), duration.type_as(pred_duration)).mul(3.1355).exp_()
        classification_cls += cls_loss(pred_target, target.type_as(pred_target))
        
        test_loss += cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(pred_duration, duration.type_as(pred_duration))
        # test_loss += cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(sig(pred_duration), duration.type_as(pred_duration))
        # loss_c = cls_loss(pred_target, target.type_as(pred_target))
        # loss_r = reg_loss(sig(pred_duration), duration.type_as(pred_duration))

        # test_loss += torch.sqrt(loss_c**2+ loss_r**2)
        # test_loss += criterion([sig(1/(log_var_a.to(device)**2)*pred_target), pred_duration], [target, duration], [log_var_a, log_var_b])
        # test_loss += criterion_0([cls_loss(pred_target, target.type_as(pred_target)), reg_loss(sig(pred_duration), duration.type_as(pred_duration))], [log_var_a, log_var_b])
        # test_loss += criterion_1([cls_loss(pred_target, target.type_as(pred_target)), reg_loss(pred_duration, duration.type_as(pred_duration))], [log_var_a, log_var_b])
        testing_progress.update()
    avg_test_loss = test_loss.item()*BATCH_SIZE / len(test_dataset)
    avg_duration_MSE = duration_MSE.item()*BATCH_SIZE / len(test_dataset)
    avg_classification_cls = classification_cls.item()*BATCH_SIZE / len(test_dataset)
    testing_progress.set_postfix(testing_loss=avg_test_loss)

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_list = [a.squeeze().tolist() for a in y_test]
    # 
    tn, fp, fn, tp = confusion_matrix(y_list, y_pred_list).ravel()

    # # # # # # # #
    # all dataset #
    all_y_pred_list = []
    all_y_test = []
    duration_MSE = 0
    classification_cls = 0
    # 
    dataset_progress = tqdm(total=len(all_dataset_loader),
                               desc="All dataset progress")
    # testing_progress.reset()
    with torch.no_grad():
      
      test_loss = 0.0
      avg_test_loss = 0.0
      for data, target, duration in dataloaders['dataset']:
        data, target, duration = data.to(device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)
        # 
        y_test_pred = torch.sigmoid(pred_target)
        y_pred_tag = torch.round(y_test_pred).type_as(target)
        all_y_pred_list.extend(y_pred_tag.cpu().numpy())
        all_y_test.extend(target.cpu().numpy())
        # 
        duration_MSE += reg_loss(sig(pred_duration), duration.type_as(pred_duration)).mul(3.1355).exp_()
        test_loss += cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(pred_duration, duration.type_as(pred_duration))
        # test_loss += criterion([sig(1/(log_var_a.to(device)**2)*pred_target), pred_duration], [target, duration], [log_var_a, log_var_b])
        # test_loss += criterion_0([cls_loss(pred_target, target.type_as(pred_target)), reg_loss(sig(pred_duration), duration.type_as(pred_duration))], [log_var_a, log_var_b])
        # test_loss += torch.mean(cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(sig(pred_duration), duration.type_as(pred_duration)))
        # loss_c = 
        classification_cls += cls_loss(pred_target, target.type_as(pred_target))
        # loss_r = reg_loss(sig(pred_duration), duration.type_as(pred_duration))

        # test_loss += torch.sqrt(loss_c**2+ loss_r**2)
        # loss_c = cls_loss(pred_target, target.type_as(pred_target))
        # loss_r = reg_loss(sig(pred_duration), duration.type_as(pred_duration))

        # test_loss += torch.norm(torch.cat((loss_c, loss_r)))
        dataset_progress.update()
    avg_test_loss = test_loss.item()*BATCH_SIZE / len(dataset)
    avg_dataset_duration_MSE = duration_MSE.item()*BATCH_SIZE / len(dataset)
    avg_dataset_classification_cls = classification_cls.item()*BATCH_SIZE / len(dataset)
    dataset_progress.set_postfix(all_loss=avg_test_loss)

    all_y_pred_list = [a.squeeze().tolist() for a in all_y_pred_list]
    all_y_test = [a.squeeze().tolist() for a in all_y_test]

    _tn, _fp, _fn, _tp = confusion_matrix(all_y_test, all_y_pred_list).ravel()

    # closing progress bars
    
    epochs.close()
    training_progress.close()
    validation_progress.close()
    testing_progress.close()
    dataset_progress.close()
    
    
    # print()
    # print()
    # 
    
    # print(y_pred_list)
    # exit()
    # y_pred_list = [item for sublist in y_pred_list for item in sublist]
    # y_list = [item for sublist in y_list for item in sublist]

    

    
    # Testing
    print("testing performace")
    logging.info("testing performace")
    print(f"duration mse: {avg_duration_MSE}")
    logging.info(f"duration mse: {avg_duration_MSE}")
    print(f"classifier loss: {avg_classification_cls}")
    logging.info(f"classifier loss: {avg_classification_cls}")
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    logging.info(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    print(classification_report(y_list, y_pred_list))
    logging.info(classification_report(y_list, y_pred_list))

    print("overall performace")
    logging.info("overall performace")
    print(f"duration mse: {avg_dataset_duration_MSE}")
    logging.info(f"duration mse: {avg_dataset_duration_MSE}")
    print(f"classifier loss: {avg_dataset_classification_cls}")
    logging.info(f"classifier loss: {avg_dataset_classification_cls}")
    print(f"tn: {_tn}, fp: {_fp}, fn: {_fn}, tp: {_tp}")
    logging.info(f"tn: {_tn}, fp: {_fp}, fn: {_fn}, tp: {_tp}")
    print(classification_report(all_y_test, all_y_pred_list))
    logging.info(classification_report(all_y_test, all_y_pred_list))

    
    plt.plot(training_losses, 'r', label="training loss")
    plt.plot(validation_losses, 'b',  label="validation loss")
    plt.legend()
    fig_out_file = os.path.join(dataset_path, f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{EPOCH}.png")
    plt.savefig(fig_out_file, dpi=300, bbox_inches='tight')


    # Found standard deviations (ground truth is 10 and 1):
    # std_1 = torch.exp(log_var_a)**0.5
    # std_2 = torch.exp(log_var_b)**0.5
    # print([std_1.item(), std_2.item()])



# model.eval()
# with torch.no_grad():
#     for X_batch, labels in val_dataloader:
#         X_batch = X_batch#.to(device)
#         y_test_pred = model(X_batch)
#         y_test_pred = torch.sigmoid(y_test_pred)
#         y_pred_tag = torch.round(y_test_pred)
#         y_pred_list.append(y_pred_tag.cpu().numpy())
#         y_test.append(labels.cpu().numpy())
# y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
# y_list = [a.squeeze().tolist() for a in y_test]