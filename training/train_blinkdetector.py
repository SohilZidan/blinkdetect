#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import logging
import shutil
import random
from tqdm import tqdm
import paramiko 

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
    argparser.add_argument("--generate_fnfp_plots", action="store_true")
    
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
    log_file = os.path.join(dataset_path, f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{EPOCH}.txt")
    if os.path.exists(log_file):
          os.remove(log_file)
    logging.basicConfig(filename=log_file,  level=logging.INFO)
    
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

      for data, target, duration,_,__ in dataloaders['train']:

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
        for data, target, duration,_,__ in dataloaders['val']:
          data, target, duration= data.to(device), target.to(device), duration.to(device)
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
    _pids = []
    _rngs = []
    # 
    testing_progress = tqdm(total=len(test_dataset_loader),
                               desc="Testing progress")
    # testing_progress.reset()
    with torch.no_grad():
      
      test_loss = 0.0
      avg_test_loss = 0.0
      for data, target, duration, _pid, _rng in dataloaders['test']:
        data, target, duration = data.to(device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)
        # 
        y_test_pred = torch.sigmoid(pred_target)
        y_pred_tag = torch.round(y_test_pred).type_as(target)
        y_pred_list.extend(y_pred_tag.cpu().numpy())
        y_test.extend(target.cpu().numpy())
        # # # # # # # # # # #
        _pids.extend(_pid)
        _rngs.extend(_rng)
        # # # # # # # # # # #
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
    all_pids = []
    all_rngs = []
    duration_MSE = 0
    classification_cls = 0
    # 
    dataset_progress = tqdm(total=len(all_dataset_loader),
                               desc="All dataset progress")
    # testing_progress.reset()
    with torch.no_grad():
      
      test_loss = 0.0
      avg_test_loss = 0.0
      for data, target, duration,_pid, _rng in dataloaders['dataset']:
        data, target, duration = data.to(device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)
        # 
        all_pids.extend(_pid)
        all_rngs.extend(_rng)
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


    all_combined = [(x,x_pred, _id, _rng) for x, x_pred, _id, _rng in list(zip(all_y_test, all_y_pred_list, all_pids, all_rngs)) if x!=x_pred]
    test_combined = [(x,x_pred, _id, _rng) for x, x_pred, _id, _rng in list(zip(y_list, y_pred_list, _pids, _rngs)) if x!=x_pred]
    

    
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
    logging.info("FP+FN")
    logging.info(test_combined)

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
    logging.info("FP+FN")
    logging.info(all_combined)
    # print(all_y_test, all_y_pred_list)

    
    plt.plot(training_losses, 'r', label="training loss")
    plt.plot(validation_losses, 'b',  label="validation loss")
    plt.legend()
    fig_out_file = os.path.join(dataset_path, f"{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{EPOCH}.png")
    plt.savefig(fig_out_file, dpi=300, bbox_inches='tight')

    # # # # # # #
    fp_fn = os.path.join(dataset_path, f"fp+fn-{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}-{EPOCH}")
    os.makedirs(fp_fn, exist_ok=True)

    # # # # # # # # # # # # # # # # # # # # # # # 
    
    if args.generate_fnfp_plots:
      paramiko.util.log_to_file('logfile.log')
      host = "nipg11.inf.elte.hu"
      port = 10113
      transport = paramiko.Transport((host, port))
      password = "S9hiLLai*"
      username = "zidan"
      transport.connect(username = username, password = password)

      sftp = paramiko.SFTPClient.from_transport(transport)
      # print(type(sftp), sftp.getcwd())
      # print(sftp.get_channel())
      # print(sftp.listdir())
      # # # # # # # # # # # # # # # # # # # # # # # 

      # TEST
      test_fp_fn = os.path.join(fp_fn, "test")
      dataset_path = os.path.dirname(dataset_path)
      dataset_path = os.path.join(dataset_path, "plots")
      sample_number = 1
      if len(test_combined) > 100:
          test_combined = random.sample(test_combined, 100)
      for _actual, _prediction, _pid, _rng in tqdm(test_combined, total=len(test_combined), desc="test"):
          # 
          _start, _stop = _rng.split("-")
          _to = f"{test_fp_fn}/{_actual}/{_pid}_sample_{sample_number}"
          if not os.path.exists(_to):
            os.makedirs(_to)
          shutil.copyfile(f"{dataset_path}/{_pid}_[{_rng}]_{_actual}.png", f"{_to}/{_pid}_[{_rng}]_{_actual}.png")
          # 
          for i in range(int(_start), int(_stop)):
            if i < 0: continue
            _from = f"./Blinking/dataset/BlinkingValidationSetVideos/{_pid}/frames/{i:06}.png"
            _to1 = f"{_to}/{i:06}.png"
            sftp.get(_from, _to1)
            # shutil.copyfile(_from, _to)
          
          sample_number+=1

      # ALL
      test_fp_fn = os.path.join(fp_fn, "all")
      dataset_path = os.path.dirname(dataset_path)
      dataset_path = os.path.join(dataset_path, "plots")
      if len(all_combined) > 100:
          all_combined = random.sample(all_combined, 100)
      for _actual, _prediction, _pid, _rng in tqdm(all_combined, total=len(all_combined), desc="all"):
          # 
          _start, _stop = _rng.split("-")
          _to = f"{test_fp_fn}/{_actual}/{_pid}_sample_{sample_number}"
          if not os.path.exists(_to):
            os.makedirs(_to)
          shutil.copyfile(f"{dataset_path}/{_pid}_[{_rng}]_{_actual}.png", f"{_to}/{_pid}_[{_rng}]_{_actual}.png")
          # 
          for i in range(int(_start)-5, int(_stop)+5):
            if i < 0: continue
            _from = f"./Blinking/dataset/BlinkingValidationSetVideos/{_pid}/frames/{i:06}.png"
            _to1 = f"{_to}/{i:06}.png"
            sftp.get(_from, _to1)
            # shutil.copyfile(_from, _to)
          sample_number+=1

      sftp.close()
      transport.close()
    # Found standard deviations (ground truth is 10 and 1):
    # std_1 = torch.exp(log_var_a)**0.5
    # std_2 = torch.exp(log_var_b)**0.5
    # print([std_1.item(), std_2.item()])
