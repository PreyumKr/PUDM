import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from dataset import get_dataloader
from util import training_loss, calc_diffusion_hyperparams, find_max_epoch, print_size, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from shutil import copyfile
import copy
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict

def split_data(data):
    label = data['label'].cuda()
    X = data['complete'].cuda()
    condition = data['partial'].cuda()
    return X, condition, label

def train(
        config_file,
        model_path,
        dataset,
        root_directory,
        output_directory,
        tensorboard_directory,
        n_epochs,
        epochs_per_ckpt,
        iters_per_logging,
        learning_rate,
):
    local_path = dataset
    tb = SummaryWriter(os.path.join(root_directory, local_path, tensorboard_directory))

    output_directory = os.path.join(root_directory, local_path, output_directory)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    try:
        copyfile(config_file, os.path.join(output_directory, os.path.split(config_file)[1]))
    except:
        print('The two files are the same, no need to copy')

    print("output directory is", output_directory, flush=True)
    print("Config file has been copied from %s to %s" % (config_file,
        os.path.join(output_directory, os.path.split(config_file)[1])), flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    trainloader = get_dataloader(trainset_config)

    # Model setup with DataParallel
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = PointNet2CloudCondition(pointnet_config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        net = nn.DataParallel(net, device_ids=[0, 1])  # Explicitly specify GPUs
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    time0 = time.time()
    epoch = 0
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        print(f"---- Loading : {model_path} ----")
        time0 -= checkpoint['training_time_seconds']
        print('checkpoint model loaded successfully', flush=True)
    except:
        print('No valid checkpoint model found, start training from initialization.', flush=True)

    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs)
    loss_function = nn.MSELoss()
    n_iter = epoch * loader_len

    while n_iter < n_iters + 1:
        epoch += 1
        for batch_idx, data in enumerate(trainloader):
            X, condition, label = split_data(data)
            X, condition, label = X.to(device), condition.to(device), label.to(device)
            optimizer.zero_grad()

            loss = training_loss(
                net,
                loss_function,
                X,
                diffusion_hyperparams,
                label=label,
                condition=condition,
            )

            reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                cprint("[{}]\tepoch({}): {} \titeration({}/{}): {} \tMSE loss: {:.6f}".format(
                    time.ctime(),
                    n_epochs,
                    epoch,
                    n_iter,
                    n_iters,
                    batch_idx,
                    loss.item()),
                    "blue"
                )
                tb.add_scalar("Log-Train-Loss", torch.log(loss).item(), n_iter)
                tb.add_scalar("Log-Train-Reduced-Loss", np.log(reduced_loss), n_iter)

            n_iter += 1

        if epoch % epochs_per_ckpt == 0:
            checkpoint_name = 'pointnet_ckpt_{}_{:.6f}.pkl'.format(epoch, loss.item())
            checkpoint_path = os.path.join(output_directory, checkpoint_name)
            torch.save({
                'iter': n_iter,
                'model_state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_time_seconds': int(time.time() - time0),
                'epoch': epoch
            }, checkpoint_path)
            cprint(f"---- Save : {checkpoint_path} ----", "red")

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Uncomment if needed to force GPU visibility

    dataset = "PUGAN"
    check_name = ""
    model_path = f"./exp_{dataset.lower()}/{dataset}/logs/checkpoint/{check_name}"
    alpha = 1.0
    gamma = 0.5
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-a', '--alpha', type=int, default=alpha)
    parser.add_argument('-g', '--gamma', type=int, default=gamma)
    parser.add_argument('-m', '--model_path', type=str, default=model_path)
    args = parser.parse_args()

    args.config = f"./exp_configs/{args.dataset}.json"
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    global train_config
    global pointnet_config
    global diffusion_config
    global trainset_config
    global diffusion_hyperparams

    train_config = config["train_config"]
    pointnet_config = config["pointnet_config"]
    diffusion_config = config["diffusion_config"]
    if train_config['dataset'] == 'PU1K':
        trainset_config = config["pu1k_dataset_config"]
    elif train_config['dataset'] == 'PUGAN':
        trainset_config = config['pugan_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    train(
        args.config,
        args.model_path,
        **train_config
    )