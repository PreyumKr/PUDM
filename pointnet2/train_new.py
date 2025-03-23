import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from dataset import get_dataloader
from util import calc_diffusion_hyperparams, find_max_epoch, print_size, set_seed
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition  # Generator
from models.discriminator import PointCloudDiscriminator  # New discriminator model
from shutil import copyfile
import copy
from json_reader import replace_list_with_string_in_a_dict, restore_string_to_list_in_a_dict

def split_data(data):
    label = data['label'].cuda()
    X = data['complete'].cuda()  # Real complete point cloud
    condition = data['partial'].cuda()  # Condition (partial point cloud)
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

    # Initialize models
    generator = PointNet2CloudCondition(pointnet_config).cuda()  # Generator
    discriminator = PointCloudDiscriminator().cuda()  # New discriminator
    generator.train()
    discriminator.train()

    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Load checkpoint if available
    time0 = time.time()
    epoch = 0
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        epoch = int(checkpoint['epoch'])
        time0 -= checkpoint['training_time_seconds']
        print(f"---- Loading : {model_path} ----")
    except:
        print('No valid checkpoint found, training from scratch.', flush=True)

    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs)

    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()  # GAN loss
    reconstruction_loss = nn.MSELoss()  # To ensure fidelity to the target

    n_iter = epoch * loader_len
    while n_iter < n_iters + 1:
        epoch += 1
        for data in trainloader:
            X, condition, label = split_data(data)

            # Sample timestep and noise
            t = torch.randint(0, diffusion_hyperparams["T"], (X.size(0),), device=X.device).long()
            noise = torch.randn_like(X)
            noisy_X = diffusion_hyperparams["sqrt_alphas_cumprod"][t, None, None] * X + \
                      diffusion_hyperparams["sqrt_one_minus_alphas_cumprod"][t, None, None] * noise

            # Train Discriminator
            d_optimizer.zero_grad()
            real_output = discriminator(X)
            fake_output = discriminator(generator(noisy_X, t, condition).detach())
            d_loss_real = adversarial_loss(real_output, torch.ones_like(real_output))
            d_loss_fake = adversarial_loss(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            generated_X = generator(noisy_X, t, condition)
            d_output = discriminator(generated_X)
            g_adv_loss = adversarial_loss(d_output, torch.ones_like(d_output))
            g_rec_loss = reconstruction_loss(generated_X, X)
            g_loss = g_adv_loss + 0.1 * g_rec_loss  # Balance adversarial and reconstruction losses
            g_loss.backward()
            g_optimizer.step()

            # Logging
            if n_iter % iters_per_logging == 0:
                cprint("[{}]\tepoch({}/{})\titer({}/{})\tD Loss: {:.6f}\tG Loss: {:.6f}".format(
                    time.ctime(), epoch, n_epochs, n_iter, n_iters, d_loss.item(), g_loss.item()), "blue")
                tb.add_scalar("D_Loss", d_loss.item(), n_iter)
                tb.add_scalar("G_Loss", g_loss.item(), n_iter)

            n_iter += 1

        # Save checkpoint
        if epoch % epochs_per_ckpt == 0:
            checkpoint_name = f'ddgan_ckpt_{epoch}_{g_loss.item():.6f}.pkl'
            checkpoint_path = os.path.join(output_directory, checkpoint_name)
            torch.save({
                'iter': n_iter,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'training_time_seconds': int(time.time() - time0),
                'epoch': epoch
            }, checkpoint_path)
            cprint(f"---- Save : {checkpoint_path} ----", "red")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Config
    dataset = "PUGAN"
    check_name = ""
    model_path = f"./exp_{dataset.lower()}/{dataset}/logs/checkpoint/{check_name}"
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-m', '--model_path', type=str, default=model_path)
    args = parser.parse_args()

    args.config = f"./exp_configs/{args.dataset}.json"
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)
    print('The configuration is:')
    print(json.dumps(replace_list_with_string_in_a_dict(copy.deepcopy(config)), indent=4))

    # Global variables
    global train_config, pointnet_config, diffusion_config, trainset_config, diffusion_hyperparams
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

    train(args.config, args.model_path, **train_config)