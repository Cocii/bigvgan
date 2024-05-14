import warnings

import sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, MAX_WAV_VALUE
from models import BigVGAN, MultiPeriodDiscriminator, MultiResolutionDiscriminator,\
    feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, plot_spectrogram_clipped, scan_checkpoint, load_checkpoint, save_checkpoint, save_audio
import torchaudio as ta
from pesq import pesq
from tqdm import tqdm
import auraloss
from rawnet_model import RawNet
from tssd_model import SSDNet1D
from AASIST import aasist
import torch.nn as nn
import yaml
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
torch.backends.cudnn.benchmark = False

def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value {}'.format(val))

def compute_eer(label, pred, pos_label):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=pos_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

def cal_roc_eer(probs):
    """
    probs: tensor, number of samples * 3, containing softmax probabilities
    row wise: [genuine prob, fake prob, label]
    TP: True Fake
    FP: False Fake
    """
    all_labels = probs[:, 2]
    zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
    one_index = torch.nonzero(all_labels).squeeze(-1)
    zero_probs = probs[zero_index, 0]
    one_probs = probs[one_index, 0]

    threshold_index = torch.linspace(-0.1, 1.01, 10000)
    tpr = torch.zeros(len(threshold_index),)
    fpr = torch.zeros(len(threshold_index),)
    cnt = 0
    for i in threshold_index:
        tpr[cnt] = one_probs.le(i).sum().item()/len(one_probs)
        fpr[cnt] = zero_probs.le(i).sum().item()/len(zero_probs)
        cnt += 1

    sum_rate = tpr + fpr
    distance_to_one = torch.abs(sum_rate - 1)
    eer_index = distance_to_one.argmin(dim=0).item()
    out_eer = 0.5*(fpr[eer_index] + 1 - tpr[eer_index]).numpy()
    return out_eer

def cal_accuracy(probs):
    correct = 0
    total = 0 
    for prob in probs:
        prediction = 1 if prob[0] < prob[1] else 0
        label = prob[2]
        if prediction == label:
            correct += 1
        total += 1
            
    accuracy = correct / total
    return accuracy

def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    # set seed and device
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))
    current_script_directory = os.path.dirname(os.path.abspath(__file__))

    ## define rawnet2 detector
    config_path = os.path.join(current_script_directory, "configs", 'rawnet_config.yaml')
    config_rawnet = read_yaml(config_path)
    rawnet = RawNet(config_rawnet['model'],device)
    rawnet = (rawnet).to(device)
    rawnet.apply(init_weights)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    ## define tssdnet
    tssdnet = SSDNet1D()
    tssdnet = tssdnet.to(device)
    num_total_learnable_params = sum(i.numel() for i in tssdnet.parameters() if i.requires_grad)
    print('Number of learnable params: {}.'.format(num_total_learnable_params))
    optim_tssd = optim.Adam(tssdnet.parameters(), lr=0.001, weight_decay = 0.0001)

    ## define aasistnet
    aasist_config_path = os.path.join(current_script_directory, "configs", 'aasis_config.conf')
    with open(aasist_config_path, "r") as f_json:
        aasist_config = json.loads(f_json.read())
    aasistnet = aasist(aasist_config["model_config"]).to(device)
    aasist_optim_config = aasist_config["optim_config"]
    print("aasistnet params: {}".format(sum(p.numel() for p in aasistnet.parameters())))

    # define BigVGAN generator
    generator = BigVGAN(h).to(device)
    print("Generator params: {}".format(sum(p.numel() for p in generator.parameters())))

    # define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)
    print("Discriminator mpd params: {}".format(sum(p.numel() for p in mpd.parameters())))

    # define additional discriminators. BigVGAN uses MRD as default
    mrd = MultiResolutionDiscriminator(h).to(device)
    print("Discriminator mrd params: {}".format(sum(p.numel() for p in mrd.parameters())))

    # create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        # print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(os.path.join(a.checkpoint_path, "generator"), 'g_')
        cp_do = scan_checkpoint(os.path.join(a.checkpoint_path, "discriminator"), 'do_')
        cp_tssd = scan_checkpoint(os.path.join(a.checkpoint_path, "tssd"), 'tssd_')
        cp_rawnet = scan_checkpoint(os.path.join(a.checkpoint_path, "rawnet"), 'rawnet_')
        cp_aasist = scan_checkpoint(os.path.join(a.checkpoint_path, "aasist"), 'aasist_')
        print("cp_g: ", cp_g)
        print("cp_do: ", cp_do)
        print("cp_tssd: ", cp_tssd)
        print("cp_rawnet: ", cp_rawnet)
        print("cp_aasist: ", cp_aasist)

    # load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        mrd.load_state_dict(state_dict_do['mrd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if cp_tssd is None or cp_rawnet is None or cp_aasist is None:
        state_dict_tssd = None
        state_dict_rawnet = None
        state_dict_aasist = None
    else:
        state_dict_tssd = load_checkpoint(cp_tssd, device)
        state_dict_rawnet = load_checkpoint(cp_rawnet, device)
        state_dict_aasist = load_checkpoint(cp_aasist, device)
        tssdnet.load_state_dict(state_dict_tssd['model_state_dict'])
        rawnet.load_state_dict(state_dict_rawnet['model_state_dict'])
        aasistnet.load_state_dict(state_dict_aasist['model_state_dict'])

    

    # initialize DDP, optimizers, and schedulers
    # multi gpus
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_rawnet = torch.optim.Adam(rawnet.parameters(), lr=config_rawnet['lr'], weight_decay = 0.0001)
    optim_aasist = torch.optim.Adam(aasistnet.parameters(),
                                     lr=aasist_optim_config['base_lr'],
                                     betas=aasist_optim_config['betas'],
                                     weight_decay=aasist_optim_config['weight_decay'],
                                     amsgrad=str_to_bool(
                                         aasist_optim_config['amsgrad']))
    
    ## optimizers definition
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    if state_dict_rawnet is not None and state_dict_tssd is not None and state_dict_aasist is not None:
        optim_rawnet.load_state_dict(state_dict_rawnet['optim_rawnet'])
        optim_aasist.load_state_dict(state_dict_aasist['optimizer_state_dict'])
        optim_tssd.load_state_dict(state_dict_tssd['optimizer_state_dict'])


    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_tssd = optim.lr_scheduler.ExponentialLR(optim_tssd, gamma=0.95)
    scheduler_rawnet = optim.lr_scheduler.ExponentialLR(optim_rawnet, gamma=0.95)


    # define training and validation datasets
    # unseen_validation_filelist will contain sample filepaths outside the seen training & validation dataset
    # example: trained on LibriTTS, validate on VCTK
    training_filelist, validation_filelist, list_unseen_validation_filelist = get_dataset_filelist(a)


    aasist_optim_config["epochs"] = 1
    aasist_optim_config["steps_per_epoch"] = len(training_filelist)
    total_steps = aasist_optim_config['epochs'] * \
            aasist_optim_config['steps_per_epoch']
    scheduler_aasist = torch.optim.lr_scheduler.LambdaLR(
            optim_aasist,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                aasist_optim_config['lr_min'] / aasist_optim_config['base_lr']))
    
    trainset = MelDataset(training_filelist, h, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, is_seen=True)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir, is_seen=True)
        validation_loader = DataLoader(validset, num_workers=h.num_workers, shuffle=False,
                                       sampler=None,
                                       batch_size= h.batch_size,
                                       pin_memory=True,
                                       drop_last=True)

        list_unseen_validset = []
        list_unseen_validation_loader = []
        for i in range(len(list_unseen_validation_filelist)):
            unseen_validset = MelDataset(list_unseen_validation_filelist[i], h, h.segment_size, h.n_fft, h.num_mels,
                                         h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                                         fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                                         base_mels_path=a.input_mels_dir, is_seen=False)
            unseen_validation_loader = DataLoader(unseen_validset, num_workers=1, shuffle=False,
                                                  sampler=None,
                                                  batch_size=1,
                                                  pin_memory=True,
                                                  drop_last=True)
            list_unseen_validset.append(unseen_validset)
            list_unseen_validation_loader.append(unseen_validation_loader)

        # Tensorboard logger
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        if a.save_audio: # also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, 'samples'), exist_ok=True)

    # validation loop
    # "mode" parameter is automatically defined as (seen or unseen)_(name of the dataset)
    # if the name of the dataset contains "nonspeech", it skips PESQ calculation to prevent errors
    def validate(rank, a, h, loader, mode="seen"):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        tssdnet.eval()
        rawnet.eval()
        aasistnet.eval()
        torch.cuda.empty_cache()

        val_err_tot = 0
        val_pesq_tot = 0
        val_mrstft_tot = 0
        probs_tssds = []
        probs_rawnets = []
        probs_aasists = []

        # modules for evaluation metrics
        pesq_resampler = ta.transforms.Resample(h.sampling_rate, 16000).cuda()
        loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

        if a.save_audio: # also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode)), exist_ok=True)
            os.makedirs(os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps)), exist_ok=True)

        with torch.no_grad():
            print("step {} {} speaker validation...".format(steps, mode))

            # loop over validation set and compute metrics
            for j, batch in enumerate(tqdm(loader, desc='Processing batches')):
                x, y, _, y_mel = batch
                y = y.to(device)
                if hasattr(generator, 'module'):
                    y_g_hat = generator.module(x.to(device))
                else:
                    y_g_hat = generator(x.to(device))
                y_mel = y_mel.to(device, non_blocking=True)
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                            h.hop_size, h.win_size,
                                            h.fmin, h.fmax_for_loss)
                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                # PESQ calculation. only evaluate PESQ if it's speech signal (nonspeech PESQ will error out)
                if not "nonspeech" in mode: # skips if the name of dataset (in mode string) contains "nonspeech"
                    # resample to 16000 for pesq
                    y_16k = pesq_resampler(y)
                    y_g_hat_16k = pesq_resampler(y_g_hat.squeeze(1))
                    y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                    y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                    try:
                        val_pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, 'wb')
                    except:
                        pass

                # MRSTFT calculation
                val_mrstft_tot += loss_mrstft(y_g_hat.squeeze(1), y).item()

                # EER, AUC, balanced_accuracy calculation for TSSD, Rawnet, AASIST
                with torch.no_grad():
                    # labels preparation. 1 for fake, 0 for real
                    batch_size = y_g_hat.detach().size(0)
                    label_y_fake = torch.ones(batch_size).type(torch.int64).to(device) # 1 for fake
                    label_y_real = torch.zeros(batch_size).type(torch.int64).to(device) # 0 for real
                    samples_real = y.squeeze(1)
                    samples_fake = y_g_hat.detach().squeeze(1)
                    labels_y_mixed = torch.cat((label_y_real, label_y_fake), dim=0)
                    samples_mixed = torch.cat((samples_real, samples_fake), dim=0)
                    random_indices = torch.randperm(len(samples_mixed))
                    labels_random = labels_y_mixed[random_indices]
                    samples_random = samples_mixed[random_indices]

                    # TSSD
                    samples_tssd = samples_random.unsqueeze(1)
                    tssd_out = tssdnet(samples_tssd)
                    t1_tssd = F.softmax(tssd_out, dim=1)
                    t2_tssd = labels_random.unsqueeze(-1)
                    row = torch.cat((t1_tssd, t2_tssd), dim=1)
                    probs_tssd = torch.empty(0, 3).to(device)
                    probs_tssd = torch.cat((probs_tssd, row), dim=0)
                    probs_tssds.extend(probs_tssd.detach().cpu().tolist())

                    # Rawnet
                    rawnet_out = rawnet(samples_random)
                    t1_rawnet = F.softmax(rawnet_out, dim=1)
                    t2_rawnet = labels_random.unsqueeze(-1)
                    row = torch.cat((t1_rawnet, t2_rawnet), dim=1)
                    probs_rawnet = torch.empty(0, 3).to(device)
                    probs_rawnet = torch.cat((probs_rawnet, row), dim=0)
                    probs_rawnets.extend(probs_rawnet.detach().cpu().tolist())

                    # AASIST
                    aasist_out = aasistnet(samples_random.squeeze(1))
                    aasist_out = aasist_out[1]
                    t1_aasist = F.softmax(aasist_out, dim=1)
                    t2_aasist = labels_random.unsqueeze(-1)
                    row = torch.cat((t1_aasist, t2_aasist), dim=1)
                    probs_aasist = torch.empty(0, 3).to(device)
                    probs_aasist = torch.cat((probs_aasist, row), dim=0)
                    probs_aasists.extend(probs_aasist.detach().cpu().tolist())
                    
                # log audio and figures to Tensorboard
                if j % a.eval_subsample == 0:  # subsample every nth from validation set
                    if steps >= 0:
                        sw.add_audio('gt_{}/y_{}'.format(mode, j), y[0], steps, h.sampling_rate)
                        if a.save_audio: # also save audio to disk if --save_audio is set to True
                            save_audio(y[0], os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode), '{:04d}.wav'.format(j)), h.sampling_rate)
                        sw.add_figure('gt_{}/y_spec_{}'.format(mode, j), plot_spectrogram(x[0]), steps)

                    sw.add_audio('generated_{}/y_hat_{}'.format(mode, j), y_g_hat[0], steps, h.sampling_rate)
                    if a.save_audio: # also save audio to disk if --save_audio is set to True
                        save_audio(y_g_hat[0, 0], os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps), '{:04d}.wav'.format(j)), h.sampling_rate)
                    # spectrogram of synthesized audio
                    y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                h.sampling_rate, h.hop_size, h.win_size,
                                                h.fmin, h.fmax)
                    sw.add_figure('generated_{}/y_hat_spec_{}'.format(mode, j),
                                plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()[0]), steps)
                    
            val_err = val_err_tot / (j + 1)
            val_pesq = val_pesq_tot / (j + 1)
            val_mrstft = val_mrstft_tot / (j + 1)

            # calculate EER, AUC, balanced_accuracy for TSSD
            eer_tssd = cal_roc_eer(torch.tensor(probs_tssds))
            auc_tssd = roc_auc_score(np.array(probs_tssds)[:, 2], np.array(probs_tssds)[:, 0])
            bal_acc_tssd = balanced_accuracy_score(np.array(probs_tssds)[:, 2], np.argmax(np.array(probs_tssds)[:, :2], axis=1))

            # calculate EER, AUC, balanced_accuracy for Rawnet
            eer_rawnet = cal_roc_eer(torch.tensor(probs_rawnets))
            auc_rawnet = roc_auc_score(np.array(probs_rawnets)[:, 2], np.array(probs_rawnets)[:, 0])
            bal_acc_rawnet = balanced_accuracy_score(np.array(probs_rawnets)[:, 2], np.argmax(np.array(probs_rawnets)[:, :2], axis=1))

            # calculate EER, AUC, balanced_accuracy for AASIST  
            eer_aasist = cal_roc_eer(torch.tensor(probs_aasists))
            auc_aasist = roc_auc_score(np.array(probs_aasists)[:, 2], np.array(probs_aasists)[:, 0])
            bal_acc_aasist = balanced_accuracy_score(np.array(probs_aasists)[:, 2], np.argmax(np.array(probs_aasists)[:, :2], axis=1))

            # check eer value 
            eer_tssd0 = compute_eer(np.array(probs_tssds)[:, 2], np.array(probs_tssds)[:, 0], 0)
            eer_aasist0 = compute_eer(np.array(probs_aasists)[:, 2], np.array(probs_aasists)[:, 0], 0)
            eer_rawnet0 = compute_eer(np.array(probs_rawnets)[:, 2], np.array(probs_rawnets)[:, 0], 0)

            # log part
            epoch_str = 'Steps : {:d}, '.format(
                            steps)
            tssd_str = '    TSSD_EER : {:4.5f}, TSSD_AUC : {:4.5f}, TSSD_Bal_acc : {:4.5f}, '.format(
                            eer_tssd, 1-auc_tssd, bal_acc_tssd)
            raw_str = '    Rawnet_EER : {:4.5f}, Rawnet_AUC : {:4.5f}, Rawnet_Bal_acc : {:4.5f}, '.format(
                            eer_rawnet, 1-auc_rawnet, bal_acc_rawnet)
            # add_str = '    TSSD_EER0 : {:4.5f}, Rawnet_EER0 : {:4.5f}'.format(eer_tssd0, eer_rawnet0)
            aasist_str = '    Aasist_EER : {:4.5f}, Aasist_AUC : {:4.5f}, Aasist_Bal_acc : {:4.5f}, '.format(
                            eer_aasist, 1-auc_aasist, bal_acc_aasist)
            add_str = '    TSSD_EER0 : {:4.5f}, Aasist_EER0 : {:4.5f}, Rawnet_EER0 : {:4.5f}'.format(eer_tssd0, eer_aasist0, eer_rawnet0)
            
            log_str = epoch_str+'\n'+tssd_str+'\n'+aasist_str+'\n'+raw_str+'\n'+add_str

            log_dir = os.path.join(a.checkpoint_path, 'loss_log')
            os.makedirs(log_dir, exist_ok=True)
            txt_file = os.path.join(log_dir, 'eer_log.txt')
            with open(txt_file, 'a') as f:
                f.write(log_str + '\n')

            # log evaluation metrics to Tensorboard
            sw.add_scalar("validation_{}/mel_spec_error".format(mode), val_err, steps)
            sw.add_scalar("validation_{}/pesq".format(mode), val_pesq, steps)
            sw.add_scalar("validation_{}/mrstft".format(mode), val_mrstft, steps)
            sw.add_scalar("validation_{}/tssd_eer".format(mode), eer_tssd, steps)
            sw.add_scalar("validation_{}/tssd_auc".format(mode), auc_tssd, steps)
            sw.add_scalar("validation_{}/tssd_bal_acc".format(mode), bal_acc_tssd, steps)
            sw.add_scalar("validation_{}/rawnet_eer".format(mode), eer_rawnet, steps)
            sw.add_scalar("validation_{}/rawnet_auc".format(mode), auc_rawnet, steps)
            sw.add_scalar("validation_{}/rawnet_bal_acc".format(mode), bal_acc_rawnet, steps)
            sw.add_scalar("validation_{}/aasist_eer".format(mode), eer_aasist, steps)
            sw.add_scalar("validation_{}/aasist_auc".format(mode), auc_aasist, steps)
            sw.add_scalar("validation_{}/aasist_bal_acc".format(mode), bal_acc_aasist, steps)

        generator.train()
        tssdnet.train()
        rawnet.train()
        aasistnet.train()
        torch.cuda.empty_cache()
    # if the checkpoint is loaded, start with validation loop
    if steps != 0 and rank == 0 and not a.debug:
        if not a.skip_seen:
            validate(rank, a, h, validation_loader, # validation_loader, train_loader
                     mode="seen")
        # for i in range(len(list_unseen_validation_loader)):
        #     validate(rank, a, h, list_unseen_validation_loader[i],
        #              mode="unseen_{}".format(list_unseen_validation_loader[i].dataset.name))
    # exit the script if --evaluate is set to True
    if a.evaluate:
        exit()

    # main training loop
    rawnet.train()
    aasistnet.train()
    tssdnet.train()
    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(tqdm(train_loader, desc='Processing batches')):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)
            y_g_hat = generator(x)
            # y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
            #                               h.fmin, h.fmax_for_loss)

            ########################################
            #                 TSSD                 #
            ######################################## 
            # 1 for fake, 0 for real
            
            batch_size = y_g_hat.detach().size(0)
            label_y_fake = torch.ones(batch_size).type(torch.int64).to(device) # 1 for fake
            label_y_real = torch.zeros(batch_size).type(torch.int64).to(device) # 0 for real
            samples_real = y.squeeze(1)
            samples_fake = y_g_hat.detach().squeeze(1)
            labels_y_mixed = torch.cat((label_y_real, label_y_fake), dim=0)
            samples_mixed = torch.cat((samples_real, samples_fake), dim=0)
            
            # compute tssd loss
            optim_tssd.zero_grad()
            random_indices = torch.randperm(len(samples_mixed))
            labels_random = labels_y_mixed[random_indices]
            samples_random = samples_mixed[random_indices]
            samples_tssd = samples_random.unsqueeze(1)
            tssd_out = tssdnet(samples_tssd)
            tssd_loss = criterion(tssd_out, labels_random)
            tssd_loss.backward()
            optim_tssd.step()

            ########################################
            #                Rawnet                #
            ########################################   

            # # 1 for fake, 0 for real
            optim_rawnet.zero_grad()
            rawnet_out = rawnet(samples_random)
            rawnet_loss = criterion(rawnet_out, labels_random)
            rawnet_loss.backward()
            optim_rawnet.step()
            

            ########################################
            #                Aasist                #
            ########################################   

            # 1 for fake, 0 for real
            optim_aasist.zero_grad()
            aasist_out = aasistnet(samples_random.squeeze(1))
            # aasist_out[0] representation, aasist_out[1] is result of seperation
            aasist_out = aasist_out[1]
            aasist_loss = criterion(aasist_out, labels_random)
            aasist_loss.backward()
            optim_aasist.step()


            ########################################
            #                BigvGAN               #
            ######################################## 
            # # discriminator 
            # optim_d.zero_grad()

            # # MPD
            # y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            # loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # # MRD
            # y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            # loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # loss_disc_all = loss_disc_s + loss_disc_f

            # # whether to freeze D for initial training steps
            # if steps >= a.freeze_step:
            #     loss_disc_all.backward()
            #     grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000.)
            #     grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000.)
            #     optim_d.step()
            # else:
            #     print("WARNING: skipping D training for the first {} steps".format(a.freeze_step))
            #     grad_norm_mpd = 0.
            #     grad_norm_mrd = 0.

            # # generator
            # optim_g.zero_grad()

            # # L1 Mel-Spectrogram Loss
            # loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # # MPD loss
            # y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            # loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            # loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            # # MRD loss
            # y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            # loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            # loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            # if steps >= a.freeze_step:
            #     loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            # else:
            #     print("WARNING: using regression loss only for G for the first {} steps".format(a.freeze_step))
            #     loss_gen_all = loss_mel

            # loss_gen_all.backward()
            # grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1000.)
        
            # optim_g.step()

            if rank == 0:

                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    # with torch.no_grad():
                        # mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    log_str = 'Steps : {:d}'.format(steps)
                    # log_gen_loss = ', Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(
                        # loss_gen_all, mel_error, time.time() - start_b)
                    log_tssd_loss = ', TSSD_loss : {:4.3f}, Aasist_loss : {:4.3f}'.format(tssd_loss, aasist_loss)
                    log_rawnet_loss = ', Rawnet_loss : {:4.3f}'.format(rawnet_loss)
                    log_str = log_str + log_tssd_loss + log_rawnet_loss
                    print(log_str)

                    # Write to txt file
                    log_dir = os.path.join(a.checkpoint_path, 'loss_log')
                    os.makedirs(log_dir, exist_ok=True)
                    txt_file = os.path.join(log_dir, 'training_log.txt')
                    with open(txt_file, 'a') as f:
                        f.write(log_str + '\n')
                        
                # checkpointing
                if steps % (a.checkpoint_interval) == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(os.path.join(a.checkpoint_path, "generator"), steps)
                    os.makedirs(os.path.join(a.checkpoint_path, "generator"), exist_ok=True)
                    save_checkpoint(checkpoint_path, {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})

                    checkpoint_path = "{}/do_{:08d}".format(os.path.join(a.checkpoint_path, "discriminator"), steps)
                    os.makedirs(os.path.join(a.checkpoint_path, "discriminator"), exist_ok=True)
                    save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                    'mrd': (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                                                    'optim_g': optim_g.state_dict(),
                                                    'optim_d': optim_d.state_dict(),
                                                    'steps': steps,
                                                    'epoch': epoch})

                    checkpoint_path = "{}/tssd_{:08d}".format(os.path.join(a.checkpoint_path, "tssd"), steps)
                    os.makedirs(os.path.join(a.checkpoint_path, "tssd"), exist_ok=True)
                    save_checkpoint(checkpoint_path, {'epoch': epoch,
                                                    'model_state_dict': tssdnet.state_dict(),
                                                    'optimizer_state_dict': optim_tssd.state_dict(),
                                                    'scheduler_state_dict': scheduler_tssd.state_dict()})
                    
                    checkpoint_path = "{}/aasist_{:08d}".format(os.path.join(a.checkpoint_path, "aasist"), steps)
                    os.makedirs(os.path.join(a.checkpoint_path, "aasist"), exist_ok=True)
                    save_checkpoint(checkpoint_path, {'epoch': epoch,
                                                    'model_state_dict': aasistnet.state_dict(),
                                                    'optimizer_state_dict': optim_aasist.state_dict(),
                                                    'scheduler_state_dict': scheduler_aasist.state_dict()})
                    
                    checkpoint_path = "{}/rawnet_{:08d}".format(os.path.join(a.checkpoint_path, "rawnet"), steps)
                    os.makedirs(os.path.join(a.checkpoint_path, "rawnet"), exist_ok=True)
                    save_checkpoint(checkpoint_path, {'model_state_dict': rawnet.state_dict(),
                                                    'optim_rawnet': optim_rawnet.state_dict(),
                                                    'scheduler_state_dict': scheduler_rawnet.state_dict()})
                    
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    # sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    # sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    # sw.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                    # sw.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                    # sw.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                    # sw.add_scalar("training/grad_norm_mpd", grad_norm_mpd, steps)
                    # sw.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                    # sw.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                    # sw.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                    # sw.add_scalar("training/grad_norm_mrd", grad_norm_mrd, steps)
                    # sw.add_scalar("training/grad_norm_g", grad_norm_g, steps)
                    sw.add_scalar("training/learning_rate_d", scheduler_d.get_last_lr()[0], steps)
                    sw.add_scalar("training/learning_rate_g", scheduler_g.get_last_lr()[0], steps)
                    sw.add_scalar("training/epoch", epoch+1, steps)

                # validation
                if steps % a.validation_interval == 0:
                    # plot training input x so far used
                    for i_x in range(x.shape[0]):
                        sw.add_figure('training_input/x_{}'.format(i_x), plot_spectrogram(x[i_x].cpu()), steps)
                        sw.add_audio('training_input/y_{}'.format(i_x), y[i_x][0], steps, h.sampling_rate)

                    # seen and unseen speakers validation loops
                    if not a.debug and steps != 0:
                        validate(rank, a, h, validation_loader,
                                 mode="seen")
                        # for i in range(len(list_unseen_validation_loader)):
                        #     validate(rank, a, h, list_unseen_validation_loader[i],
                        #              mode="unseen_{}".format(list_unseen_validation_loader[i].dataset.name))
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        scheduler_tssd.step()
        scheduler_aasist.step()
        scheduler_rawnet.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LibriTTS')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LibriTTS/train-full.txt')
    parser.add_argument('--input_validation_file', default='LibriTTS/val-full.txt')
    parser.add_argument('--list_input_unseen_wavs_dir', nargs='+', default=['LibriTTS', 'LibriTTS'])
    parser.add_argument('--list_input_unseen_validation_file', nargs='+', default=['LibriTTS/dev-clean.txt', 'LibriTTS/dev-other.txt'])
    parser.add_argument('--checkpoint_path', default='exp/bigvgan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=100000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=50000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=50000, type=int)
    parser.add_argument('--freeze_step', default=0, type=int,
                        help='freeze D for the first specified steps. G only uses regression loss for these steps.')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--debug', default=False, type=bool,
                        help="debug mode. skips validation loop throughout training")
    parser.add_argument('--evaluate', default=False, type=bool,
                        help="only run evaluation from checkpoint and exit")
    parser.add_argument('--eval_subsample', default=5, type=int,
                        help="subsampling during evaluation loop")
    parser.add_argument('--skip_seen', default=False, type=bool,
                        help="skip seen dataset. useful for test set inference")
    parser.add_argument('--save_audio', default=False, type=bool,
                        help="save audio of test set inference to disk")
    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
