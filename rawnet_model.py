import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
import sys
from audio_utils import *
import random
from audiomentations import Compose, AddGaussianNoise, Mp3Compression, HighPassFilter
import pandas as pd


def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)


def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in train_loader:

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct / num_total) * 100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class LoadTrainData_RawNet(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs=16000):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

        self.balance_batch = False

        df = pd.DataFrame(labels.items(), columns=['path', 'label'])
        self.real_list = list(df[df['label'] == 0]['path'])
        self.fake_list = list(df[df['label'] == 1]['path'])

        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=.3)
        ])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        if self.balance_batch:
            if index % 2 == 0:
                track_name = random.choice(self.real_list)
            else:
                track_name = random.choice(self.fake_list)
        else:
            track_name = self.list_IDs[index]

        x, fs = read_audio(track_name, fs=self.fs, trim=False)
        y = self.labels[track_name]
        audio_len = len(x)
        # x_win = pad(x, self.cut)

        # # data augmentation
        # x = self.augment_noise(samples=x, sample_rate=fs)
        # x = self.augment_mp3(samples=x, sample_rate=fs)
        # x = self.augment_highpass(samples=x, sample_rate=fs)

        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        last_valid_start_sample = audio_len - self.win_len_samples
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        x_win = x[start_sample : start_sample + self.win_len_samples]
        x_win = Tensor(x_win)

        return x_win, y


class LoadEvalData_RawNet(Dataset):
    def __init__(self, list_IDs, win_len, fs=16000):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # self.cut = 70000
        track = self.list_IDs[index]
        x, fs = read_audio(track, fs=self.fs, trim=False)
        audio_len = len(x)

        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        # TEST ON THE MIDDLE WINDOW
        start_sample = int(0.5*(len(x) - self.win_len_samples))
        x_win = x[start_sample: start_sample + self.win_len_samples]
        x_inp = Tensor(x_win)

        return x_inp, track


class LoadEvalData_RawNet_MULTI(Dataset):
    def __init__(self, list_IDs, win_len, fs=16000):
    # def __init__(self, list_IDs, list_IDs_real, win_len, fs=16000):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.list_len = len(list_IDs)

        # self.list_IDs_real = list_IDs_real
        # self.list_len_real = len(list_IDs_real)

        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = self.win_len * self.fs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 70000
        track = self.list_IDs[index]
        x, fs = read_audio(track, trim=True)

        idx_2 = random.randint(0, self.list_len - 1)
        track2 = self.list_IDs[idx_2]
        # idx_2 = random.randint(0, self.list_len_real-1)
        # track2 = self.list_IDs_real[idx_2]
        x2, fs = read_audio(track2, trim=True)

        x = mix_tracks(x, x2)

        audio_len = len(x)

        # We should have already discarded all audio files less than one second
        if audio_len < self.win_len_samples:
            # x = np.pad(x, ((0,0), (0, self.win_len_samples - audio_len)))
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        x_win = x[:self.win_len_samples]
        x_inp = Tensor(x_win)

        return x_inp, track


class LoadEvalDataWindow(Dataset):
    def __init__(self, wav_path, win_len, hop_size, X, fsamp):
        '''self.list_IDs	: list of strings (each string: utt key) '''

        self.wav_path = wav_path
        self.win_len = win_len
        self.hop_size = hop_size
        self.X = X
        self.fs = fsamp
        # self.X, self.fs = read_audio(wav_path, dur=180)

    def __len__(self):
        return int(np.floor((len(self.X) - self.win_len*self.fs)/(self.hop_size*self.fs))+1)

    def __getitem__(self, index):

        x_win = self.X[int(index*self.hop_size*self.fs):int(round((index*self.hop_size + self.win_len) * self.fs))]
        x_inp = Tensor(x_win)

        return x_inp, index


class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=22050,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)  # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x):
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal)

        band_pass_filter = self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()

        self.device = device

        self.Sinc_conv = SincConv(device=self.device,
                                  out_channels=d_args['filts'][0],
                                  kernel_size=d_args['first_conv'],
                                  in_channels=d_args['in_channels']
                                  )

        # self.Sinc_conv = SincConv(out_channels=d_args['filts'][0],
        #                            kernel_size=d_args['first_conv'])

        self.first_bn = nn.BatchNorm1d(num_features=d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True))
        self.block1 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features=d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size=d_args['filts'][2][-1],
                          hidden_size=d_args['gru_node'],
                          num_layers=d_args['nb_gru_layer'],
                          batch_first=True)

        self.fc1_gru = nn.Linear(in_features=d_args['gru_node'],
                                 out_features=d_args['nb_fc_node'])

        self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'],
                                 out_features=d_args['nb_classes'], bias=True)

        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):

        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)  # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2  # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)  # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3  # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)  # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4  # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)  # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5  # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        output = self.logsoftmax(x)

        return output

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []

        l_fc.append(nn.Linear(in_features=in_features,
                              out_features=l_out_features))

        return nn.Sequential(*l_fc)

    def _make_layer(self, nb_blocks, nb_filts, first=False):
        layers = []
        # def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts=nb_filts,
                                         first=first))
            if i == 0: nb_filts[0] = nb_filts[1]

        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn=None):
        if print_fn == None: printfn = print
        model = self

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()

        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)