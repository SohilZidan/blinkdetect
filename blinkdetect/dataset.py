import json
from random import sample
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BlinkDataSet(Dataset):
    def __init__(self, annotation_path, transform=False):
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

        if transform:
            self.tsfrm = transforms.Compose([ToTensor(), Normalize((0,1))])
        else:
            self.tsfrm = transforms.Compose([ToTensor()])

    def __len__(self):
        return len(self.annotations)


class BlinkDataset4C(BlinkDataSet):
    def __init__(self, annotation_path, transform=False):
        super(BlinkDataset4C, self).__init__(annotation_path, transform)


    def __getitem__(self, idx):
        sample = self.annotations[idx]
        _pid=sample['pid']
        _rng=sample['range']
        _signals = self.tsfrm(sample)
        _input1 = _signals['eyelids_dist']
        _input2 = _signals['std_r']
        _input3 = _signals['std_g']
        _input4 = _signals['std_b']

        features = torch.stack([_input1, _input2, _input3, _input4], dim=0)

        label = _signals['is_blink']

        if int(_signals['is_blink']) == 0:
            duration = torch.tensor([0], dtype=torch.float32)
        else:
            duration = torch.tensor([_signals['blink_length']], dtype=torch.float32).log_()/3.1355

        return features, label, duration, _pid, _rng, torch.tensor(sample['yaws']),torch.tensor(sample['pitchs']) # 3.1355 is the max of log(duration)


class BlinkDataset1C(BlinkDataSet):
    def __init__(self, annotation_path, transform=False):
        super(BlinkDataset1C, self).__init__(annotation_path, transform)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        _pid=sample['pid']
        _rng=sample['range']
        _signals = self.tsfrm(sample)
        _input1 = _signals['eyelids_dist']

        features = torch.stack([_input1], dim=0)
        label = _signals['is_blink']

        if _signals['blink_length'] == 0:
            duration = torch.tensor([_signals['blink_length']], dtype=torch.float32)
        else:
            duration = torch.tensor([_signals['blink_length']], dtype=torch.float32).log_()/3.1355

        return features, label, duration, _pid, _rng, torch.tensor(sample['yaws']),torch.tensor(sample['pitchs']) # 3.1355 is the max of ln(duration)


class BlinkDataset2C(BlinkDataSet):    
    def __init__(self, annotation_path, transform=False):
        super(BlinkDataset2C, self).__init__(annotation_path, transform)


    def __getitem__(self, idx):
        sample = self.annotations[idx]

        _pid=sample['pid']
        _rng=sample['range']

        _signals = self.tsfrm(sample)

        _input1 = _signals['eyelids_dist'] if isinstance( _signals['eyelids_dist'], torch.Tensor) else torch.tensor(_signals['eyelids_dist'])
        _input2 = _signals['std_r']
        _input3 = _signals['std_g']
        _input4 = _signals['std_b']
        # 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue 
        avg_std = torch.mean(torch.stack([0.299* _input2, 0.587 * _input3, 0.114 * _input4], dim=0), dim=0)

        features = torch.stack([_input1, avg_std], dim=0)

        label = _signals['is_blink']

        if _signals['blink_length'] == 0:
            duration = torch.tensor([_signals['blink_length']], dtype=torch.float32)
        else:
            duration = torch.tensor([_signals['blink_length']], dtype=torch.float32).log_() / 3.1355

        return features, label, duration, _pid, _rng, torch.tensor(sample['yaws']), torch.tensor(sample['pitchs']) # 3.1355 is the max of log(duration)


class ToTensor(object):
    features_list = ["std_r", "std_g", "std_b", "eyelids_dist", "iris_diameter"]
    def __call__(self, sample):
        for ftr in self.features_list:
            sample[ftr] = torch.tensor(sample[ftr])
        return sample


class Normalize(object):
    """Normalized
    """
    def __init__(self, feature_range=(0,1)):
        self._min = feature_range[0]
        self._max = feature_range[1]


    def __call__(self, sample):
        sample['eyelids_dist'] = torch.div(sample['eyelids_dist'], sample['iris_diameter'])
        return sample
