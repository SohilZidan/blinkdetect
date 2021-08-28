import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BlinkDataSet(Dataset):
  def __init__(self, annotation_path, transform=False):
    with open(annotation_path, "r") as f:
      self.annotations = json.load(f)
    # 
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
    # 
    _signals = self.tsfrm(self.annotations[idx])
    _input1 = _signals['eyelids_dist']
    _input2 = _signals['std_r']
    _input3 = _signals['std_g']
    _input4 = _signals['std_b']
    # 
    features = torch.stack([_input1, _input2, _input3, _input4], dim=0)
    label = torch.tensor([_signals['is_blink']])
    # 
    if _signals['blink_length'] == 0:
      duration = torch.tensor([_signals['blink_length']], dtype=torch.float32)
    else:
      duration = torch.tensor([_signals['blink_length']], dtype=torch.float32).log_()/3.1355
    # 
    return features, label, duration # 3.1355 is the max of log(duration)



class BlinkDataset1C(BlinkDataSet):
    
  def __init__(self, annotation_path, transform=False):
    super(BlinkDataset1C, self).__init__(annotation_path, transform)

  def __getitem__(self, idx):
    # 
    _signals = self.tsfrm(self.annotations[idx])
    _input1 = _signals['eyelids_dist']
    # 
    features = torch.stack([_input1], dim=0)
    label = torch.tensor([_signals['is_blink']])
    # 
    if _signals['blink_length'] == 0:
      duration = torch.tensor([_signals['blink_length']], dtype=torch.float32)
    else:
      duration = torch.tensor([_signals['blink_length']], dtype=torch.float32).log_()/3.1355
    return features, label, duration # 3.1355 is the max of log(duration)




class BlinkDataset2C(Dataset):
    
  def __init__(self, annotation_path, transform=False):
    super(BlinkDataset1C, self).__init__(annotation_path, transform)

  def __getitem__(self, idx):
    # 
    _signals = self.tsfrm(self.annotations[idx])
    _input1 = _signals['eyelids_dist'] if isinstance( _signals['eyelids_dist'], torch.Tensor) else torch.tensor(_signals['eyelids_dist'])
    _input2 = _signals['std_r']
    _input3 = _signals['std_g']
    _input4 = _signals['std_b']
    avg_std = torch.mean(torch.stack([_input2, _input3, _input4], dim=0), dim=0)
    # 
    features = torch.stack([_input1, avg_std], dim=0)
    label = torch.tensor([_signals['is_blink']])
    # 
    if _signals['blink_length'] == 0:
      duration = torch.tensor([_signals['blink_length']], dtype=torch.float32)
    else:
      duration = torch.tensor([_signals['blink_length']], dtype=torch.float32).log_()/3.1355
    return features, label, duration # 3.1355 is the max of log(duration)



class ToTensor(object):
  
  def __call__(self, sample):
    sample['std_r'] = torch.tensor(sample['std_r'])
    sample['std_g'] = torch.tensor(sample['std_g'])   
    sample['std_b'] = torch.tensor(sample['std_b'])
    sample['eyelids_dist'] = torch.tensor(sample['eyelids_dist'])

    return sample



class Normalize(object):
  """Normalized 
  """

  def __init__(self, feature_range=(0,1)):
    self._min = feature_range[0]
    self._max = feature_range[1]
    
  
  def __call__(self, sample):

    min_std_r=0.0
    max_std_r= 70.57724795565552
    sample['std_r'] = (sample['std_r'] - min_std_r) / (max_std_r-min_std_r)
    sample['std_r'] = sample['std_r'] * (self._max - self._min) - self._min

    min_std_g= 0.0
    max_std_g =69.65935905429274
    sample['std_g'] = (sample['std_g'] - min_std_g) / (max_std_g-min_std_g)
    sample['std_g'] = sample['std_g'] * (self._max - self._min) - self._min

    min_std_b= 0.0
    max_std_b =72.89523973711425
    sample['std_b'] = (sample['std_b'] - min_std_b) / (max_std_b-min_std_b)
    sample['std_b'] = sample['std_b'] * (self._max - self._min) - self._min

    min_eyelids= 0
    max_eyelids= 10.102511040619552
    sample['eyelids_dist'] = (sample['eyelids_dist'] - min_eyelids) / (max_eyelids-min_eyelids)
    sample['eyelids_dist'] = sample['eyelids_dist'] * (self._max - self._min) - self._min

    return sample