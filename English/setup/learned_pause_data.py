import os
import numpy as np
from dataset import Dataset
from utils.tools import pad_1D


class LearnedPauseDataset(Dataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        basename, speaker = self.basename[index],self.speaker[index]
        sample['boundary'] = np.load(os.path.join(self.preprocessed_path,'boundary',f'{speaker}-boundary-{basename}.npy'))
        if len(sample['boundary']) != len(sample['text']):
            raise ValueError('Boundary conditioning and phoneme sequence differ')
        return sample

    def reprocess(self, data, idxs):
        return super().reprocess(data,idxs) + (pad_1D([data[i]['boundary'] for i in idxs]),)
