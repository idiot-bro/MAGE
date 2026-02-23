import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_path, device='cpu', cuda_id = None):
        self.file_path = file_path
        self.device = device
        self.cuda_id = cuda_id
        self.data = self.load_data(file_path)

    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def load_data(self, file_path):
        print('Loading data...{}'.format(file_path))
        if not file_path.lower().endswith('.pt'):
            raise FileNotFoundError
        if self.device == 'cpu':
            data = torch.load(file_path, map_location=self.device)
        else:
            data = torch.load(file_path, map_location=torch.device(f"cuda:{self.cuda_id}"))
        return data.float()

def CustomDataloader(dataset, batch_size = 30, shuffle = True, pin_memory = True, collate_fn = None, num_workers = 0):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                    pin_memory=pin_memory, collate_fn=collate_fn, num_workers=num_workers)


if __name__ == '__main__':
    pass


