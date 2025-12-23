from torch.utils.data import Dataset


class DisambiguationCorpus(Dataset):
    """
    Dataset for disambiguation tasks, modified for metaphor detection.
    """
    def __init__(self, file_path: str):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence, label = line.strip().split('\t')
                self.data.append((sentence, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
