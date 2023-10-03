import torch
from pytorchltr.datasets import MSLR10K

class MSLR(MSLR10K):
    def __init__(self, split, fold = 1, **kwargs):
        if split == 'val':
            split = 'vali'
        assert split in ['train', 'vali', 'test']
        super().__init__(split=split, fold=fold)
        
    @staticmethod
    def collate_fn(list_sampler = None, **kwargs):
        def _collate_fn(batch):
            org_batch = super(MSLR, MSLR).collate_fn(list_sampler)(batch)
            xs, ys, n = org_batch.features, org_batch.relevance, org_batch.n
            data = dict(
                data = xs[:, :, :-5], 
                labels_qs = xs[:, :, -5], 
                labels_qs2 = xs[:, :, -4], 
                labels_qucc = xs[:, :, -3],
                labels_ucc = xs[:, :, -2],
                labels_udt = xs[:, :, -1],
                labels_rel = ys, 
                n = n)
            return data
        return _collate_fn
        
    def task_names(self):
        return ['qs', 'qs2', 'qucc', 'ucc', 'udt', 'rel']
        
if __name__ == "__main__":
    dst = MSLR(split='train')
    dst_orig = MSLR10K(split='train')
    loader = torch.utils.data.DataLoader(dst, batch_size=64, shuffle=False, num_workers=4, collate_fn=dst.collate_fn())
    loader_orig = torch.utils.data.DataLoader(dst_orig, batch_size=64, shuffle=False, num_workers=4, collate_fn=dst_orig.collate_fn())
    
    tmp = next(iter(loader))
    tmp_orig = next(iter(loader_orig))
    
    print('Data')
    print(tmp['data'].shape, tmp['data'][0,:,0])
    print(tmp_orig.features.shape, tmp_orig.features[0,:,0])
    
    print('Labels: QualityScore')
    print(tmp['labels_qs'].shape, tmp['labels_qs'][0])
    print(tmp_orig.features[0,:,-5])

    print('Labels:Relevance')    
    print(tmp['labels_rel'].shape, tmp['labels_rel'][0])
    print(tmp_orig.relevance.shape, tmp_orig.relevance[0])
    
    print('n')
    print(tmp['n'].shape, tmp['n'])
    print(tmp_orig.n.shape, tmp_orig.n)