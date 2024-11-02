from .base_score import ProcessedDataset
from .fved_score import FVeDScore
import torch
from torch.utils.data import DataLoader


class TVeDScore(FVeDScore):
    def __init__(self, checkpoint_path='metrics/fved/vgg_16_pretrained.pth', device='cuda', reduction='mean',
                 layers=4, patch_width=32, patch_stride=16, patch_height=32):
        super().__init__(checkpoint_path, device, reduction, layers)
        self.patch_width = patch_width
        self.patch_stride = patch_stride
        self.patch_height = patch_height
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_height, self.patch_width), stride=self.patch_stride)
        self.reshape = lambda x: x.reshape(3, self.patch_height, self.patch_width, -1).permute(3, 0, 1, 2)

    def digest(self, dataset, batch_size=32, verbose=False):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        ids, labels, features = self.get_activations(loader, verbose)
        return ProcessedDataset(ids, labels, features)

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        imgs = [self.reshape(self.unfold(img)) for img in imgs]
        ids = sum([[b[1], ] * img.size(0) for img, b in zip(imgs, batch)], [])
        img_ids = torch.Tensor(sum([[i, ] * img.size(0) for i, img in enumerate(imgs)], [])).long()
        return torch.cat(imgs, dim=0), ids, img_ids

    @torch.no_grad()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        features = []
        labels = []
        ids = []
        for i, (images, authors, images_ids) in enumerate(loader):
            images = images.to(self.device)

            pred = self.model(images)
            pred = pred.squeeze(-2)

            if self.reduction == None:
                raise NotImplementedError
            elif self.reduction == 'mean':
                labels.append(authors)
                ids.append(i * loader.batch_size + images_ids)
                pred = pred.unsqueeze(0) if loader.batch_size == 1 else pred
                features.append(pred.cpu())

            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)
        if verbose:
            print(' OK')

        ids = torch.cat(ids).long()
        labels = sum(labels, [])
        features = torch.cat(features, dim=0)
        return ids, labels, features
