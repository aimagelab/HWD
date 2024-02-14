import torch
from torchvision import models
from torchmetrics.image.fid import NoTrainInceptionV3
from .fid.inception import InceptionV3
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from .base_score import BaseScore, ProcessedDataset, BaseBackbone


class AdjustDims(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.reshape(B, C)
        return x.reshape(B, C, -1)


class VGG16Backbone(BaseBackbone):
    def __init__(self, url, batch_size=1):
        super().__init__()
        self.url = url
        self.batch_size = batch_size
        self.model = self.load_model()
    
    def load_model(self):
        model = models.vgg16(num_classes=10400)

        if self.url is not None:
            checkpoint = torch.hub.load_state_dict_from_url(self.url, progress=True, map_location='cpu')
            model.load_state_dict(checkpoint)

        modules = list(model.features.children())
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(AdjustDims())
        return torch.nn.Sequential(*modules)
    
    @torch.inference_mode()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        features, labels, ids = [], [], []
        for i, (images, authors, _) in enumerate(loader):
            images = images.to(next(self.model.parameters()).device)

            pred = self.model(images)
            pred = pred.squeeze(-2)

            pred = pred.unsqueeze(0) if pred.ndim == 1 else pred
            labels.append(authors)
            ids.append([i * loader.batch_size + d for d in range(len(authors))])
            features.append(pred.cpu())

            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)
        if verbose:
            print(' OK')

        ids = torch.Tensor(sum(ids, [])).long()
        labels = sum(labels, [])
        features = torch.cat(features, dim=0)
        return ids, labels, features
    
    def collate_fn(self, batch):
        imgs_width = [x[0].size(2) for x in batch]
        max_width = max(imgs_width)
        imgs = torch.stack([torch.nn.functional.pad(x[0], (0, max_width - x[0].size(2))) for x in batch], dim=0)
        ids = [x[1] for x in batch]
        return imgs, ids, torch.Tensor(imgs_width)
    
    def __call__(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        ids, labels, features = self.get_activations(loader, verbose)
        return ProcessedDataset(ids, labels, features)


class InceptionV3Backbone(BaseBackbone):
    def __init__(self, url, batch_size=128):
        super().__init__()
        self.url = url
        self.batch_size = batch_size
        self.model = InceptionV3(url=self.url)
    
    @torch.inference_mode()
    def get_activations(self, loader, verbose=False):
        self.model.eval()

        features = []
        labels = []
        for i, (images, authors) in enumerate(loader):
            images = images.to(next(self.model.parameters()).device)

            pred = self.model(images)[0]

            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            features.append(pred.squeeze(-1, -2).cpu())
            labels.append(list(authors))
            if verbose:
                print(f'\rComputing activations {i + 1}/{len(loader)}', end='', flush=True)

        if verbose:
            print(' OK')

        features = torch.cat(features, dim=0)
        labels = sum(labels, [])
        ids = torch.arange(len(labels))
        return ids, labels, features
    
    def __call__(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        ids, labels, features = self.get_activations(loader, verbose)
        return ProcessedDataset(ids, labels, features)

    