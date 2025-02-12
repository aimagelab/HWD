import torch
import warnings
from tqdm import tqdm
from torchvision import models
from torchmetrics.image.fid import NoTrainInceptionV3
from .fid.inception import InceptionV3
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from .base_score import BaseScore, ProcessedDataset, BaseBackbone
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from . import gs

class LayerActivations:
    def __init__(self, model):
        self.activations = {}
        self.handles = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        def hook_fn(module, input, output, name):
            if name not in self.activations:
                self.activations[name] = []
            self.activations[name].append(output)

        for name, layer in model.named_modules():
            handle = layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name))
            self.handles.append(handle)

    def clear(self):
        self.activations = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        return False


class AdjustDims(torch.nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.swapaxes(1, 3)
        return x.reshape(B, H * W, C)


class VGG16Backbone(BaseBackbone):
    def __init__(self, url, num_classes=1000, batch_size=1):
        super().__init__()
        self.url = url
        self.batch_size = batch_size
        if self.batch_size > 1:
            warnings.warn('With batch_size > 1 you can achieve faster inference but with a numerical differnece.')
        self.model = self.load_model(num_classes)
    
    def load_model(self, num_classes):
        model = models.vgg16(num_classes=num_classes)

        if self.url is not None:
            checkpoint = torch.hub.load_state_dict_from_url(self.url, progress=True, map_location='cpu')
            model.load_state_dict(checkpoint)

        modules = list(model.features.children())
        # modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(AdjustDims())
        return torch.nn.Sequential(*modules)
    
    @torch.inference_mode()
    def get_activations(self, loader, verbose=False, stream=False):
        self.model.eval()
        device = next(self.model.parameters()).device
        features, authors_list, ids = [], [], []
        loader_bar = tqdm(enumerate(loader), desc='Computing activations', disable=not verbose, total=len(loader))
        for i, (images, authors, imgs_width) in loader_bar:
            images = images.to(device)

            # Create the mask to remove the padding tokens
            imgs_features = (imgs_width / 32).floor().int() * int(images.size(2) / 32)
            features_mask = torch.arange(imgs_features.max())
            features_mask = features_mask.unsqueeze(0).repeat(images.size(0), 1)
            features_mask = features_mask < imgs_features.unsqueeze(1)

            preds = self.model(images)
            preds = preds[features_mask]

            local_ids = [i * loader.batch_size + d for d in range(len(authors))]
            ids.extend(a for a, n in zip(local_ids, imgs_features.tolist()) for _ in range(n))
            authors_list.extend(a for a, n in zip(authors, imgs_features.tolist()) for _ in range(n))
            features.append(preds.cpu())

            if stream:
                ids = torch.Tensor(ids).long()
                features = torch.cat(features, dim=0)
                yield ids, authors_list, features
                features, authors_list, ids = [], [], []
        if stream:
            return
        ids = torch.Tensor(ids).long()
        features = torch.cat(features, dim=0)
        yield ids, authors_list, features
    
    def stream(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=self.collate_fn)
        for ids, authors, features in self.get_activations(loader, verbose, stream=True):
            yield ProcessedDataset(ids, authors, features)
    
    def collate_fn(self, batch):
        imgs, authors, _ = zip(*batch)
        imgs_width = [img.size(2) for img in imgs]
        max_width = max(imgs_width)
        imgs = torch.stack([torch.nn.functional.pad(img, (0, max_width - img.size(2))) for img in imgs], dim=0)
        return imgs, authors, torch.Tensor(imgs_width)
    
    def __call__(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=self.collate_fn)
        ids, authors, features = next(self.get_activations(loader, verbose))
        return ProcessedDataset(ids, authors, features)
    

class ActivationsVGG16Backbone(VGG16Backbone):
    @torch.inference_mode()
    def get_activations(self, loader, verbose=False, stream=False):
        assert stream, 'The activations can only be computed with stream==True'
        self.model.eval()
        device = next(self.model.parameters()).device
        loader_bar = tqdm(enumerate(loader), desc='Computing activations', disable=not verbose, total=len(loader))
        for i, (images, authors, imgs_width) in loader_bar:
            images = images.to(device)

            with LayerActivations(self.model) as activations:
                self.model(images)
                style_features = activations.activations
            
            layers_keys = ['0', '5', '10', '19', '28']
            preds = [style_features[k][0] for k in layers_keys]
            preds_T = list(zip(*[p.split(1) for p in preds]))
            
            ids = torch.Tensor([i * loader.batch_size + d for d in range(len(authors))]).long()
            yield ids, authors, preds_T


class InceptionV3Backbone(BaseBackbone):
    def __init__(self, url, batch_size=128):
        super().__init__()
        self.url = url
        self.batch_size = batch_size
        self.model = InceptionV3(url=self.url)
    
    @torch.inference_mode()
    def get_activations(self, loader, verbose=False, stream=False):
        self.model.eval()
        device = next(self.model.parameters()).device
        features = []
        authors_list = []
        for images, authors, _ in tqdm(loader, desc='Computing activations', disable=not verbose):
            images = images.to(device)

            pred = self.model(images)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            features.append(pred.squeeze(-1, -2).cpu())
            authors_list.extend(authors)

            if stream:
                features = torch.cat(features, dim=0)
                ids = torch.arange(len(authors_list))
                yield ids, authors_list, features
                features, authors_list, ids = [], [], []

        if stream:
            return

        features = torch.cat(features, dim=0)
        ids = torch.arange(len(authors_list))
        yield ids, authors_list, features
    
    def collate_fn(self, batch):
        imgs, authors, labels = zip(*batch)
        imgs = torch.stack(imgs)
        return imgs, authors, labels
    
    def stream(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=self.collate_fn)
        for ids, authors, features in self.get_activations(loader, verbose, stream=True):
            yield ProcessedDataset(ids, authors, features)
    
    def __call__(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=self.collate_fn)
        ids, authors, features = next(self.get_activations(loader, verbose))
        return ProcessedDataset(ids, authors, features)


class TrOCRBackbone(BaseBackbone):
    def __init__(self, path='microsoft/trocr-base-handwritten', max_new_tokens=64, batch_size=64):
        super().__init__()
        self.processor = TrOCRProcessor.from_pretrained(path)
        self.model = VisionEncoderDecoderModel.from_pretrained(path)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
    
    @torch.inference_mode()
    def get_text(self, loader, verbose=False):
        self.model.eval()
        device = next(self.model.parameters()).device
        features = []
        authors_list = []
        labels_list = []
        preds_list = []
        for images, authors, labels in tqdm(loader, desc='Computing text', disable=not verbose):
            images = self.processor(images=images, return_tensors="pt").pixel_values.to(device)

            ids = self.model.generate(images, max_new_tokens=self.max_new_tokens)
            preds = self.processor.batch_decode(ids, skip_special_tokens=True)

            authors_list.extend(authors)
            labels_list.extend(labels)
            preds_list.extend(preds)

        return preds_list, labels_list, authors_list
    
    def collate_fn(self, batch):
        imgs, authors, labels = zip(*batch)
        # max_width = max([img.size(2) for img in imgs])
        # imgs = torch.stack([torch.nn.functional.pad(img, (0, max_width - img.size(2))) for img in imgs], dim=0)
        return imgs, authors, labels
    
    def __call__(self, dataset, verbose=False):
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=1, shuffle=False, collate_fn=self.collate_fn)
        preds, labels, authors = self.get_text(loader, verbose)
        return preds, labels, authors


class GeometryBackbone(BaseBackbone):
    def __init__(self, max_workers=8, L_0=64, gamma=None, i_max=100, n=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rlts_kwargs = {
            'max_workers': max_workers,
            'L_0': L_0,
            'gamma': gamma,
            'i_max': i_max,
            'n': n
        }
    
    def __call__(self, dataset, verbose=False):
        data = torch.stack([sample[0] for sample in dataset])
        data = data.flatten(1)
        return gs.rlts_parallel(data.cpu().numpy(), verbose=verbose, **self.rlts_kwargs)