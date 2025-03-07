# Generation 

The **Standard Handwritten Text Generation** (SHTG) module `hwd.datasets.shtg` provides access to several handwriting datasets, each with configurable parameters for dataset customization. Below are the datasets and their respective configurations:

## Available Datasets

The following datasets are accessible within the module:

- **IAM Handwriting Database**
  - `IAMLines`: Lines from the IAM dataset.
  - `IAMWords`: Words from the IAM dataset.
  - `IAMLinesFromWords`: Line-level images generated from word-level style images.
  - `IAMWordsFromLines`: Word-level images generated from line-level style images.

- **RIMES Handwriting Database**
  - `RimesLines`: Lines from the RIMES dataset.

- **CVL Handwriting Database**
  - `CVLLines`: Lines from the CVL dataset.
  - `CVLWords`: Words from the CVL dataset.
  - `CVLLinesFromWords`: Line-level images generated from word-level style images.

- **Karaoke Handwriting Database**
  - `KaraokeLines`: Lines from the Karaoke dataset.
  - `KaraokeWords`: Words from the Karaoke dataset.
  - `KaraokeLinesFromWords`: Line-level images generated from word-level style images.
  - `KaraokeWordsFromLines`: Word-level images generated from line-level style images.

### Special Dataset Variants

Some datasets contain the suffix `FromWords` or `FromLines`, indicating that the style images originate from a different dataset level. For example, `IAMLinesFromWords` generates line images while using style images from `IAMWords`.

Additionally, variable-width datasets are available:

- `IAMLinesVariable`
- `IAMLinesFromVariable`

These datasets provide configurable width parameters to control the size of the generated images.

## Dataset Parameters

Each dataset can be customized using the following parameters:

- `num_style_samples` *(int, default=1)*: Number of style images sampled for each target image.
- `load_gen_sample` *(bool, default=False)*: Whether to load the generated sample.

### Additional Parameters for Specific Datasets

#### Karaoke Datasets
- `flavor` *(str)*: Specifies the dataset type. Options: `handwritten`, `typewritten`.

#### Variable-Width Datasets
- `min_width` *(int)*: Minimum width of generated images.
- `max_width` *(int)*: Maximum width of generated images.

## Example Usage

The following code demonstrates how to use the `IAMWords` dataset to generate images and save them to a directory:

```python
from hwd.datasets.shtg import IAMWords
from torchvision import transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class SHTGWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([
            self._to_height_64,
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def _to_height_64(self, img):
        width, height = img.size
        aspect_ratio = width / height
        new_width = int(64 * aspect_ratio)
        return img.resize((new_width, 64), Image.LANCZOS)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['style_imgs'] = [self.transforms(img.convert('RGB')) for img in sample['style_imgs']]
        return sample

# Load dataset
dataset = IAMWords(num_style_samples=1, load_gen_sample=False)

# Wrap dataset with transformations
dataset = SHTGWrapper(dataset)

# Define output directory
output_dir = Path("output_samples")
output_dir.mkdir(parents=True, exist_ok=True)

# Save transcriptions
dataset.dataset.save_transcriptions(output_dir)

for i, sample in enumerate(dataset):
    gen_text = sample['gen_text']
    style_imgs = sample['style_imgs']
    dst_path = output_dir / sample['dst_path']

    # Generate the image here
    out_img = ...

    # Save the image
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(dst_path)
```

## Implementation Notes

- Users must implement the image generation logic and adapt the transformations in the `SHTGWrapper` class to fit the model requirements.
- The script will save generated images in the `output_samples` directory and corresponding transcriptions in `output_samples/transcriptions.json`.

