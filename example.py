from metrics import HWDScore, FIDScore
from datasets import FolderDataset
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt

author_1 = FolderDataset(r'C:\Users\Vit\Desktop\CVL\cvl-database-1-1\trainset\words\0001', extension='tif')
author_2 = FolderDataset(r'C:\Users\Vit\Desktop\CVL\cvl-database-1-1\trainset\words\0002', extension='tif')
author_1_alt = FolderDataset(r'C:\Users\Vit\Desktop\CVL\cvl-database-1-1\trainset\words\0001', extension='tif')
author_2_alt = FolderDataset(r'C:\Users\Vit\Desktop\CVL\cvl-database-1-1\trainset\words\0002', extension='tif')

hwd = HWDScore()
fid = FIDScore()

hwd_author_1 = hwd.digest(author_1)
hwd_author_2 = hwd.digest(author_2)

fid_author_1 = fid.digest(author_1)
fid_author_2 = fid.digest(author_2)

hwd_11, hwd_22, hwd_12, hwd_21 = [], [], [], []
fid_11, fid_22, fid_12, fid_21 = [], [], [], []

custom_range = np.linspace(0, 1, 11)
for i in custom_range:
    print(i)
    t = transforms.ColorJitter(brightness=i, contrast=i, saturation=i, hue=i / 2)
    author_1_alt.preprocess = t
    author_2_alt.preprocess = t

    hwd_tmp_1 = hwd.digest(author_1_alt)
    hwd_tmp_2 = hwd.digest(author_2_alt)

    fid_tmp_1 = fid.digest(author_1_alt)
    fid_tmp_2 = fid.digest(author_2_alt)

    hwd_11.append(hwd.distance(hwd_author_1, hwd_tmp_1))
    hwd_22.append(hwd.distance(hwd_author_2, hwd_tmp_2))
    hwd_12.append(hwd.distance(hwd_author_1, hwd_tmp_2))
    hwd_21.append(hwd.distance(hwd_author_2, hwd_tmp_1))

    fid_11.append(fid.distance(fid_author_1, fid_tmp_1))
    fid_22.append(fid.distance(fid_author_2, fid_tmp_2))
    fid_12.append(fid.distance(fid_author_1, fid_tmp_2))
    fid_21.append(fid.distance(fid_author_2, fid_tmp_1))



# make the plot
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(custom_range, hwd_11)
axs[0, 1].plot(custom_range, hwd_22)
axs[1, 0].plot(custom_range, hwd_12)
axs[1, 1].plot(custom_range, hwd_21)

axs[0, 0].plot(custom_range, fid_11)
axs[0, 1].plot(custom_range, fid_22)
axs[1, 0].plot(custom_range, fid_12)
axs[1, 1].plot(custom_range, fid_21)

axs[0, 0].set_title('same author 1')
axs[0, 1].set_title('same author 2')
axs[1, 0].set_title('12')
axs[1, 1].set_title('21')

import json
with open('data.json', 'w') as f:
    data = {
        'hwd_11': hwd_11,
        'hwd_22': hwd_22,
        'hwd_12': hwd_12,
        'hwd_21': hwd_21,
        'fid_11': fid_11,
        'fid_22': fid_22,
        'fid_12': fid_12,
        'fid_21': fid_21,
    }
    json.dump(data, f)

# save the figure
fig.savefig('same_author.png', dpi=300)
plt.close(fig)
