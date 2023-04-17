from datasets import CVLDataset, IAMDataset, LeopardiDataset, NorhandDataset, LAMDataset, RimesDataset, HKRDataset
from datasets import KHATTDataset, CHSDataset, BanglaWritingDataset


def main():
    # BanglaWriting
    bangla_writing_path = r'/mnt/FoMo_AIISDH/datasets/BanglaWriting'
    dataset = BanglaWritingDataset(bangla_writing_path)
    print(len(dataset))
    print(dataset[0][1])
    return

    # CHS
    chs_path = r'/mnt/FoMo_AIISDH/datasets/CHS'
    dataset = CHSDataset(chs_path)
    print(len(dataset))
    print(dataset[0][1])

    #KHATT
    khatt_path = r'/mnt/FoMo_AIISDH/datasets/KHATT_Arabic'
    dataset = KHATTDataset(khatt_path)
    print(len(dataset))
    print(dataset[0][1])

    # HKR
    hkr_path = r'/mnt/FoMo_AIISDH/datasets/HKR/20200923_Dataset_Words_Public'
    dataset = HKRDataset(hkr_path)
    print(len(dataset))
    print(dataset[0][1])

    # Rimes
    rimes_path = r'/home/shared/datasets/Rimes'
    dataset = RimesDataset(rimes_path)
    print(len(dataset))
    print(dataset[0][1])

    # CVL
    cvl_path = r'/home/shared/datasets/cvl-database-1-1'
    dataset = CVLDataset(cvl_path)
    print(len(dataset))
    print(dataset[0][1])

    # IAM
    iam_path = r'/home/shared/datasets/IAM'
    dataset = IAMDataset(iam_path)
    print(len(dataset))
    print(dataset[0][1])

    # Leopardi
    leopardi_path = r'/home/shared/datasets/leopardi'
    dataset = LeopardiDataset(leopardi_path)
    print(len(dataset))
    print(dataset[0][1])

    # Norhand
    norhand_path = r'/home/shared/datasets/Norhand'
    dataset = NorhandDataset(norhand_path)
    print(len(dataset))
    print(dataset[0][1])

    # LAM
    lam_path = r'/home/shared/datasets/LAM'
    dataset = LAMDataset(lam_path)
    print(len(dataset))
    print(dataset[0][1])


if __name__ == '__main__':
    main()
