from datasets import CVLDataset, IAMDataset, LeopardiDataset, NorhandDataset, LAMDataset, RimesDataset, SaintGallDataset
from datasets import KHATTDataset, CHSDataset, BanglaWritingDataset


def main():
    # BanglaWriting
    # bangla_writing_path = r'/mnt/FoMo_AIISDH/datasets/BanglaWriting'
    # dataset = BanglaWritingDataset(bangla_writing_path)
    # print(len(dataset))
    # print(dataset[0][1])
    # return

    def print_dataset(db):
        per_auth = len(db) / len(db.author_ids) if len(db.author_ids) > 0 else None
        print(db.__class__.__name__, len(db), len(db.author_ids), per_auth)

    # SaintGall
    saint_gall_path = r'/home/shared/datasets/SaintGall'
    dataset = SaintGallDataset(saint_gall_path)
    print_dataset(dataset)

    # CHS
    # chs_path = r'/mnt/FoMo_AIISDH/datasets/CHS'
    # dataset = CHSDataset(chs_path)
    # print(dataset.__class__.__name__, len(dataset), len(dataset.author_ids), len(dataset) / len(dataset.author_ids))

    #KHATT
    khatt_path = r'/home/shared/datasets/KHATT_Arabic'
    dataset = KHATTDataset(khatt_path)
    print_dataset(dataset)

    # BANGLA
    bangla_path = r'/home/shared/datasets/BanglaWriting'
    dataset = BanglaWritingDataset(bangla_path)
    print_dataset(dataset)

    # CHS
    chs_path = r'/home/shared/datasets/CHS'
    dataset = CHSDataset(chs_path)
    print_dataset(dataset)

    # HKR
    # hkr_path = r'/mnt/FoMo_AIISDH/datasets/HKR/20200923_Dataset_Words_Public'
    # dataset = HKRDataset(hkr_path)
    # print(dataset.__class__.__name__, len(dataset), len(dataset.author_ids), len(dataset) / len(dataset.author_ids))

    # Rimes
    rimes_path = r'/home/shared/datasets/Rimes'
    dataset = RimesDataset(rimes_path)
    print_dataset(dataset)

    # CVL
    cvl_path = r'/home/shared/datasets/cvl-database-1-1'
    dataset = CVLDataset(cvl_path)
    print_dataset(dataset)

    # IAM
    iam_path = r'/home/shared/datasets/IAM'
    dataset = IAMDataset(iam_path)
    print_dataset(dataset)

    # Leopardi
    leopardi_path = r'/home/shared/datasets/leopardi'
    dataset = LeopardiDataset(leopardi_path)
    print_dataset(dataset)

    # Norhand
    norhand_path = r'/home/shared/datasets/Norhand'
    dataset = NorhandDataset(norhand_path)
    print_dataset(dataset)

    # LAM
    lam_path = r'/home/shared/datasets/LAM'
    dataset = LAMDataset(lam_path)
    print_dataset(dataset)


if __name__ == '__main__':
    main()
