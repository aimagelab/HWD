from datasets import CVLDataset, IAMDataset, LeopardiDataset, NorhandDataset, LAMDataset, RimesDataset


def main():

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
