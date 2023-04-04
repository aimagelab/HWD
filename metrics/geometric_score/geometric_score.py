class GeometricScore:
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def compute(self, batch_size=128, verbose=False):
        print()