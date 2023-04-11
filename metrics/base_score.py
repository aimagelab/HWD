
class BaseScore:
    def __call__(self, dataset1, dataset2, **kwargs):
        data1 = self.digest(dataset1, **kwargs)
        data2 = self.digest(dataset2, **kwargs)
        return self.distance(data1, data2)

    def digest(self, dataset, **kwargs):
        return dataset

    def distance(self, data1, data2):
        raise NotImplementedError

