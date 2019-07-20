from torchvision.datasets import ImageFolder


class ImageTargetFolder(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is targets value of given example.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.targets is not None:
            target = self.targets[index]

        return sample, target
