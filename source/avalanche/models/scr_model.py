import torch
from torch import nn
from avalanche.models import NCMClassifier


class SCRModel(nn.Module):
    """
    Supervised Contrastive Replay Model.
    It uses an NCM Classifier during evaluation and a projection network
    during training.
    The input is passed through a feature extractor and then normalized
    before being fed to the classifier.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        projection: nn.Module,
        num_classes: int | None = None,
    ):
        """
        :param feature_extractor: a pytorch module that given the input
            examples extracts the hidden features
        :param projection: a pytorch module that takes as input the output
            of the feature extractor
        :param num_classes: optional fixed NCM output size for benchmarks
            without class-timeline metadata.
        """
        super().__init__()
        self.ncm = NCMClassifier(normalize=False, num_classes=num_classes)
        self.feature_extractor = feature_extractor
        self.train_classifier = projection
        self.eval_classifier = self.ncm

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        if self.training:
            return self.train_classifier(x)
        else:
            return self.eval_classifier(x)
