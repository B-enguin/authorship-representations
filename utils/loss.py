import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(a, b):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return a @ b.T

class MultipleNegativesSymmetricRankingLoss(nn.Module):
    """
    Multiple Negatives Symmetric Ranking Loss
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesSymmetricRankingLoss.py#L13-L98
    """

    def __init__(self, scale = 20.0):
        super(MultipleNegativesSymmetricRankingLoss, self).__init__()
        self.scale = scale
        self.CE = nn.CrossEntropyLoss()

    def forward(self, a, c):
        scores = cosine_similarity(a, c) * self.scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )

        anchor_positive_scores = scores[:, 0 : len(c)]
        forward_loss = self.CE(scores, labels)
        backward_loss = self.CE(anchor_positive_scores.transpose(0, 1), labels)
        return (forward_loss + backward_loss) / 2
        