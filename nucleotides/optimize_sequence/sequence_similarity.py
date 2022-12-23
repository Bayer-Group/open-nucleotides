import torch


class SimpleDistance:
    """
    Simple fraction of bases that is the same in two sequences.
    Does not take into account deletions and insertions.
    """

    def __init__(self, reference_sequence: torch.LongTensor):
        self.reference_sequence = reference_sequence

    def __call__(self, batch: torch.LongTensor):
        distance = (batch != self.reference_sequence).float().mean(dim=1)
        print("Sequence Distance:", distance.mean())
        return distance


class FlatBottomPotential:
    def __init__(self, flat=0.05, factor=20, exponent=2):
        self.flat = flat
        self.a = factor
        self.exponent = exponent

    def __call__(self, distance):
        return self.a * torch.clamp((distance - self.flat), min=0) ** self.exponent


class DistanceLoss:
    def __init__(self, reference_sequence=None, flat=0.01, a=20, exponent=2, use=True):
        self.potential = FlatBottomPotential(flat - flat, factor=a, exponent=exponent)
        self.distance_function = SimpleDistance(reference_sequence=reference_sequence)
        self.use = use

    def __call__(self, batch):
        if not self.use:
            return 0
        return self.potential(self.distance_function(batch))
