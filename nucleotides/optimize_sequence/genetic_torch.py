import torch

from nucleotides.optimize_sequence.sequence_similarity import DistanceLoss
from nucleotides.util.util import sequence_to_ints


class GeneticAlgorithm:
    """Genetic Algorithm implemented using Pytorch"""

    def __init__(
            self,
            predictor,
            target,
            mask=None,
            sequence_length=None,
            population_size=1000,
            mutation_rate=0.005,
            n_tournaments=3,
            device="cpu",
            auto_balance_loss=True,
            keep_best_predictions=True,
            starting_population=None,
            penalize_distance=True,
    ):
        self.model = predictor.to(device)
        self.device = device

        if not isinstance(starting_population, torch.LongTensor):
            starting_population = torch.LongTensor(
                [
                    sequence_to_ints(sequence)
                    if isinstance(sequence, str)
                    else sequence
                    for sequence in starting_population
                ]
            )

        if sequence_length and starting_population is not None:
            if sequence_length != starting_population.shape[-1]:
                raise ValueError(
                    f"Both sequence length and starting population defined with different sequence lengths. ({sequence_length} and {starting_population.shape[-1]}) "
                )
            self.sequence_length = sequence_length
        elif sequence_length:
            self.sequence_length = sequence_length
        elif starting_population is not None:
            self.sequence_length = starting_population.shape[-1]
        else:
            self.sequence_length = 1000

        if mask is None:
            mask = torch.ones(len(target))
        self.mask = mask.bool().to(device)

        n_positive = target[self.mask].sum()
        n_negative = self.mask.sum() - n_positive

        if auto_balance_loss and n_negative and n_positive:
            pos_weight = n_negative / n_positive
        else:
            pos_weight = None

        self.loss_function = torch.nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=pos_weight
        )

        self.starting_population = starting_population
        self.population = self._create_population(sequence_length, population_size)
        self._eye = torch.eye(4, device=self.device)
        self.target = target.unsqueeze(0).expand(population_size, -1).to(device)

        self.mutation_rate = mutation_rate
        self.n_tournaments = n_tournaments

        self.keep_best_predictions = keep_best_predictions
        self.best_predictions = []
        self.loss_curve = []

        if not penalize_distance:
            self.distance_loss = DistanceLoss(use=False)
        else:
            # REMARK: what if starting population is a real population and not just one sequence?
            self.distance_loss = DistanceLoss(
                reference_sequence=self.starting_population[0].to(device)
            )

    def _create_population(self, sequence_length, population_size):
        if self.starting_population is not None:
            n_starting_population = self.starting_population.shape[0]
            factor = population_size // n_starting_population
            remainder = population_size % n_starting_population
            return torch.cat(
                [
                    self.starting_population.repeat(factor, 1),
                    self.starting_population[:remainder],
                ]
            ).to(self.device)
        return torch.randint(0, 4, (population_size, sequence_length)).to(
            self.device
        )

    def _onehot(self, batch):
        return self._eye[batch].transpose(1, 2)

    def _score(self, batch):
        logits = self.model(self._onehot(batch))
        logits_selected = logits[:, self.mask]
        target_selected = self.target[:, self.mask]
        losses = self.loss_function(logits_selected, target_selected).mean(
            dim=1
        ) + self.distance_loss(batch)
        best_index = torch.argmin(losses)
        best_predictions = torch.nn.functional.sigmoid(logits[best_index])
        best_predictions_selected = torch.nn.functional.sigmoid(
            logits_selected[best_index]
        )
        if self.keep_best_predictions:
            self.best_predictions.append(best_predictions.cpu().detach().numpy())
        self.loss_curve.append(losses.mean().cpu().detach())
        print(
            "mean prediction positive class:",
            best_predictions_selected[target_selected[0] == 1].mean().item(),
        )
        print(
            "mean prediction negative class:",
            best_predictions_selected[target_selected[0] == 0].mean().item(),
        )
        print("loss:", losses.mean().item())
        return losses

    def _select(self, batch):
        loss = self._score(batch)
        losses = []
        batches = []
        for _ in range(self.n_tournaments):
            perm = torch.randperm(len(batch), device=self.device)
            losses.append(loss[perm])
            batches.append(batch[perm])
        losses = torch.stack(losses, dim=-1)
        batches = torch.stack(batches, dim=-1)
        selector = (
            losses.argmin(dim=-1)
            .unsqueeze(dim=1)
            .unsqueeze(dim=1)
            .repeat(1, self.sequence_length, 1)
        )
        result = torch.gather(batches, -1, selector).squeeze()
        return result

    def _mate(self, batch):
        reverse_batch = torch.flip(batch, [0])
        print(batch.shape)
        region, _ = torch.sort(
            torch.randint(
                low=1,
                high=self.sequence_length - 1,
                size=(len(batch), 2),
                device=self.device,
            )
        )
        low = region[:, 0]
        high = region[:, 1]
        template = (
            torch.arange(0, self.sequence_length).unsqueeze(0).repeat(len(batch), 1)
        ).to(self.device)
        mask = (template >= low.unsqueeze(-1)) & (template < high.unsqueeze(-1))
        batch = batch * (~mask) + reverse_batch * mask
        return batch

    def _mutate(self, batch):
        flips = torch.randint_like(batch, low=0, high=3) * torch.bernoulli(
            torch.ones_like(batch) * self.mutation_rate
        )
        batch = ((batch + flips) % 4).long()
        return batch

    def _sort_by_loss(self, batch):
        losses = self._score(batch)
        print(losses.shape)
        _, indices = torch.sort(losses, descending=False)
        print(_)
        return batch[indices]

    def fit(self, n_generations=20):
        population = self.population
        for i in range(n_generations):
            print(i)
            population = self._select(population)
            population = self._mate(population)
            population = self._mutate(population)
        population = self._sort_by_loss(population)
        self.population = population
