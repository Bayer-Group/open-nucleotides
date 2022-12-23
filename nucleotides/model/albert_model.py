import torch
from transformers import AlbertConfig, AlbertForSequenceClassification

from nucleotides.model.lightning_model import FunctionalModel


class AlbertModel(FunctionalModel):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        config = AlbertConfig(
            vocab_size=9,
            embedding_size=3,
            num_labels=hparams.n_targets,
            max_position_embeddings=512,
            num_attention_heads=32,
            num_hidden_layers=8,
            hidden_size=1024,
            intermediate_size=1024,
        )
        self.model = AlbertForSequenceClassification(config=config)
        print(self.model)

    def forward(self, x):
        indexed_tokens = x.argmax(1) + 5
        class_tokens = torch.ones(size=(len(indexed_tokens), 1)).long().cuda() * 2
        indexed_tokens = torch.cat((class_tokens, indexed_tokens), dim=1)
        out = self.model(indexed_tokens)
        return out.logits
