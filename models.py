import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


class BilingualSentenceClassifier(nn.Module):
    """Model that classifies MT vs HT based on two-sentence inputs: source and
    translation.

    NOTE: this model was not used as part of the publication, but is part of ongoing
    future work.

    The idea is that the mapping from source to translation provides useful
    information to distinguish MT from HT. For example, the way the word ordering
    changes as a result of translation.

    We use a multilingual Transformer, since providing both source and translation
    means that each data instance consists of a bilingual sentence pair, i.e. German
    and English. After passing the input sentence pair through the model, we take the
    mean embedding for source and translation, and pass them through a MLP.

    Currently only works with Roberta-XML.
    """

    def __init__(self, hf_model, emb_size, dropout=0.1):
        super().__init__()
        self.hf_model = hf_model
        self.drop = nn.Dropout(p=dropout)
        self.dense = nn.Linear(2 * emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, 2)

        self.num_labels = 2  # binary classification
        # NOTE: the token ids are only tested for Roberta-XML.
        self.pad_token_id = 1
        self.eos_token_id = 2

    def forward(self, input_ids, attention_mask, labels=None, *args, **kwargs):
        outputs = self.hf_model(input_ids, attention_mask=attention_mask)
        embs = outputs.last_hidden_state  # (bsz, seq_len, hidden_size)

        # Group the embeddings by sentence and take the mean.
        first_eos_idx = (input_ids == self.eos_token_id).float().argmax(-1)
        last_eos_idx = (input_ids == self.pad_token_id).float().argmax(-1).sub(1)
        sent1_emb = torch.empty([embs.shape[0], embs.shape[2]], device=embs.device)
        sent2_emb = torch.empty([embs.shape[0], embs.shape[2]], device=embs.device)
        # For Roberta(-XLM), the <SEP> token is equal to </s></s>, where <EOS> is
        # </s>. So the sentences are separated by two tokens.
        for i, (idx1, idx2) in enumerate(zip(first_eos_idx, last_eos_idx)):
            sent1_emb[i] = embs[i, 1:idx1].mean(0)
            sent2_emb[i] = embs[i, idx1 + 2 : idx2].mean(0)

        out = torch.cat([sent1_emb, sent2_emb], -1)
        out = self.drop(out)
        out = self.dense(out)
        out = torch.tanh(out)
        out = self.drop(out)
        logits = self.out_proj(out)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
