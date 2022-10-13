import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from pyrl.utils.meta import Registry, build_from_cfg

ATTENTION_LAYERS = Registry("attention layer")


def compute_attention(score, v, dropout=None, mask=None):
    """
    :param score: [B, NH, NQ, NK]
    :param v: Value [B, NH, NK, E]
    :param mask: [B, NQ, NK]
    :param dropout:
    :return: [B, NH, NQ, E]
    """
    if mask is not None:
        mask = mask[:, None]
        score = score * mask + (-1e8) * (1 - mask)
    score = F.softmax(score, dim=-1)  # [B, NH, NQ, NK]
    if dropout is not None:
        score = dropout(score)
    return torch.einsum("bnij,bnjk->bnik", score, v)  # [B, NH, NQ, E]


class MultiHeadedAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        """
        :param embed_dim: The dimension of feature in each entity.
        :param num_heads: The number of attention heads.
        :param latent_dim:
        :param dropout:
        """
        super().__init__()
        self.sqrt_latent_dim = np.sqrt(latent_dim)
        self.w_k = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_o = nn.Parameter(torch.empty(num_heads, latent_dim, embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        if hasattr(self, "q"): 
            nn.init.xavier_normal_(self.q)
        if hasattr(self, "w_q"): 
            nn.init.xavier_normal_(self.w_q)
        if hasattr(self, "w_kr"):  
            nn.init.xavier_normal_(self.w_kr)

    def get_atten_score(self, x, *args, **kwargs):
        raise NotImplementedError


@ATTENTION_LAYERS.register_module()
class AttentionPooling(MultiHeadedAttentionBase):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.q = nn.Parameter(torch.empty(num_heads, 1, latent_dim))
        self._reset_parameters()

    def get_atten_score(self, x):
        k = torch.einsum("blj,njd->bnld", x, self.w_k)  # [B, NH, N, EL]
        score = torch.einsum("nij,bnkj->bnik", self.q, k) / self.sqrt_latent_dim  # [B, NH, 1, NK]
        return score

    def forward(self, x, mask=None, *args, **kwargs):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, 1, N] [batch size, 1, length]
        :return: [B, E] [batch_size, embed_dim] one feature with size
        """
        # print(x.shape, self.w_v.shape)
        # exit(0)
        v = torch.einsum("blj,njd->bnld", x, self.w_v)  # [B, NH, N, EL]
        score = self.get_atten_score(x)
        out = compute_attention(score, v, self.dropout, mask)
        out = torch.einsum("bnlj,njk->blk", out, self.w_o)  # [B, 1, E]
        out = out[:, 0]
        return out


@ATTENTION_LAYERS.register_module()
class MultiHeadAttention(MultiHeadedAttentionBase):
    """
    Attention is all you need:
        https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.w_q = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self._reset_parameters()

    def get_atten_score(self, key, query):
        q = torch.einsum("blj,njd->bnld", query, self.w_q)  # [B, NH, NQ, EL]
        k = torch.einsum("blj,njd->bnld", key, self.w_k)  # [B, NH, NK, EL]
        score = torch.einsum("bnij,bnkj->bnik", q, k) / self.sqrt_latent_dim  # [B, NH, NQ, NK]
        return score

    def forward(self, key, query, mask=None, *args, **kwargs):
        """
        :param key: [B, NK, E] [batch size, length of keys, embed_dim] the input to the layer, a tensor of shape
        :param query: [B, NQ, E] [batch size, length of queries, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, NQ, NK] [batch size, length of keys, length of queries]
        :return: [B, NQ, E] [batch_size, length, embed_dim] Features after self attention
        """
        score = self.get_atten_score(key, query)
        v = torch.einsum("blj,njd->bnld", key, self.w_v)  # [B, NH, NK, EL]
        out = compute_attention(score, v, self.dropout, mask)  # [B, NH, NQ, E]
        out = torch.einsum("bnlj,njk->blk", out, self.w_o)  # [B, NQ, E]
        out = self.dropout(out)
        return out


@ATTENTION_LAYERS.register_module()
class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super(MultiHeadSelfAttention, self).__init__(embed_dim, num_heads, latent_dim, dropout)

    def forward(self, x, mask=None, *args, **kwargs):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length]
        :return: [B, N, E] [batch_size, length, embed_dim] Features after self attention
        """
        return super(MultiHeadSelfAttention, self).forward(x, x, mask, *args, **kwargs)

def build_attention_layer(cfg, default_args=None):
    return build_from_cfg(cfg, ATTENTION_LAYERS, default_args)
