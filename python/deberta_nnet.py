import torch
from torch import nn, Tensor


from torch.autograd.function import FunctionCtx

from .intf_py_communicator import PyCommunicator

from .deberta_config import DebertaConfig

import math

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/deberta_v2/modeling_deberta_v2.py

# copypast
# https://github.com/huggingface/transformers/blob/617b21273a349bd3a94e2b3bfb83f8089f45749b/src/transformers/activations.py#L49


def gelu(input: torch.Tensor) -> torch.Tensor:
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class XSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, mask: Tensor, dim: int) -> Tensor:
        ctx.dim = dim  # type: ignore
        rmask = ~(mask.to(torch.bool))
        output = input.masked_fill(
            rmask, torch.tensor(torch.finfo(input.dtype).min))
        output = torch.softmax(output, ctx.dim)  # type: ignore
        output.masked_fill_(rmask, 0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(  # type: ignore
            ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None, None]:
        (output,) = ctx.saved_tensors  # type: ignore
        inputGrad = torch._softmax_backward_data(  # type: ignore
            grad_output, output, ctx.dim, output.dtype)  # type: ignore
        return inputGrad, None, None


class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            relative_pos: Tensor,
            rel_embeddings: Tensor) -> Tensor:
        self_output = self.self(
            hidden_states,
            attention_mask,
            relative_pos,
            rel_embeddings)
        attention_output = self.output(self_output, hidden_states)

        return attention_output


class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # intermediate を通ったものと通ってないものを受け取ってブレンド
    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Tensor,
                relative_pos: Tensor,
                rel_embeddings: Tensor) -> Tensor:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            relative_pos,
            rel_embeddings)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

# conv_kernel_size という config 項目が deberta_xlarge以上で存在しているが、3060ちゃんではメモリが足りないので無視でいいだろう


class DebertaV2Encoder(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore

        self.layer = nn.ModuleList([DebertaV2Layer(config)
                                    for _ in range(config.num_hidden_layers)])
        # self.relative_attentionは当然true
        self.max_relative_positions = config.max_position_embeddings
        # 英文の場合256あれば単語の相対距離としては十分だといえるが、
        # AIの場合そうではないと思われるので、position_bucketsは使わない
        # self.position_buckets =
        pos_ebd_size = self.max_relative_positions * 2

        self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        # v1にはない
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        # convlayerは実装しない

    def get_rel_embedding(self) -> Tensor:
        # rel_embeddings は 2*len x hidden_size
        rel_embeddings = self.rel_embeddings.weight
        rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask: Tensor):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * \
                extended_attention_mask.squeeze(-2).unsqueeze(-1)

        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        # attention_maskを二次元化している。なんでだ？ 全体としては四次元化している
        # attention_mask = torch.tensor([[1, 0, 1],[0,1,1]])
        # tensor([[[[1, 0, 1],
        #      [0, 0, 0],
        #      [1, 0, 1]]],
        #    [[[0, 0, 0],
        #      [0, 1, 1],
        #      [0, 1, 1]]]])
        return attention_mask

    def get_rel_pos(self, hidden_states: Tensor) -> Tensor:
        q = hidden_states.size(-2)
        relative_pos = build_relative_position(
            q, q, device=hidden_states.device)
        # relative_pos は 1 x len x len
        return relative_pos

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # input_maskはconvでしか使わないのでいらない
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states)
        rel_embeddings = self.get_rel_embedding()
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states, attention_mask, relative_pos, rel_embeddings)

        return hidden_states


def build_relative_position(query_size: int, key_size: int, device: torch.device) -> Tensor:
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script  # type: ignore
def c2p_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, relative_pos: Tensor) -> Tensor:
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script  # type: ignore
def p2c_dynamic_expand(c2p_pos: Tensor, query_layer: Tensor, key_layer: Tensor) -> Tensor:
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])

# v2とかつけんでええのんか


class DisentangledSelfAttention(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                # 文字列の自動連結は良くない文法であります
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        # v1では in_proj q_bias v_bias となっていた。in_projに3倍量を持たせていたが、なんでkにbiasがないのかわからない。
        self.query_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(
            config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = True
        self.pos_att_type = ["p2c", "c2p"]

        self.max_relative_positions = config.max_position_embeddings
        self.pos_ebd_size = self.max_relative_positions

        self.pos_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.attention_probs_dropout_probs)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        # b x len x hidden_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        # b x len x num_heads x head_size
        x = x.view(new_x_shape)
        # b x num_heads x len x head_size
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        # b*num_heads x len x head_size  x は上の行の x
        # 3次元にするのはbmmで処理するため。bmmの方が4次元で計算するより速いようだ。

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Tensor,
                relative_pos: Tensor,
                rel_embeddings: Tensor):
        # layers は b*num_heads x len x head_size
        query_layer = self.transpose_for_scores(self.query_proj(hidden_states))
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))

        scale_factor = 3
        scale = torch.sqrt(torch.tensor(
            query_layer.size(-1), dtype=torch.float) * scale_factor)
        # attention_scores は b*num_heads x len x len
        attention_scores = torch.bmm(
            query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        rel_embeddings = self.pos_dropout(rel_embeddings)
        rel_att = self.disentangled_attention_bias(
            query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
        )
        attention_scores = attention_scores + rel_att
        # b x num_heads x len x len に戻す
        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        )
        # XSoftmax内でattention_maskが0のところを最小値まで落としてSoftmaxを行い、その後maskが0のところに0を入れ直す
        # なのでmaskされたところのattention_probsは0になっている
        # ここで調べているのはposition a,b にある単語同士の関連度である
        attention_probs = XSoftmax.apply(  # type: ignore
            attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        # b*num_heads x len x len にしてbmm
        # b*num_heads x len x head_size になる
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2),
                                 attention_probs.size(-1)), value_layer
        )
        # b x len x num_heads x head_sizeに戻す
        context_layer = (
            context_layer.view(-1, self.num_attention_heads,
                               context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        # b x len x head_sizeに戻す
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    def disentangled_attention_bias(self,
                                    query_layer: Tensor,
                                    key_layer: Tensor,
                                    relative_pos: Tensor,
                                    rel_embeddings: Tensor,
                                    scale_factor: int) -> Tensor:
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        elif relative_pos.dim() != 4:
            raise ValueError(
                f"Relative position ids must be of dim d or 3 or 4. {relative_pos.dim()}")

        # relative_pos は 1 x 1 x len x len

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[0:att_span * 2, :].unsqueeze(0)

        # rel_embeddings はもともと２次元。上でunsqueezeされて 1* (2*len) * hidden_sizeとなっている
        # transpose_for_scores で 1 x num_heads x 2*len x head_size になる。
        # query_layer.size(0) は b x num_heads なので
        # b x num_heads x 2*len x head_sizeになる
        pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings)
                                                    ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)

        pos_key_layer = self.transpose_for_scores(
            self.key_proj(rel_embeddings)).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)

        score: Tensor = 0  # type: ignore

        if "c2p" in self.pos_att_type:
            scale = torch.sqrt(
                torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor)
            # query は b*num_heads x len x head_size, pos_key は b*num_heads x 2*len x head_size
            # b*num_heads x len x 2*len ができる
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # gatherの処理は
            # out[i][j][k] = input[i][j][index[i][j][k]]
            # index は 1xlenxlen を b*num_heads, len, lenにexpand
            # c2p_att もb*num_heads, len, lenとなる
            c2p_att = torch.gather(
                c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand(
                    [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)])
            )
            score += c2p_att / scale.to(dtype=c2p_att.dtype)

        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1)))
            p2c_pos = torch.clamp(-relative_pos + att_span,
                                  0, att_span * 2 - 1)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand(
                [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
            )).transpose(-1, -2)
            score += p2c_att / scale.to(dtype=p2c_att.dtype)

        return score


class DebertaV2Embeddings(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        # 0 以外を pad_token_idにします？
        pad_token_id = 0
        self.embedding_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            self.embedding_size, padding_idx=pad_token_id)
        # どの設定ファイルを見てもposition_biased_inputはfalse・・・必要そうな気はするが無視する
        self.position_biased_input = False
        if config.absolute_position_embeddings:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size)
            self.register_buffer(
                "position_ids",
                torch.arange(
                    config.max_position_embeddings).expand((1, -1)),
                persistent=False
            )
        else:
            self.position_embeddings = None

        # config.type_vocab_size はどう見てもfalse。
        # embedding_size != hidden_sizeは意味不明なので無視
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_biased_inputがfalseなので position_ids も使ってない

    def forward(self, input_ids: Tensor, mask: Tensor) -> Tensor:
        inputs_embeds: Tensor = self.word_embeddings(input_ids)
        if self.position_embeddings is not None:
            inputs_embeds = inputs_embeds + \
                self.position_embeddings(self.position_ids)
        embeddings: Tensor = self.LayerNorm(inputs_embeds)

        mask = mask.unsqueeze(2)
        mask = mask.to(embeddings.dtype)
        embeddings = embeddings * mask
        embeddings = self.dropout(embeddings)
        return embeddings


def init_weights(module: nn.Module, config: DebertaConfig, s: set[nn.Module]):
    if module in s:
        return
    s.add(module)

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

    for m in module.modules():
        init_weights(m, config, s)


class DebertaV2Model(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore

        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = 0
        self.config = config

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # setしないだろう
    # def set_input_embeddings

    # prune処理はDebertaでは実装されていないそうだ　ならなんでdefしてる？
    # def _prune_heads

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

        embedding_output: torch.Tensor = self.embeddings(
            input_ids, attention_mask)

        encoder_outputs: torch.Tensor = self.encoder(
            embedding_output, attention_mask)

        return encoder_outputs


class ContextPooler(nn.Module):
    def __init__(self, config: DebertaConfig):
        super().__init__()  # type: ignore
        self.dense = nn.Linear(config.hidden_size,
                               config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states: Tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = gelu(pooled_output)
        return pooled_output


class DebertaV2ForSequenceClassification(nn.Module):
    def __init__(self, py: PyCommunicator, config: DebertaConfig, ):
        super().__init__()  # type: ignore

        self.num_labels = py.move_len()

        self.deberta = DebertaV2Model(config)
        self.pi_pooler = ContextPooler(config)
        self.win_rate_pooler = ContextPooler(config)

        self.pi_classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.pi_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.win_rate_classifier = nn.Linear(config.hidden_size, 1)
        self.win_rate_dropout = nn.Dropout(config.hidden_dropout_prob)

        init_weights(self, config, set())

    def forward(self, input_ids: torch.Tensor) -> tuple[Tensor, Tensor]:
        batch_size = input_ids.size(0)
        input_ids = input_ids.view(batch_size, -1)
        # -1 ～ 1 を 2 ～ 4 にする
        input_ids = (input_ids + 3)

        # 最初にCLS(1)を追加
        ones = torch.ones(batch_size, 1)
        input_ids = torch.cat((ones, input_ids), dim=1)
        input_ids = input_ids.long().contiguous().cuda()

        attention_mask = torch.ones(input_ids.size()).cuda()

        outputs = self.deberta(
            input_ids,
            attention_mask)

        encoder_layer = outputs

        pooled_output = self.pi_pooler(encoder_layer)
        pooled_output = self.pi_dropout(pooled_output)
        pi = self.pi_classifier(pooled_output)

        pooled_output = self.win_rate_pooler(encoder_layer)
        pooled_output = self.win_rate_dropout(pooled_output)
        win_rate = self.win_rate_classifier(pooled_output)

        return nn.functional.log_softmax(pi, dim=1), torch.tanh(win_rate)


if __name__ == "__main__":

    # 例としてBATCH_SIZE=2, LEN=4のテンソルを作成
    batch_size, length = 2, 4
    original_tensor = torch.randn(batch_size, length)
    print(f"Original tensor:\n{original_tensor}")

    # 先頭に追加する要素を作成（各バッチに1を追加）
    new_elements = torch.ones(batch_size, 1)
    print(f"New elements:\n{new_elements}")

    # 新しいテンソルを先頭に結合
    modified_tensor = torch.cat((new_elements, original_tensor), dim=1)
    print(f"Modified tensor:\n{modified_tensor}")
