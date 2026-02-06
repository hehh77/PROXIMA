
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Optional, Tuple, Mapping
from torch.distributions import Bernoulli
import numpy as np
from tqdm import trange
import warnings

from Src.model.model import (
    GeneEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    BatchLabelEncoder,
    FlashscGPTLayer,
    FlashscGPTGenerator,
    FastTransformerEncoderWrapper,
    FlashTransformerEncoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    MoeDecoder,
    ExprDecoder,
    ClsDecoder,
    MVCDecoder,
    AdversarialDiscriminator
)


class TransformerModel_add_embedding(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.8,
        explicit_zero_prob: bool = False,
        use_generative_training=False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        use_MVC_impute: bool = False,
        impute_MVC_knn_k: Optional[int] = None,
        use_moe_dec: bool = False,
        esm_embedding_matrix: torch.Tensor = None,
        freeze_esm: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        self.use_MVC_impute = use_MVC_impute
        self.impute_MVC_knn_k = impute_MVC_knn_k
        self.use_moe_dec = use_moe_dec

        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])



        self.use_esm_features = esm_embedding_matrix is not None
        if self.use_esm_features:
            esm_embedding_dim = esm_embedding_matrix.shape[1]
            
            # 1. 创建专用的、冻结的ESM嵌入层
            self.esm_encoder = nn.Embedding.from_pretrained(
                esm_embedding_matrix, 
                freeze=freeze_esm
            )
            
            # 2. 创建一个专门用于ESM嵌入的LayerNorm层
            self.esm_norm = nn.LayerNorm(esm_embedding_dim)
            
            # 3. 创建融合层
            self.fusion_layer = nn.Linear(d_model + esm_embedding_dim, d_model)
            self.fusion_norm = nn.LayerNorm(d_model)



        self.flag_encoder = nn.Embedding(2, d_model)

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling
        # Batch Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        if use_generative_training:
            encoder_layers = FlashscGPTLayer(
                d_model,
                nhead,
                d_hid,
                dropout,
                batch_first=True,
                norm_scheme=self.norm_scheme,
            )
            self.transformer_encoder = FlashscGPTGenerator(encoder_layers, nlayers)
        elif use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        if self.use_moe_dec:
            self.decoder = MoeDecoder(
                d_model,
                num_experts=4,
                use_batch_labels=use_batch_labels,
            )
        else:
            self.decoder = ExprDecoder(
                d_model,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if n_cls > 1:
            self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
        
        if use_MVC_impute:
            self.impute_mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        self._check_batch_labels(batch_labels)

        src_emb = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src_emb

        if self.use_esm_features:
            # 2. 获取ESM嵌入，并对其进行LayerNorm
            esm_emb = self.esm_encoder(src)
            esm_emb = self.esm_norm(esm_emb) # <<< 关键步骤：对ESM嵌入进行层归一化
            
            # 3. 拼接两种已经归一化过的嵌入
            concatenated_emb = torch.cat([src_emb, esm_emb], dim=-1)
            
            # 4. 通过融合层进行投影和最终的归一化
            fused_emb = self.fusion_layer(concatenated_emb)
            fused_emb = self.fusion_norm(fused_emb)
        else:
            fused_emb = src_emb



        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src_emb * values
        else:
            total_embs = src_emb + values

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output

    def transformer_generate(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
        input_cell_emb: Optional[Tensor] = None,  # (batch, seq_len, embsize)
    ) -> Tuple[Tensor, Tensor]:
        self._check_batch_labels(batch_labels)

        pcpt_token_embs = self.encoder(pcpt_genes)  # (batch, pcpt_len, embsize)
        pcpt_values = self.value_encoder(pcpt_values)  # (batch, pcpt_len, embsize)
        pcpt_total_embs = pcpt_token_embs + pcpt_values

        assert self.input_emb_style != "scaling"
        if gen_genes is not None:
            gen_token_embs = self.encoder(gen_genes)  # (batch, gen_len, embsize)
            self.cur_gene_token_embs = torch.cat(
                [pcpt_token_embs, gen_token_embs], dim=1
            )
            gen_flags = self.flag_encoder(
                torch.tensor(1).to(pcpt_values.device)
            ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)

            gen_total_embs = gen_token_embs + gen_flags
        else:
            self.cur_gene_token_embs = pcpt_token_embs
            gen_total_embs = None

        if input_cell_emb is not None:
            pcpt_total_embs[:, 0, :] = input_cell_emb

        pcpt_output, gen_output = self.transformer_encoder(
            pcpt_total_embs,
            gen_total_embs,
            pcpt_key_padding_mask=pcpt_key_padding_mask,
            gen_key_padding_mask=gen_key_padding_mask,
        )

        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels` is True"
            )

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (batch, embsize)
            src(:obj:`Tensor`): shape (batch, seq_len)
            values(:obj:`Tensor`): shape (batch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (batch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            batch_labels(:obj:`Tensor`): shape (batch,), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration
        try:
            self._check_batch_labels(batch_labels)
        except:
            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.encoder(src)  # (batch, seq_len, embsize)

        if values is not None:
            values = self.value_encoder(values)  # (batch, seq_len, embsize)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        output = mlm_output["pred"]  # (batch, seq_len)

        return output  # (batch, seq_len)

    def _extend_output(
        self,
        output: Mapping[str, Tensor],
        transformer_output: Tensor,
        batch_emb: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        MVC_impute: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:

        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        
        if MVC_impute:
            coordinates = output['coordinates']
            K_NN = self.impute_MVC_knn_k
            dist = torch.cdist(coordinates, coordinates, p=2)  # No EPS added here
    
            # Select the top K nearest neighbors excluding self
            topk_index = torch.topk(dist, k=K_NN+1, dim=-1, largest=False, sorted=True)[1]
            topk_index = topk_index[:, 1:]  # Exclude the self from the topk results
    
            # Select nearest neighbors for all cells
            NN_cells = transformer_output[topk_index, 0, :]  # Shape [n_cells, K_NN, 512]

            # Compute the mean of nearest neighbors
            cell_emb_mean = NN_cells.mean(1)  # Shape [n_cells, 512]
    
            if self.use_batch_labels:
                batch_emb = batch_emb  # Use batch embeddings for all cells
                # TODO: Add slide embedding logic here if necessary

            # Decoder step
            out_mvc = self.impute_mvc_decoder(
                cell_emb_mean
                if not self.use_batch_labels
                else torch.cat([cell_emb_mean, batch_emb], dim=1),
                self.cur_gene_token_embs
            )
    
            output["impute_pred"] = out_mvc['pred']

        return output

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Mapping[str, Tensor]:
        """
        Wrapper to call either generative_forward or perceptual_forward, depending
        on the value of the "generative_training" kwarg.
        """
        if "generative_training" not in kwargs:
            # raise ValueError("generative_training kwarg is required")
            warnings.warn(
                "generative_training kwarg is required but not provided! "
                "Using False and calling perceptual_forward instead"
            )
            return self.perceptual_forward(*args, **kwargs)

        # get the generative training flag and pop it out
        do_generative_training = kwargs.pop("generative_training")
        if do_generative_training:
            return self.generative_forward(*args, **kwargs)
        else:
            return self.perceptual_forward(*args, **kwargs)

    def generative_forward(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        MVC_impute: bool = False,
        do_sample: bool = False,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            pcpt_genes (:obj:`Tensor`): token ids of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_values (:obj:`Tensor`): token values of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_key_padding_mask (:obj:`Tensor`): mask for pcpt_genes, shape
                [batch_size, seq_len]
            gen_genes (:obj:`Tensor`): token ids of the generative part, shape
                [batch_size, seq_len]
            gen_key_padding_mask (:obj:`Tensor`): mask for gen_genes, shape
                [batch_size, seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            do_sample (:obj:`bool`): whether to do sampling from bernoulli for
                generated zero predictions.
            input_cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size,
                embsize]

        Returns:
            :obj:`Mapping[str, Tensor]`:
                - pred (:obj:`Tensor`): prediction, shape [batch_size, seq_len]
                - cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size,
                    embsize]
        """

        pcpt_output, gen_output = self.transformer_generate(
            pcpt_genes,
            pcpt_values,
            pcpt_key_padding_mask,
            gen_genes,
            gen_key_padding_mask,
            batch_labels,
            input_cell_emb=input_cell_emb,
        )
        if gen_output is None:
            transformer_output = pcpt_output
        else:
            transformer_output = torch.cat([pcpt_output, gen_output], dim=1)
        
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)

        output = {}
        decoder_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=decoder_output["zero_probs"])
            full_preds = bernoulli.sample() * decoder_output["pred"]
            output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
            output["gen_preds"] = full_preds[:, pcpt_genes.shape[1] :]
        else:
            full_preds = decoder_output["pred"]  # (batch, seq_len)
            output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
            output["gen_preds"] = full_preds[:, pcpt_genes.shape[1] :]
        if self.explicit_zero_prob:
            output["zero_probs"] = decoder_output["zero_probs"]
        if MVC_impute:
            output['coordinates'] = coordinates

        output = self._extend_output(
            output,
            transformer_output,
            batch_emb=batch_emb if self.use_batch_labels else None,
            CLS=CLS,
            MVC=MVC,
            ECS=ECS,
            MVC_impute=MVC_impute,
            do_sample=do_sample,
        )

        return output

    def perceptual_forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
        CLS: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        MVC_impute: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(
            src, values, src_key_padding_mask, batch_labels
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        if MVC_impute:
            output['values'] = values
            output['coordinates'] = coordinates

        output = self._extend_output(
            output,
            transformer_output,
            batch_emb=batch_emb if self.use_batch_labels else None,
            CLS=CLS,
            MVC=MVC,
            ECS=ECS,
            MVC_impute=MVC_impute,
            do_sample=do_sample,
        )

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device)
                if batch_labels is not None
                else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs