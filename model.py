import torch
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel, GPT2Config


class PatchLevelDecoder(PreTrainedModel):
    """
    Patch-level decoder: produces an embedding per 44-column patch and prepends
    BOS / control tokens (time-signature + sequence-length embedding) before
    feeding everything through a vanilla GPT-2 block.
    """
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = 44           # 88 / 2
        # Flattened one-hot (44 × 257) → hidden dim
        self.patch_embedding = torch.nn.Linear(44 * 257, config.n_embd)
        # Embeds total sequence length (0-127)
        self.seq_length_embedding = torch.nn.Embedding(128, config.n_embd)
        # Embeds time-signature index (0-4)
        self.time_signature_embedding = torch.nn.Embedding(5, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)

    def forward(
        self,
        patches: torch.Tensor,
        time_signature: torch.Tensor,
        thumb_len: torch.Tensor,
        masks=None,
    ):
        """
        Parameters
        ----------
        patches : Tensor  [bs, L_patches, 44]
            Patch tokens to be embedded.
        time_signature : Tensor  [bs, 1]
            Index 0–4 indicating the current meter.
        thumb_len : Tensor  [bs, 1]
            Sequence-length token (quantised target length).
        masks : Tensor or None
            Optional attention masks.

        Returns
        -------
        Base GPT-2 output (same structure as transformers.GPT2Model)
        """
        # (1) One-hot encode patches (ignore last dummy row) → [bs, L-1, 44, 257]
        patches = torch.nn.functional.one_hot(
            patches[:, :-1, :], num_classes=257
        ).to(self.dtype)

        # (2) Flatten to [bs, L-1, 44×257]
        patches = patches.reshape(len(patches), -1, 44 * 257)

        # (3) Linear projection → [bs, L-1, d_model]
        patches = self.patch_embedding(patches.to(self.device))

        # (4) Build control embeddings
        ts_embed = self.time_signature_embedding(time_signature)     # [bs, 1, d_model]
        seq_len_embed = self.seq_length_embedding(thumb_len)         # [bs, 1, d_model]

        # (5) Prepend time-signature token
        patches = torch.cat((ts_embed, patches), dim=1)
        # (6) Prepend BOS / length control token
        patches = torch.cat((seq_len_embed, patches), dim=1)

        if masks is None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches, attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    """
    Character-level decoder: auto-regressively expands each patch embedding
    into its 44-length token sequence.
    """
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.base = GPT2LMHeadModel(config)
        self.config = config

    def forward(self, encoded_patches: torch.Tensor, target_patches: torch.Tensor):
        """
        Parameters
        ----------
        encoded_patches : Tensor  [bs, L_patch, d_model]
            Patch embeddings from PatchLevelDecoder.
        target_patches : Tensor  [bs, L_patch, 44]
            Ground-truth patch tokens.

        Returns
        -------
        transformers.CausalLMOutputWithCrossAttentions
        """
        encoded_patches = encoded_patches.reshape(-1, self.config.n_embd)   # [bs*L, d]
        target_patches = target_patches.reshape(-1, 44)                     # [bs*L, 44]

        # Insert BOS (value==1) at position 0
        target_patches = torch.cat(
            (torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id,
             target_patches),
            dim=1,
        )                                                                   # [bs*L, 45]

        labels = target_patches.clone()
        labels[labels == 256] = -100    # Ignore PAD=256 when computing loss

        # Map tokens → embeddings
        inputs_embeds = torch.nn.functional.embedding(
            target_patches, self.base.transformer.wte.weight
        )                                                                   # [bs*L, 45, d]

        # Prepend the encoded patch representation
        inputs_embeds = torch.cat(
            (encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]),
            dim=1,
        )

        return self.base(inputs_embeds=inputs_embeds, labels=labels)

    def generate(
        self,
        encoded_patches: torch.Tensor,
        max_length: int = 44,
        temperature: float = 0.7,
        top_k: int = 10,
    ):
        """
        Auto-regressively sample 44 tokens for each patch.

        Returns
        -------
        Tensor  [bs, 1, 44]
            Generated patch tokens.
        """
        inputs_embeds = encoded_patches                  # [bs, 1, d_model]
        token_idx = []                                   # list of [bs, 1]

        for _ in range(max_length):
            outputs = self.base(inputs_embeds=inputs_embeds)
            logits = outputs.logits[:, -1, :] / temperature

            # Top-k filtering in logit space
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, top_k_indices, top_k_values)
                logits = mask

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)      # [bs, 1]
            token_idx.append(next_token)

            next_embed = torch.nn.functional.embedding(
                next_token, self.base.transformer.wte.weight
            )
            inputs_embeds = torch.cat((inputs_embeds, next_embed), dim=1)

        token_idx = torch.cat(token_idx, dim=1)                    # [bs, 44]
        token_idx = token_idx.view(encoded_patches.size(0), 1, -1) # [bs, 1, 44]
        return token_idx


class PixelGenLMHeadModel(PreTrainedModel):
    """
    Hierarchical transformer model:
        * PatchLevelDecoder (coarse timeline)
        * CharLevelDecoder  (fine intra-patch tokens)
    """

    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)

    # --------------------------- TRAINING ---------------------------

    def forward(
        self,
        patches: torch.Tensor,
        time_signature: torch.Tensor,
        thumb_len: torch.Tensor,
        masks: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        patches : Tensor  [bs, L_patches, 44]
        masks   : optional attention masks

        Returns
        -------
        Output from CharLevelDecoder (includes loss when labels present)
        """
        encoded = self.patch_level_decoder(
            patches, time_signature=time_signature, thumb_len=thumb_len, masks=masks
        )["last_hidden_state"][:, 1:, :]          # remove control token
        return self.char_level_decoder(encoded, patches)

    # ------------------------ INFERENCE (STANDARD) ------------------

    def generate(
        self,
        patches: torch.Tensor,
        time_signature: torch.Tensor,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 10,
        seq_len: int = 30,
    ):
        """
        Autoregressively extend `patches` until reaching `max_length`
        or until three EOS(257) rows have been seen.
        """
        idx_eos_seen = 0
        pad_counter = 0
        generated_patches = patches

        for _ in range(max_length - generated_patches.size(1)):
            encoded_all = self.patch_level_decoder(
                generated_patches, time_signature, thumb_len=seq_len
            )["last_hidden_state"]
            last_encoded = encoded_all[:, -1, :]
            gen_token = self.char_level_decoder.generate(
                last_encoded.unsqueeze(1),
                max_length=44,
                temperature=temperature,
                top_k=top_k,
            )                                           # [bs, 1, 44]

            # EOS(257) handling: replace with PAD(256) and track count
            if torch.eq(gen_token, 257).any():
                idx_eos_seen += 1
                gen_token[:, :, :] = 256
                if idx_eos_seen == 3:
                    break

            # Additional padding every 5 steps after two EOS rows
            if idx_eos_seen == 2:
                pad_counter += 1
                if pad_counter % 5 == 0:
                    gen_token[:, :, :] = 256

            generated_patches = torch.cat((generated_patches, gen_token), dim=1)

        return generated_patches

    # --------------------- INFERENCE (SPECIAL VARIANT) --------------

    def generate_special(self, gnd_patches: torch.Tensor, div_patches: torch.Tensor,time_signature :torch.Tensor, max_length=64,temperature=1.0, top_k: int = 10,seq_len:int = 30):
        """
        Generate patches based on the input patches in an auto-regressive manner.
        :param patches: the patches to be encoded  #[bs, l_patches, 44]
        :param max_length: the maximum number of steps to generate (including the initial patches)
        :return: the generated patches
        """
        idx = 0
        len = 0 
        flg = 0
        # Initialize with the provided patches
        generated_patches =  div_patches  # Start with the provided patches

        # Start the autoregressive generation loop
        for step in range(max_length - generated_patches.size(1)):  # Stop when reaching the max_length
            # Step 1: Pass the current patches into PatchLevelDecoder to get the next encoded patch
            
            encoded_patches = self.patch_level_decoder(generated_patches,time_signature,thumb_len = seq_len)["last_hidden_state"]
            # Step 2: Use only the last hidden state from encoded_patches for CharLevelDecoder
            last_encoded_patch = encoded_patches[:, -1, :]  # Get the last hidden state for the last patch
            # Step 3: Use CharLevelDecoder to generate the next patch token based on the last hidden state
            generated_token = self.char_level_decoder.generate(last_encoded_patch.unsqueeze(1),
                                                               max_length=44,temperature=temperature,top_k = top_k)  # Generate the next token for the patch
            # Step 4: Append the generated token back into the sequence
            # [bs 1 44] 的token idx 出现257 idx 的token 则停止生成
            if torch.eq(generated_token, 257).any():
                idx += 1
                generated_token[:,:,:] = 256
                if idx == 3:
                     break  # Exit the loop if token 257 is found
            if idx == 2:
                len+=1
                if flg < gnd_patches.size(1):
                    generated_token = gnd_patches[:,flg,:].unsqueeze(1)
                    flg += 1
                    if len % 5 == 0 and len !=0:
                        generated_token[:,:,:] = 256
                        flg -= 1

                if len % 5 == 0 and len !=0:
                    generated_token[:,:,:] = 256

                
            
            generated_patches = torch.cat((generated_patches, generated_token), dim=1)

        return generated_patches  # Return the generated patches
