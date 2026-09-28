# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from IPython import embed

class WorldModel(nn.Module):
    def __init__(
            self,
            model_lang,
            model_vision,
            model_causal_transformer,
            hidden_size,
            sequence_length,
            chunk_size,
            per_latent_action_len,
            latent_action_codebook_size,
            latent_action_pred,
            img_feat_dim,
            lang_feat_dim,
            freeze_lang=True,
            freeze_vision=True,
            obs_pred=True,
            use_timestep_embedding=False,
            use_latent_action_pos_embedding=False,
            **kwargs
    ):
        super(WorldModel, self).__init__()
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.per_latent_action_len = per_latent_action_len
        self.latent_action_codebook_size = latent_action_codebook_size
        
        self.latent_action_pred = latent_action_pred

        # GPT
        self.hidden_size = hidden_size
        self.model_causal_transformer = model_causal_transformer

        # Language Encoder
        self.model_lang = model_lang
        self.freeze_lang = freeze_lang

        self.obs_pred = obs_pred

        if freeze_lang:
            for _, param in self.model_lang.named_parameters():
                param.requires_grad = False
        
        # Visual Encoder
        self.model_vision = model_vision
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for _, param in self.model_vision.named_parameters():
                param.requires_grad = False
                
        self.lang_feat_dim = lang_feat_dim
        self.img_feat_dim = img_feat_dim
        
        # Condition embedding
        self.embed_condition = nn.Embedding(1, hidden_size)

        # Embedding function for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)

        # Embedding function for vision
        self.embed_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        
        # Embedding functions for latent actions
        self.embed_latent_action = nn.Embedding(latent_action_codebook_size, hidden_size)

        # Timestep Embeddings
        self.use_timestep_embedding = use_timestep_embedding
        # print(f"use_timestep_embedding: {self.use_timestep_embedding}")
        if self.use_timestep_embedding:
            self.embed_timestep = nn.Embedding(sequence_length, hidden_size)

        # Latent Motion Positional Embeddings
        self.use_latent_action_pos_embedding = use_latent_action_pos_embedding
        # print(f"use_latent_action_pos_embedding: {self.use_latent_action_pos_embedding}")
        if self.use_latent_action_pos_embedding:
            self.embed_latent_action_pos = nn.Embedding(per_latent_action_len+1, hidden_size)

        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        if self.latent_action_pred:
            # Latent action query token
            self.latent_action_queries = nn.Embedding(1, hidden_size)

            # Latent action prediction
            self.pred_latent_action_head = nn.Linear(hidden_size, latent_action_codebook_size, bias=False)

        if self.obs_pred:
            self.obs_query = nn.Parameter(torch.zeros(1, per_latent_action_len, hidden_size), requires_grad=True) 

            latent_dim = 128 # 与encoder中latent_dim相同
            self.pred_obs_head = nn.Linear(hidden_size, latent_dim, bias=False)


    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = ['model_lang', 'model_vision']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict

    def forward(self, 
                rgb, # (b, 1, c, h, w)
                language,
                latent_action_ids, # (b, per_latent_action_len)
                train=True,
                lang_attention_mask=None,
                **kwargs
    ):
        latent_action_x = None
        latent_action_preds = None

        batch_size, _, c, h, w = rgb.shape
        sequence_length = self.sequence_length
        

        # Embed language
        if self.freeze_lang:
            with torch.no_grad():
                lang_embeddings = self.model_lang(input_ids=language, attention_mask=lang_attention_mask).last_hidden_state
        else:
            lang_embeddings = self.model_lang(input_ids=language, attention_mask=lang_attention_mask).last_hidden_state
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, n_lang_tokens, h)

        # Get obs and patch feature from Visual Encoder
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb.dtype, device=rgb.device).view(1, 1, c, 1, 1)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], dtype=rgb.dtype, device=rgb.device).view(1, 1, c, 1, 1)
        rgb = (rgb - rgb_mean) / (rgb_std + 1e-6)
        if self.freeze_vision:
            with torch.no_grad():
                img_embeddings = self.model_vision(rgb.reshape(batch_size, c, h, w))  # (b, img_tokens, img_feat_dim)
        else:
            img_embeddings = self.model_vision(rgb.reshape(batch_size, c, h, w))  # (b, img_tokens, img_feat_dim)
        img_embeddings = self.embed_img(img_embeddings.float())  # (b, 1, h)
        
        cond_stacked_inputs = torch.cat((lang_embeddings, img_embeddings), dim=1)  # (b, n_cond_tokens, h)
        obs_query = self.obs_query.repeat(batch_size, 1, 1) if self.obs_pred else None
        if self.obs_pred:
            cond_stacked_inputs = torch.cat((cond_stacked_inputs, obs_query), dim=1)  # (b, n_cond_tokens + n_obs_tokens, h)

        latent_action_queries = self.latent_action_queries.weight  # (1, h)
        latent_action_queries = latent_action_queries.view(1, 1, self.hidden_size).repeat(batch_size, 1, 1)  # (b, 1, h)
        latent_action_embeddings = self.embed_latent_action(latent_action_ids) # (b, per_latent_action_len, h)
        act_stacked_inputs = torch.cat((latent_action_queries, latent_action_embeddings), dim=1)  # (b, per_latent_action_len+1, h)

        if self.use_latent_action_pos_embedding:
            latent_action_pos_embeddings = self.embed_latent_action_pos.weight # (per_latent_action_len+1, h)
            act_stacked_inputs = act_stacked_inputs + latent_action_pos_embeddings


        # Number of tokens
        n_lang_tokens = lang_embeddings.shape[1]
        n_vision_tokens = img_embeddings.shape[1]
        n_obs_tokens = obs_query.shape[1] if self.obs_pred else 0
        n_cond_tokens = n_lang_tokens + n_vision_tokens + n_obs_tokens # n_cond_tokens

        n_tokens = 1 + self.per_latent_action_len
        
        # Layer norm
        cond_stacked_inputs = cond_stacked_inputs.reshape(batch_size, n_cond_tokens, self.hidden_size) # (b, n_cond_tokens, h)
        stacked_inputs = torch.cat([cond_stacked_inputs, act_stacked_inputs], dim=1) # (b, n_cond_tokens + n_tokens, h)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Attention Mask
        cond_stacked_attention_mask = torch.ones((batch_size, 1, n_cond_tokens), dtype=torch.long, device=rgb.device)
        if lang_attention_mask is not None:
            cond_stacked_attention_mask[:, :, :n_lang_tokens] = lang_attention_mask.view(batch_size, 1, -1) # (b, 1, n_cond_tokens)

        act_stacked_attention_mask = torch.ones((batch_size, sequence_length, 1), dtype=torch.long, device=rgb.device)
        act_stacked_attention_mask = act_stacked_attention_mask.repeat(1, 1, n_tokens) # (b, t, n_tokens)

        stacked_attention_mask = torch.cat([cond_stacked_attention_mask.reshape(batch_size, n_cond_tokens), \
            act_stacked_attention_mask.reshape(batch_size, n_tokens * sequence_length)], dim=1) # (b, n_cond_tokens + t*n_tokens)
        stacked_attention_mask = stacked_attention_mask.reshape(batch_size, 1, 1, n_cond_tokens + n_tokens * sequence_length)
        stacked_attention_mask = stacked_attention_mask.repeat(1, 1, n_cond_tokens + n_tokens * sequence_length, 1) # (b, 1, n_cond_tokens + t*n_tokens, n_cond_tokens + t*n_tokens)

        # GPT forward pass
        transformer_outputs = self.model_causal_transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        act_x = x[:, n_cond_tokens:] # (b, n_tokens, h)
        obs_x = x[:, (n_cond_tokens-n_obs_tokens):n_cond_tokens] if self.obs_pred else None # (b, sequence_length, n_tokens - obs_query_start_i, h)

        # Latent action prediction
        if train:
            latent_action_x = act_x[:, :self.per_latent_action_len] # (b, per_latent_action_len, h)
            latent_action_preds = self.pred_latent_action_head(latent_action_x) # (b, per_latent_action_len, latent_action_codebook_size)
            obs_x = self.pred_obs_head(obs_x) if self.obs_pred else None # (b, sequence_length, n_tokens - obs_tokens, latent_dim)
        else:
            act_x = act_x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)
            latent_action_id_preds, act_x = self.decode_latent_action(
                act_x=act_x, 
                stacked_inputs=stacked_inputs,
                stacked_attention_mask=stacked_attention_mask, 
                latent_action_query_token_start_i=0, 
                n_cond_tokens=n_cond_tokens,
                **kwargs
            )
            latent_action_x = act_x.reshape(batch_size, sequence_length*n_tokens, self.hidden_size)
            latent_action_x = latent_action_x[:, :self.per_latent_action_len]

        prediction = {
            'latent_action_preds': latent_action_preds, # (b, per_latent_action_len, latent_action_codebook_size)
            'latent_action_feature': latent_action_x,   # (b, per_latent_action_len, h)
            'cond_input_feature': cond_stacked_inputs,   # (b, n_cond_tokens, h)
            'obs_features': obs_x, # (b, sequence_length, n_obs_tokens, latent_dim)
        }
        if self.latent_action_pred and (not train):
            prediction['latent_action_id_preds'] = latent_action_id_preds
            
        return prediction


    def decode_latent_action(self, act_x, stacked_inputs, stacked_attention_mask, latent_action_query_token_start_i, n_cond_tokens,
                             temperature=1.0, sample=False, top_k=0, top_p=1.0, beam_size=1, parallel=True, buffer_len=10):
        
        if parallel == True:
            assert sample==False and beam_size==1
            
        if sample==False and beam_size==1:
            return self.greedy_decode(
                act_x=act_x, 
                stacked_inputs=stacked_inputs, 
                stacked_attention_mask=stacked_attention_mask, 
                latent_action_query_token_start_i=latent_action_query_token_start_i, 
                parallel=parallel,
                buffer_len=buffer_len,
                n_cond_tokens=n_cond_tokens
            )
        else:
            return self.beam_search_decode(
                act_x=act_x, 
                stacked_inputs=stacked_inputs, 
                stacked_attention_mask=stacked_attention_mask, 
                latent_action_query_token_start_i=latent_action_query_token_start_i,
                temperature=temperature, 
                sample=sample, 
                top_k=top_k, 
                top_p=top_p, 
                beam_size=beam_size,
                buffer_len=buffer_len,
                n_cond_tokens=n_cond_tokens
            )


    def greedy_decode(self, act_x, stacked_inputs, stacked_attention_mask, latent_action_query_token_start_i, n_cond_tokens, parallel=True, buffer_len=10):

        batch_size, sequence_length, n_tokens, _ = act_x.shape
        cond_stacked_inputs = stacked_inputs[:, :n_cond_tokens]
        act_stacked_inputs = stacked_inputs[:, n_cond_tokens:]

        latent_action_id_preds = []

        for j in range(self.per_latent_action_len):
            if parallel:
                cur_latent_action_hidden_states = act_x[:, :, latent_action_query_token_start_i+j] # (b, t, h)
            else:
                cur_latent_action_hidden_states = act_x[:, buffer_len-1, latent_action_query_token_start_i+j] # (b, h)

            logits = self.pred_latent_action_head(cur_latent_action_hidden_states) # (b, t, latent_action_codebook_size) or (b, latent_action_codebook_size)
            probs = F.softmax(logits, dim=-1) # (b, t, latent_action_codebook_size) or (b, latent_action_codebook_size)
            cur_pred_latent_action_ids = torch.argmax(probs, dim=-1) # (b, t) or (b,)
            cur_pred_latent_action_embeddings = self.embed_latent_action(cur_pred_latent_action_ids) # (b, t, h) or  # (b, h)

            if self.use_latent_action_pos_embedding:
                cur_latent_action_pos_embedding = self.embed_latent_action_pos.weight[j+1] # (h,)
                cur_pred_latent_action_embeddings += cur_latent_action_pos_embedding

            if self.use_timestep_embedding:
                if parallel:
                    cur_time_embeddings = self.embed_timestep.weight # (t, h)
                else:
                    cur_time_embeddings = self.embed_timestep.weight[buffer_len-1] # (h,)
                cur_pred_latent_action_embeddings += cur_time_embeddings

            cur_pred_latent_action_inputs = self.embed_ln(cur_pred_latent_action_embeddings) # (b, t, h) or (b, h)
            act_stacked_inputs = act_stacked_inputs.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)

            if parallel:
                act_stacked_inputs[:, :, latent_action_query_token_start_i+j+1] = cur_pred_latent_action_inputs
            else:
                act_stacked_inputs[:, buffer_len-1, latent_action_query_token_start_i+j+1] = cur_pred_latent_action_inputs

            act_stacked_inputs = act_stacked_inputs.reshape(batch_size, n_tokens * sequence_length, self.hidden_size)
            stacked_inputs = torch.cat([cond_stacked_inputs, act_stacked_inputs], dim=1)
            transformer_outputs = self.model_causal_transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=None # stacked_attention_mask,
            )
            x = transformer_outputs['last_hidden_state']
            act_x = x[:, n_cond_tokens:] # (b,t*n_tokens, h)
            act_x = act_x.reshape(batch_size, sequence_length, n_tokens, self.hidden_size)
            latent_action_id_preds.append(cur_pred_latent_action_ids)
        latent_action_id_preds = torch.stack(latent_action_id_preds, dim=-1) # (b, t, per_latent_action_len) or (b, per_latent_action_len)
        return latent_action_id_preds, act_x


    def beam_search_decode(self, act_x, stacked_inputs, stacked_attention_mask, latent_action_query_token_start_i, n_cond_tokens,
                           temperature=1.0, sample=False, top_k=0, top_p=1.0, beam_size=1, buffer_len=10):

        batch_size, sequence_length, n_tokens, hidden_size = act_x.shape

        cond_stacked_inputs = stacked_inputs[:, :n_cond_tokens]
        cond_stacked_inputs = cond_stacked_inputs.unsqueeze(1).expand(-1, beam_size, -1, -1)
        cond_stacked_inputs = cond_stacked_inputs.reshape(batch_size * beam_size, n_cond_tokens, self.hidden_size)

        act_stacked_inputs = stacked_inputs[:, n_cond_tokens:]
        act_stacked_inputs = act_stacked_inputs.reshape(batch_size, sequence_length, n_tokens, self.hidden_size) # (b, t, n, h)

        latent_action_id_preds = torch.zeros((batch_size, beam_size, self.per_latent_action_len), dtype=torch.long, device=act_x.device) # (b, beam_size, per_latent_action_len)
        act_stacked_inputs = act_stacked_inputs.unsqueeze(1).expand(-1, beam_size, -1, -1, -1) # (b, beam_size, t, n, h)
        # stacked_attention_mask = # stacked_attention_mask.unsqueeze(1).expand(-1, beam_size, -1, -1, -1) # (b, beam_size, 1, cond_n+t*n, t*n)
        
        for j in range(self.per_latent_action_len):
            cur_latent_action_hidden_states = act_x[:, buffer_len-1, latent_action_query_token_start_i+j]  # (b, h) or (b*beam_size, h)
            logits = self.pred_latent_action_head(cur_latent_action_hidden_states)  # (b, latent_action_codebook_size) or (b*beam_size, latent_action_codebook_size)
            
            if sample:
                logits = logits / temperature
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = F.softmax(logits, dim=-1) # (b, latent_action_codebook_size) or (b*beam_size, latent_action_codebook_size)

            if j == 0:
                # At the first step, select top-k candidates for each batch
                if sample:
                    top_k_indices = torch.multinomial(probs, num_samples=beam_size) # (b, beam_size)
                    top_k_scores = torch.gather(probs.log(), -1, top_k_indices) # (b, beam_size)
                    top_k_scores, _indices = torch.sort(top_k_scores, descending=True, dim=1)
                    top_k_indices = torch.gather(top_k_indices, -1, _indices)
                else:
                    top_k_scores, top_k_indices = torch.topk(probs.log(), beam_size, dim=-1, largest=True, sorted=True)  # (b, beam_size)

                beam_scores = top_k_scores # (b, beam_size)
                latent_action_id_preds[:, :, j] = top_k_indices # (b, beam_size)
            else:
                # For the subsequent steps, calculate scores for all possible next tokens, and select top-k candidates
                probs = probs.view(batch_size, beam_size, self.latent_action_codebook_size)
                scores = beam_scores.unsqueeze(2) + probs.log()  # (b, beam_size, latent_action_codebook_size)
                scores = scores.view(batch_size, -1) # (b, beam_size * latent_action_codebook_size)

                if sample:
                    probs = torch.exp(scores) # (b, beam_size * latent_action_codebook_size)
                    top_k_indices = torch.multinomial(probs, num_samples=beam_size) # (b, beam_size)
                    top_k_scores = torch.gather(scores, -1, top_k_indices) # (b, beam_size)
                    top_k_scores, _indices = torch.sort(top_k_scores, descending=True, dim=1)
                    top_k_indices = torch.gather(top_k_indices, -1, _indices)
                else:
                    top_k_scores, top_k_indices = torch.topk(scores, beam_size, dim=-1, largest=True, sorted=True)  # (b, beam_size)

                # Update the scores and sequences for each beam
                beam_scores = top_k_scores
                prev_beam_indices = top_k_indices // self.latent_action_codebook_size  # (b, beam_size)
                next_token_indices = top_k_indices % self.latent_action_codebook_size  # (b, beam_size)

                latent_action_id_preds = latent_action_id_preds.gather(1, prev_beam_indices.unsqueeze(2).expand(-1, -1, self.per_latent_action_len)) # (b, beam_size, per_latent_action_len)
                latent_action_id_preds[:, :, j] = next_token_indices

                act_stacked_inputs = act_stacked_inputs.view(batch_size, beam_size, sequence_length, n_tokens, self.hidden_size)
                act_stacked_inputs = act_stacked_inputs.gather(1, prev_beam_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, sequence_length, n_tokens, hidden_size)) # (b, beam_size, t, n, h)

                # stacked_attention_mask = # stacked_attention_mask.reshape(batch_size, beam_size, 1, n_cond_tokens+sequence_length*n_tokens, n_cond_tokens+sequence_length*n_tokens)
                # stacked_attention_mask = # stacked_attention_mask.gather(1, prev_beam_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, n_cond_tokens+sequence_length*n_tokens, n_cond_tokens+sequence_length*n_tokens))

            cur_pred_latent_action_embeddings = self.embed_latent_action(latent_action_id_preds[:, :, j])  # (b, beam_size, h)

            if self.use_latent_action_pos_embedding:
                cur_latent_action_pos_embedding = self.embed_latent_action_pos.weight[j+1] # (h,)
                cur_pred_latent_action_embeddings += cur_latent_action_pos_embedding

            if self.use_timestep_embedding:
                cur_time_embeddings = self.embed_timestep.weight[buffer_len-1] # (h,)
                cur_pred_latent_action_embeddings += cur_time_embeddings
                
            cur_pred_latent_action_inputs = self.embed_ln(cur_pred_latent_action_embeddings)  # (b, beam_size, h)
            act_stacked_inputs = act_stacked_inputs.clone()
            act_stacked_inputs[:, :, buffer_len-1, latent_action_query_token_start_i+j+1] = cur_pred_latent_action_inputs
            act_stacked_inputs = act_stacked_inputs.view(batch_size * beam_size, n_tokens * sequence_length, self.hidden_size)

            stacked_inputs = torch.cat([cond_stacked_inputs, act_stacked_inputs], dim=1)
            # stacked_attention_mask = # stacked_attention_mask.reshape(batch_size*beam_size, 1, n_cond_tokens+sequence_length*n_tokens, n_cond_tokens+sequence_length*n_tokens)

            transformer_outputs = self.model_causal_transformer(
                inputs_embeds=stacked_inputs,
                attention_mask=None,
            )
            x = transformer_outputs['last_hidden_state']
            act_x = x[:, n_cond_tokens:] # (b,t*n_tokens, h)
            act_x = act_x.reshape(batch_size*beam_size, sequence_length, n_tokens, self.hidden_size)

        # Select the best sequence in each beam
        best_seq_indices = beam_scores.argmax(dim=-1)  # (b,)
        latent_action_id_preds = latent_action_id_preds.gather(1, best_seq_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.per_latent_action_len))
        latent_action_id_preds = latent_action_id_preds.squeeze(1)  # (b, per_latent_action_len)

        act_x = act_x.view(batch_size, beam_size, sequence_length, n_tokens, self.hidden_size)
        act_x = act_x.gather(1, best_seq_indices.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, sequence_length, n_tokens, self.hidden_size))
        act_x = act_x.squeeze(1)
        
        return latent_action_id_preds, act_x


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits
