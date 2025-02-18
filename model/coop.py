import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from model import objectives

def load_clip_to_cpu(args):
    backbone_name = args.pretrain_choice
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    
    model = clip.build_model(args.img_size,args.stride_size,state_dict or model.state_dict(), design_details)
    return model.float()

class CoOpPromptLearner(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        n_ctx = args.num_context
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        n_cls = args.batch_size
        
        # Initialize the prompt embeddings randomly
        prompt_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(prompt_vectors, std=0.02)
        
        # Make the prompt vectors learnable
        self.prompts = nn.Parameter(prompt_vectors)

        print('CoOp design: Contrastive Prompt Tuning')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.n_cls = n_cls
        
    def forward(self, num_batch):

        prompts = self.prompts.unsqueeze(0).expand(num_batch, -1, -1)
        return prompts 

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text_embedding):
        x = text_embedding + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class Custom_Coop_CLIP(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.prompt_learner = CoOpPromptLearner(args, clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.args = args
        self._set_task()

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
        
    def encode_text(self, text, num_batch):
        # Generate prompts and text embeddings, and then encode text features
        prompts = self.prompt_learner(num_batch)
        n_ctx = self.args.num_context
        text_embedding = self.clip_model.token_embedding(text).type(self.dtype)
        prefix = text_embedding[:, 0, :].unsqueeze(1)
        suffix = text_embedding[:, 1:-n_ctx, :]
        text_embedding = torch.cat((prefix, prompts, suffix), 1)
        text_features = self.text_encoder(text_embedding)
        text_features = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)+n_ctx].float()
        return text_features

    def forward(self, batch):

        ret = dict()
        logit_scale = self.logit_scale.exp()
        text = batch['caption_ids']
        image = batch['images']
        ret.update({'temperature': 1 / logit_scale})
        num_batch = image.shape[0]
        
        text_features = self.encode_text(text, num_batch)
        image_features = self.image_encoder(image.type(self.dtype)).float()

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(image_features, text_features, logit_scale)})
        if 'TAL' in self.current_task:
            ret.update({'TAL_loss': objectives.compute_TAL(image_features,text_features,batch['pids'],tau=self.args.tau)})
        return ret


def build_Coop(args):
    print(f"Loading CLIP (backbone: {args.pretrain_choice})")
    clip_model = load_clip_to_cpu(args)

    print("building custom vpt clip ")
    model = Custom_Coop_CLIP(args,clip_model)
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "prompt_learner" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    # model.clip_base.visual.proj.requires_grad_(True)
    # model.clip_base.text_projection.requires_grad_(True)

    enabled = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    print(f"Parameters to be updated: {enabled}")

    return model