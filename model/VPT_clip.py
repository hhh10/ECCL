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
    design_details = { "trainer": "VPT",
                    "vision_depth": args.vision_depth,
                      "vision_ctx": args.vision_ctx,
                      "language_depth": 0,
                      "language_ctx": 0}
    assert args.vision_depth >= 1, "For Vision Prompting, PROMPT_DEPTH_VISION should be >= 1"

    design_details = {"trainer": "VPT",
                      "vision_depth": args.vision_depth,
                      "vision_ctx": args.vision_ctx,
                      "language_depth": 0,
                      "language_ctx": 0}
    model = clip.build_model(args.img_size,args.stride_size,state_dict or model.state_dict(), design_details)

    return model.float()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class Custom_VPT_CLIP(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        # self.embeddings = FixedEmbeddings(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        # self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_base = clip_model
        self.args = args
        self._set_task()

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def forward(self, batch):

        ret = dict()
        logit_scale = self.logit_scale.exp()
        text = batch['caption_ids']
        image = batch['images']
        ret.update({'temperature': 1 / logit_scale})

        image_features = self.image_encoder(image.type(self.dtype))
        text_features = self.clip_base.encode_text(text)
        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(image_features, text_features, logit_scale)})
        if 'TAL' in self.current_task:
            ret.update({'TAL_loss': objectives.compute_TAL(image_features,text_features,batch['pids'],tau=self.args.tau)})
        return ret


def build_VPT(args):
    print(f"Loading CLIP (backbone: {args.pretrain_choice})")
    clip_model = load_clip_to_cpu(args)


    print("building custom vpt clip ")
    model = Custom_VPT_CLIP(args,clip_model)
    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"

    for name, param in model.named_parameters():
        if name_to_update not in name:
            # Make sure that VPT prompts are updated
            if "VPT" in name:
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