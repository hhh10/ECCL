import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.options import get_args
from model import objectives
import loralib as lora
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights


class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        n_cls = args.batch_size
        n_ctx = args.num_context
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        assert args.prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.prompt_depth  # max=12, but will create 11 such shared prompts
        
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)

        nn.init.normal_(ctx_vectors, std=0.02)
        
        print('MaPLe design: Multi-modal Prompt Learning')
        
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)

        self.proj.half()

        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        self.n_cls = n_cls
        self.n_ctx = n_ctx

    def forward(self,num_batch):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(num_batch, -1, -1)

        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return ctx, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer




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
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": args.num_context,
                      "lora":args.lora,
                    #   "lora_FFN":args.lora_FFN,
                      "lora_alpha":args.lora_alpha}

    model = clip.build_model(args.img_size,args.stride_size,state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        

    def forward(self,text, text_embedding,compound_prompts_deeper_text):
        x = text_embedding + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x @ self.text_projection
        return x

class CustomCLIP(nn.Module):
    def __init__(self, args, clip_model, num_classes):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(args, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_model = clip_model
        self.args = args
        self._set_task()

        self.embed_dim=512
        self.num_classes = num_classes

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    ### Modify forward function
    def forward(self, batch=None, text=None, img=None, img_memory=None, text_memory=None, instance_img_memory=None, instance_text_memory=None):
        if batch is not None:
            ret = dict()
            logit_scale = self.logit_scale.exp()
            text = batch['caption_ids']
            image = batch['images']
            ret.update({'temperature': 1 / logit_scale})
            num_batch = image.shape[0]

            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(
                num_batch)

            # text embedding
            n_ctx = self.args.num_context
            end_idx = - n_ctx
            text_embedding = self.clip_model.token_embedding(text).type(self.dtype)
            prefix = text_embedding[:, 0, :].unsqueeze(1)
            suffix = text_embedding[:, 1:end_idx, :]
            text_embedding = torch.cat((prefix, prompts, suffix), 1)
 
            # Modify to add MLM
            text_features = self.text_encoder(text, text_embedding, deep_compound_prompts_text)
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx,
                                                deep_compound_prompts_vision)
            t_feat = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1)+n_ctx].float()
            i_feat = image_features[:, 0, :].float()

            if 'memory' in self.current_task:
                ret.update({'memory_loss_i2t': self.args.lamda1 * objectives.compute_memory_i2t(i_feat, t_feat, text_memory, batch['pids'])})
                ret.update({'memory_loss_t2i': self.args.lamda1 * objectives.compute_memory_t2i(t_feat, i_feat, img_memory, batch['pids'])})
            if 'instance_memory' in self.current_task:
                ret.update({'instance_memory_loss_i2t': self.args.lamda2 * objectives.compute_instance_memory_i2t(i_feat, t_feat, instance_text_memory, batch['image_ids'], batch['pids'])})
                ret.update({'instance_memory_loss_t2i': self.args.lamda2 * objectives.compute_instance_memory_t2i(t_feat, i_feat, instance_img_memory, batch['image_ids'], batch['pids'])})
            if 'TAL' in self.current_task:
                ret.update({'TAL_loss': objectives.compute_TAL(i_feat,t_feat,batch['pids'],tau=self.args.tau)})

            if 'itc' in self.current_task:
                ret.update({'itc_loss':objectives.compute_itc(image_features, text_features, logit_scale)})
            # Modify to add MLM
            if 'sdm' in self.current_task:
                ret.update(
                    {'sdm_loss': objectives.compute_sdm(i_feat, t_feat, batch['pids'], logit_scale)})
         
            return ret

        elif text is not None:
            text = text
            num_batch = text.shape[0]
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(
                num_batch)

            # text embedding
            n_ctx = self.args.num_context

            ###### modify ######
            end_idx = - n_ctx
            text_embedding = self.clip_model.token_embedding(text).type(self.dtype)

            prefix = text_embedding[:, 0, :].unsqueeze(1)
            suffix = text_embedding[:, 1:end_idx, :]
            text_embedding = torch.cat((prefix, prompts, suffix), 1)

            text_features = self.text_encoder(text, text_embedding, deep_compound_prompts_text)

            # Modify to add MLM
            t_feat = text_features[torch.arange(text_features.shape[0]), text.argmax(dim=-1) + n_ctx].float()
            return t_feat
        else:
            image = img
            num_batch = image.shape[0]
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(
                num_batch)
            image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

            i_feat = image_features[:, 0, :].float()
            return i_feat


def build_maple(args,num_classes=11003):
    print("Loading CLIP")
    clip_model = load_clip_to_cpu(args)
    
    clip_param = sum(p.numel() for p in clip_model.parameters())
    num_param=sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    model = CustomCLIP(args,clip_model,num_classes)

    print("Turning off gradients in both the image and the text encoder")
    name_to_update = "prompt_learner"
    

    lora.mark_only_lora_as_trainable(model)

    for name,param in model.named_parameters():
        if 'mlp' in name and 'lora' in name:
            param.data = param.data.half()
    for name, param in model.named_parameters():
        if 'lora' in name and 'attn' in name:
            param.data = param.data.half()

    # for name,param in model.named_parameters():
    #     if 'cross' in name and 'ln' not in name or 'mlm' in name and 'ln' not in name:
    #         param.data = param.data.half()

    # for name, param in model.named_parameters():
    #     if 'classifier' in name:
    #         param.data = param.data.half()
    #         param.requires_grad_(True)


    for name, param in model.named_parameters():
        if name_to_update in name:
            # Make sure that VPT prompts are updated
            param.requires_grad_(True)
            
    # Double check
    enabled = set()
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("finetuning parametwes:",model_param/clip_param)

    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled.add(name)

    print(f"Parameters to be updated: {enabled}")

    return model



