import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_memory_i2t(image_features,text_features,text_memory, pids):
    loss_i2t = text_memory(image_features, text_features, pids)
    return loss_i2t 
def compute_memory_t2i(text_features,image_features,image_memory, pids):
    loss_t2i = image_memory(text_features, image_features, pids)
    return loss_t2i 
def compute_instance_memory_i2t(image_features,text_features,text_memory, image_ids, pids):
    loss_i2t = text_memory(image_features, text_features, image_ids, pids)
    return loss_i2t 
def compute_instance_memory_t2i(text_features,image_features,image_memory, image_ids, pids):
    loss_t2i = image_memory(text_features, image_features, image_ids, pids)
    return loss_t2i 
def compute_itc_momentum(image_features,text_features,image_features_all,text_features_all,sim_i2t_targets,
                         sim_t2i_targets,sim_i2i_targets,sim_t2t_targets, logit_scale):
    sim_i2t = logit_scale * image_features @ text_features_all
    sim_t2i = logit_scale * text_features @ image_features_all
    sim_i2i = logit_scale * image_features @ image_features_all
    sim_t2t = logit_scale * text_features @ text_features_all
    
    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets,dim=1).mean()
    loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets,dim=1).mean()
    loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets,dim=1).mean()
    
    loss = (loss_i2t + loss_t2i + loss_i2i + loss_t2t) / 4
    
    return loss
def compute_TAL(image_features, text_features, pid, tau=0.02, margin=0.2):
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels
    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss.sum()
    
def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_img_distillation_loss(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    # pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    # pid_dist = pid - pid.t()
    # labels = (pid_dist == 0).float()

    # if image_id != None:
    #     # print("Mix PID and ImageID to create soft label.")
    #     image_id = image_id.reshape((-1, 1))
    #     image_id_dist = image_id - image_id.t()
    #     image_id_mask = (image_id_dist == 0).float()
    #     labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    # i2t_cosine_theta = t2i_cosine_theta.t()
    i2i_cosine_theta = image_norm @ image_norm.t()

    # text_proj_image = logit_scale * t2i_cosine_theta
    # image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    # labels_distribute = labels / labels.sum(dim=1)

    # i2t_pred = F.softmax(image_proj_text, dim=1)
    # i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(t2i_cosine_theta, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(t2i_cosine_theta, dim=1) - F.log_softmax(t2i_cosine_theta, dim=1))

    loss = torch.mean(torch.sum(t2i_loss, dim=1)) 

    return loss


def reco_loss(images_embeddings, texts_embeddings, lambda_param=0.6):
    """
    Calculate the ReCo Loss.
    
    Args:
    - images_embeddings (torch.Tensor): Embeddings for images with shape (batch_size, embedding_dim).
    - texts_embeddings (torch.Tensor): Embeddings for texts with shape (batch_size, embedding_dim).
    - lambda_param (float): Weight for the negative pairs.
    
    Returns:
    - torch.Tensor: Calculated ReCo loss.
    """
    
    # Normalize embeddings
    images_embeddings = F.normalize(images_embeddings, p=2, dim=1)
    texts_embeddings = F.normalize(texts_embeddings, p=2, dim=1)
    
    # Cosine similarity matrix
    similarity_matrix = torch.matmul(images_embeddings, texts_embeddings.t())
    
    # Positive loss: Mean squared error for positive pairs (diagonal elements)
    positive_loss = torch.mean((torch.diag(similarity_matrix) - 1) ** 2)
    
    # Negative loss: For off-diagonal elements, penalize only those greater than 0
    mask_off_diagonal = 1 - torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    negative_pairs = similarity_matrix * mask_off_diagonal
    negative_loss = torch.sum(F.relu(negative_pairs) ** 2) / (similarity_matrix.size(0) * (similarity_matrix.size(0) - 1))
    
    # Total ReCo Loss
    reco_loss = positive_loss + lambda_param * negative_loss
    
    return reco_loss

def compute_sdm_soft(image_fetures, text_fetures, image_features_s, text_features_s, pid, logit_scale, image_id=None, factor=0.1, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
        

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)
    
    with torch.no_grad():
        image_s_norm = image_features_s / image_features_s.norm(dim=1, keepdim=True)
        text_s_norm = text_features_s / text_features_s.norm(dim=1, keepdim=True)
        t2i_sim_s = logit_scale * text_s_norm @ image_s_norm.t()
        i2t_sim_s = t2i_sim_s.t()
        i2t_label = factor * F.softmax(i2t_sim_s, dim=1) + (1 - factor) * labels_distribute
        t2i_label = factor * F.softmax(t2i_sim_s, dim=1) + (1 - factor) * labels_distribute
        

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(i2t_label + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(t2i_label + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_cyloss(image_fetures, text_fetures, logit_scale, inmodal_factor=1, crossmodal_factor=1, cyloss_fator=1):
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)
    logits_image_per_image = logit_scale * image_norm @ image_norm.t()
    logits_text_per_text = logit_scale * text_norm @ text_norm.t()
    inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (
                    logit_scale * logit_scale)
    logits_text_per_image = logit_scale * image_norm @ text_norm.t()
    logits_image_per_text = logit_scale * text_norm @ image_norm.t()
    crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (
                    logit_scale * logit_scale)
    loss = inmodal_cyclic_loss * inmodal_factor + crossmodal_cyclic_loss * crossmodal_factor
    return loss * cyloss_fator
    
    
    

# def compute_ndf(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
#     """
#     Similarity Distribution Matching
#     """
#     batch_size = image_fetures.shape[0]
#     pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
#     pid_dist = pid - pid.t()
#     labels = (pid_dist == 0).float()

#     if image_id != None:
#         # print("Mix PID and ImageID to create soft label.")
#         image_id = image_id.reshape((-1, 1))
#         image_id_dist = image_id - image_id.t()
#         image_id_mask = (image_id_dist == 0).float()
#         labels = (labels - image_id_mask) * factor + image_id_mask
#         # labels = (labels + image_id_mask) / 2

#     image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
#     text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

#     t2i_cosine_theta = text_norm @ image_norm.t()
#     i2t_cosine_theta = t2i_cosine_theta.t()

#     text_proj_image = logit_scale * t2i_cosine_theta
#     image_proj_text = logit_scale * i2t_cosine_theta

#     # normalize the true matching distribution
#     labels_distribute = labels / labels.sum(dim=1)

#     i2t_pred = F.log_softmax(image_proj_text, dim=1)
#     # i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
#     t2i_pred = F.log_softmax(text_proj_image, dim=1)
#     # t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

#     # loss_1 = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
#     #
#     # i2t_loss_ndf = labels_distribute * (torch.log(labels_distribute)) - torch.log(labels_distribute + epsilon))
#     labels_distribute_ndf = F.log_softmax(labels_distribute, dim=1)
#     sdm_i2t_loss = F.kl_div(i2t_pred, labels_distribute+epsilon, reduction='batchmean')
#     sdm_t2i_loss = F.kl_div(t2i_pred, labels_distribute+epsilon, reduction='batchmean')
#     # ndf_i2t_loss = F.kl_div(labels_distribute_ndf,i2t_pred,  reduction='batchmean')
#     # ndf_t2i_loss = F.kl_div(labels_distribute_ndf,t2i_pred,  reduction='batchmean')
#     loss = sdm_t2i_loss + sdm_i2t_loss + ndf_t2i_loss + ndf_i2t_loss
#     return loss

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

