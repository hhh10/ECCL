import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    params = []

    print(f'Using {args.lr_factor} times learning rate for random init module ')

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr = args.lr * args.lr_factor * args.fine_tuning_factor # default 5.0

        # if "bias" in key:
        #     lr = args.lr * args.bias_lr_factor
        #     weight_decay = args.weight_decay_bias
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor * args.fine_tuning_factor

        if "encoder" in key and 'proj' in key and 'lora' not in key:
            lr = args.lr * args.fine_tuning_factor
            if "bias" in key:
                lr = lr * args.lr_factor
        if "lora" in key:
            # lr = args.lr * 0.1
            lr = args.lora_lr
        # if "prompt" in key and "proj" in key:
        #     lr = args.lr * 0.5
        #     print(key)
        # elif "prompt" in key:
        #     lr = args.lr
        # else:
        #     lr = args.lr * args.fine_tuning_factor
        # if 'ln' in key:
        #     lr = args.lr * args.fine_tuning_factor




        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    # param_group = model.parameters()

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay
        )


    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )

# def build_lr_scheduler(args,optimizer):
#     lr_scheduler = args.lrscheduler
