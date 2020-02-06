from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(cfg):
    if cfg is None:
        return SGD

    else:
        opt_name = cfg["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        return key2opt[opt_name]


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, preheat, num_steps, power, i_iter, args):
    if i_iter < preheat:
        lr = lr_warmup(args.lr, i_iter, preheat)
    else:
        lr = lr_poly(args.lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

