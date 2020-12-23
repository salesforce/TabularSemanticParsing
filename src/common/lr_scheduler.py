"""
 Copyright (c) 2020, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Customized learning rate scheduler.
 Code adapted from: https://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html
"""

from torch.optim import Optimizer
from torch._six import inf
from functools import partial


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.


        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class StepLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]


class LinearScheduler(_LRScheduler):
    """
    Decay the LR linearly based on the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
         decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, optimizer, warmup_init_lrs, num_warmup_steps, num_steps, target_lrs=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        if target_lrs is None:
            target_lrs = [0 for _ in self.base_lrs]
        assert(len(self.base_lrs) == len(warmup_init_lrs) == len(num_warmup_steps) == len(target_lrs))
        self.target_lrs = target_lrs
        self.num_steps = num_steps
        self.warmup_init_lrs = warmup_init_lrs
        self.num_warmup_steps = num_warmup_steps
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                    zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_factors = [(target_lr - base_lr) / (self.num_steps - num_warmup_step)
                              for base_lr, target_lr, num_warmup_step in
                                zip(self.base_lrs, self.target_lrs, self.num_warmup_steps)]
        self.step(last_epoch + 1)

    def get_lr(self):
        return [self.update_lr(warmup_init_lr, num_warmup_step, base_lr, lr_linear_step, decay_factor)
                for warmup_init_lr, num_warmup_step, base_lr, lr_linear_step, decay_factor in
                    zip(self.warmup_init_lrs, self.num_warmup_steps, self.base_lrs, self.lr_linear_steps,
                        self.decay_factors)]

    def update_lr(self, warmup_init_lr, num_warmup_step, base_lr, lr_linear_step, decay_factor):
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + (self.last_epoch + 1) * lr_linear_step
        else:
            lr = decay_factor * (self.last_epoch + 1 - num_warmup_step) + base_lr
        return lr


class InverseSquareRootScheduler(_LRScheduler):
    """
    Code adaped from
        https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py

    Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
         decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, optimizer, warmup_init_lrs, num_warmup_steps, num_steps, target_lrs=None, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        if target_lrs is None:
            target_lrs = [0 for _ in self.base_lrs]
        assert(len(self.base_lrs) == len(warmup_init_lrs) == len(num_warmup_steps) == len(target_lrs))
        self.num_steps = num_steps
        self.warmup_init_lrs = warmup_init_lrs
        self.num_warmup_steps = num_warmup_steps
        self.target_lrs = target_lrs
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                    zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_bases = [(base_lr * num_warmup_step ** 0.5)
                              for base_lr, num_warmup_step in
                                zip(self.base_lrs, self.num_warmup_steps)]
        if target_lrs is None:
            self.offset_factors = [0 for _ in self.base_lrs]
        else:
            self.offset_factors = [(decay_base * self.num_steps ** -0.5 - target_lr) / self.num_steps
                                     for decay_base, target_lr in
                                        zip(self.decay_bases, self.target_lrs)]
        self.step(last_epoch + 1)

    def get_lr(self):
        return [self.update_lr(warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor)
                for warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor in
                    zip(self.warmup_init_lrs, self.num_warmup_steps, self.lr_linear_steps, self.decay_bases,
                        self.offset_factors)]

    def update_lr(self, warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor):
        num_steps = (self.last_epoch + 1)
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + num_steps * lr_linear_step
        else:
            lr = decay_base * num_steps ** -0.5 - offset_factor * num_steps
        return lr


class InversePowerScheduler(_LRScheduler):
    """
    Decay the LR based on the inverse power root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
         decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, optimizer, gamma, warmup_init_lrs, num_warmup_steps, last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma
        assert(len(self.base_lrs) == len(warmup_init_lrs) == len(num_warmup_steps))
        self.warmup_init_lrs = warmup_init_lrs
        self.num_warmup_steps = num_warmup_steps
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                    zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_factors = [(base_lr * num_warmup_step ** self.gamma)
                              for base_lr, num_warmup_step in
                                zip(self.base_lrs, self.num_warmup_steps)]
        self.step(last_epoch + 1)

    def get_lr(self):
        return [self.update_lr(warmup_init_lr, num_warmup_step, lr_linear_step, decay_factor)
                for warmup_init_lr, num_warmup_step, lr_linear_step, decay_factor in
                    zip(self.warmup_init_lrs, self.num_warmup_steps, self.lr_linear_steps, self.decay_factors)]

    def update_lr(self, warmup_init_lr, num_warmup_step, lr_linear_step, decay_factor):
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + (self.last_epoch + 1) * lr_linear_step
        else:
            lr = decay_factor * (self.last_epoch + 1) ** -self.gamma
        return lr


class HybridScheduler(_LRScheduler):
    """
    Code adaped from
        https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py

    Decay the LR based on the inverse square root of the update number.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.

    During warmup:
        lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
         decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, optimizer, lr_schedulers_types, warmup_init_lrs, num_warmup_steps, num_steps, target_lrs=None,
                 last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        if target_lrs is None:
            target_lrs = [0 for _ in self.base_lrs]
        assert(len(self.base_lrs) == len(warmup_init_lrs) == len(num_warmup_steps) == len(target_lrs))
        self.num_steps = num_steps
        self.warmup_init_lrs = warmup_init_lrs
        self.num_warmup_steps = num_warmup_steps
        self.target_lrs = target_lrs
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                    zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_bases = [(base_lr * num_warmup_step ** 0.5)
                              for base_lr, num_warmup_step in
                                zip(self.base_lrs, self.num_warmup_steps)]
        self.lr_linear_steps = [((base_lr - warmup_init_lr) / num_warmup_step)
                                for base_lr, warmup_init_lr, num_warmup_step in
                                zip(self.base_lrs, self.warmup_init_lrs, self.num_warmup_steps)]
        self.decay_factors = [(target_lr - base_lr) / (self.num_steps - num_warmup_step)
                              for base_lr, target_lr, num_warmup_step in
                              zip(self.base_lrs, self.target_lrs, self.num_warmup_steps)]
        if target_lrs is None:
            self.offset_factors = [0 for _ in self.base_lrs]
        else:
            self.offset_factors = [(decay_base * self.num_steps ** -0.5 - target_lr) / self.num_steps
                                     for decay_base, target_lr in
                                        zip(self.decay_bases, self.target_lrs)]
        self.step(last_epoch + 1)

    def get_lr(self):
        return [
            self.linear_update_lr(self.warmup_init_lrs[0], self.num_warmup_steps[0], self.base_lrs[0],
                                  self.lr_linear_steps[0], self.decay_factors[0]),
            self.inverse_square_update_lr(self.warmup_init_lrs[1], self.num_warmup_steps[1], self.lr_linear_steps[1],
                                          self.decay_bases[1], self.offset_factors[1])
        ]

    def linear_update_lr(self, warmup_init_lr, num_warmup_step, base_lr, lr_linear_step, decay_factor):
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + (self.last_epoch + 1) * lr_linear_step
        else:
            lr = decay_factor * (self.last_epoch + 1 - num_warmup_step) + base_lr
        return lr

    def inverse_square_update_lr(self, warmup_init_lr, num_warmup_step, lr_linear_step, decay_base, offset_factor):
        num_steps = (self.last_epoch + 1)
        if self.last_epoch < num_warmup_step:
            lr = warmup_init_lr + num_steps * lr_linear_step
        else:
            lr = decay_base * num_steps ** -0.5 - offset_factor * num_steps
        return lr


class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
