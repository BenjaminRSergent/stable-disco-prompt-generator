import re

import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.scheduler as torchscheduler
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers


class KnowledgeTransferNetwork(torchmodules.BaseModel):
    name="KnowledgeTransferModel"
    def __init__(self, student, teacher, teacher_percent=0.5, name_suffix="", max_lr=4e-4, min_lr_divisor=20, epoch_batches=150000, step_size_up_epoch_mul=1, warmup_period_epoch_mul=1, gamma=0.75, last_epoch=-1, device=None):
        super().__init__(KnowledgeTransferNetwork.name+name_suffix, device=device)
        self._student = student
        self._teacher = teacher
        self._teacher_percent = teacher_percent
        self._student_percent = (1 - teacher_percent)
        
        for param in self._teacher.parameters():
            param.requires_grad = False
        
        self._optimizer = torch.optim.NAdam(
            self._student.parameters(), max_lr, betas=(0.88, 0.995)
        )
        
        self._scheduler = torchscheduler.make_cyclic_with_warmup(optimizer=self._optimizer,
                                                                 epoch_batches=epoch_batches,
                                                                 max_lr=max_lr,
                                                                 min_lr_divisor=min_lr_divisor,
                                                                 step_size_up_epoch_mul=step_size_up_epoch_mul,
                                                                 warmup_period_epoch_mul=warmup_period_epoch_mul,
                                                                 gamma=gamma,
                                                                 last_epoch=last_epoch,
                                                                 cycle_momentum=False)

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            student_loss = self._student._calc_batch_loss(x_inputs, y_targets)
            if not self.training:
                return student_loss
            
            teacher_out = self._teacher(x_inputs)
            return student_loss * self._student_percent + self._student._calc_batch_loss(x_inputs, teacher_out) * self._teacher_percent
            
    def forward(self, x_inputs):
        return self._student(x_inputs)
    
    def get_student(self):
        return self._student
        
            