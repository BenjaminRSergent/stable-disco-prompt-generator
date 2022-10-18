import ai.stabledisco.constants as sdconsts
import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.scheduler as torchscheduler
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
import ai.stabledisco.utils as sdutils
from ai.stabledisco.decoderpipeline.knowledgetransfernetwork import \
    KnowledgeTransferNetwork
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers


class IsTextFeaturesModel(torchmodules.BaseModel):
    name = "IsTextFeatureV4"
    
    @staticmethod
    def build_teacher(**kwargs):
        return IsTextFeaturesModel(*kwargs)
    
    @staticmethod
    def build_student(**kwargs):
        return IsTextFeaturesModel(name_suffix='Student', 
                res_unit_mul = 2,
                res_layers = 2,
                dense_unit_div = 1.5,
                dense_layer_base = 1.5,
                dropout = 0.15,
                **kwargs)
    
    @staticmethod
    def build_knowledge_transfer_model(teacher=None, epoch_batches=34200, teacher_checkpoint="best", **kwargs):
        if teacher is None:
            teacher = IsTextFeaturesModel.build_teacher()
            teacher.load_weights(teacher_checkpoint, strict=False)
        
        student = IsTextFeaturesModel.build_student()
        
        return KnowledgeTransferNetwork(student, teacher, epoch_batches=epoch_batches, name_suffix=IsTextFeaturesModel.name, **kwargs)
    
    
    def __init__(self, name_suffix='',
                 res_unit_mul = 6,
                 res_layers = 4,
                 dense_unit_div = 0.5,
                 dense_layer_base = 1.5,
                 dropout = 0.0,
                 max_lr=2e-4, min_lr_divisor=20, epoch_batches=32000, step_size_up_epoch_mul=0.5, warmup_period_epoch_mul=2, gamma=0.75, last_epoch=-1, device=None):
       
        super().__init__(IsTextFeaturesModel.name+name_suffix, device=device)
        # Input = (-1, 77, num_features)
        # Output = (-1, 77, vocab_size)
        # Loss = diffable encoding

        self._res_stack = torchlayers.ResDenseStack(input_size=sdconsts.feature_width,
                                                    block_width=sdconsts.feature_width*res_unit_mul,
                                                    layers=res_layers,
                                                    dropout=dropout,
                                                    activation=nn.LeakyReLU)
        
        dense_stack_units = [(int(dense_layer_base), int(self._res_stack.out_features/dense_unit_div)),
                             (int(dense_layer_base*2), int(self._res_stack.out_features/(2*dense_unit_div))),
                             (int(dense_layer_base*4), int(self._res_stack.out_features/(4*dense_unit_div)))]
        self._dense_stack = torchlayers.DenseStack(
            self._res_stack.out_features,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0,
        )
                             
        self._prob_out = torch.nn.Linear(dense_stack_units[-1][1], 1)
        nn.init.xavier_uniform_(self._prob_out.weight)
        
        self._loss_func = torch.nn.MSELoss()

        self._optimizer = torch.optim.AdamW(
            self.parameters(), max_lr, betas=(0.89, 0.995), weight_decay=1e-2
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
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1, 1), y_targets.float().view(-1, 1))

    def forward(self, features):
        x = features.float() / features.norm(dim=-1, keepdim=True)
        x = self._res_stack(x)
        x = self._dense_stack(x)
            
        return self._prob_out(x)

    def get_text_prob(self, features):
        return self(features).reshape(-1)

    def improve_text_prob(self, features, target_prob=0.96, max_diff=0.03, per_step=1e-6,  alpha=0.8, max_divs=100, verbose=False):
        # TODO: Extract common code with improve rating
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.view(1, -1)
            features = features / features.norm(dim=-1, keepdim=True)
            before_prob = self.get_text_prob(features)
            out_features = torch.clone(features)
            cosine_change = 0
            if verbose:
                print("Start prob", before_prob)

            best_out_features = out_features.clone()
            best_out_score = before_prob

            prev_prob = before_prob
            con_better = 0
            num_divs = 0
            
            def get_adjusted_prob(other):
                return self.get_text_prob(other) + 3*sdutils.cosine_sim(features,other)/4
            
            eps_scalar = torch.tensor([1e-33], device=features.device)
            while self.get_text_prob(out_features)[0] <= target_prob and cosine_change < max_diff and num_divs < max_divs:
                dx = -torch.autograd.functional.jacobian(get_adjusted_prob, out_features.float(), create_graph=True).reshape(out_features.shape).float()
                dx_mag = dx.norm(dim=-1, keepdim=True)
                if dx_mag < eps_scalar:
                    dx /= eps_scalar
                
                dx_shift = -per_step * dx
                
                prev_out_features = out_features.clone()
                out_features += dx_shift
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                mid_prob = self.get_text_prob(out_features)
                
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))

                if mid_prob > best_out_score and cosine_change < max_diff:
                    if mid_prob - best_out_score < 1e-5:
                        per_step /= alpha
                    best_out_score = mid_prob
                    best_out_features = out_features.clone()
                
                if cosine_change > max_diff:
                    con_better = 0
                    out_features = prev_out_features
                    cosine_change = 0
                    per_step *= alpha
                    num_divs += 1
                else: 
                    con_better += 1
                    if con_better > 4:
                        per_step /= alpha
                        
           
            if verbose:
                cosine_change = abs(1.0 - (features.unsqueeze(0) @ best_out_features.T))
                print("End Prob", best_out_score)
                print("Cosine Diff of New Features", cosine_change)

            return best_out_features.squeeze(0)
