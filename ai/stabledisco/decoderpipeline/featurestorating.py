import ai.torchmodules as torchmodules
import ai.torchmodules.layers as torchlayers
import ai.torchmodules.utils as torchutils
import torch
import torch.nn as nn
from ai.stabledisco.decoderpipeline.lowerfeaturelayers import \
    LowerFeatureLayers


class FeaturesToRatingModel(torchmodules.BaseModel):
    def __init__(self, clip_model, freeze_lower=True, device=None):
        super().__init__("FeaturesToRatingV2")
        # Input = (-1, 77, num_features)
        # Output = (-1, 77, vocab_size)
        # Loss = diffable encoding
        print(clip_model.dtype)

        self._dtype = clip_model.dtype
        if device is None:
            device = torchutils.get_default_device()
        self._device = device

        self._clip_model = clip_model
        for param in self._clip_model.parameters():
            param.requires_grad = False

        clip_state_dict = self._clip_model.state_dict()
        self._transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        self._transformer_layers = len(
            set(
                k.split(".")[2]
                for k in clip_state_dict
                if k.startswith("transformer.resblocks")
            )
        )

        self._feature_expander = LowerFeatureLayers(dropout=0.05)

        if freeze_lower:
            self._feature_expander.freeze()
        else:
            self._feature_expander.unfreeze()
        
        dense_stack_units = [(1, 512),
                             (2, 128),
                             (2, 64),
                             (1, 32)]
                             
        self._dense_stack = torchlayers.DenseStack(
            self._feature_expander.out_features,
            dense_stack_units,
            activation=nn.LeakyReLU,
            dropout=0.1,
        )
        
        self._rating_out = torch.nn.Linear(self._dense_stack.out_features, 1)
        nn.init.xavier_uniform_(self._rating_out.weight)

        self._loss_func = nn.MSELoss()

        base_learning = 5e-4
        self._optimizer = torch.optim.NAdam(
            self.parameters(), base_learning, betas=(0.88, 0.995)
        )

        self._scheduler = torch.optim.lr_scheduler.CyclicLR(
            self._optimizer,
            base_lr=base_learning / 6,
            max_lr=base_learning,
            step_size_up=8000,
            mode="triangular",
            cycle_momentum=False,
        )

    def _calc_batch_loss(self, x_inputs, y_targets):
        with torch.autocast(device_type="cuda"):
            outputs = self(x_inputs)
            return self._loss_func(outputs.view(-1, 1), y_targets.view(-1, 1))

    def forward(self, features):
        features = features / features.norm(dim=-1, keepdim=True)
        x = self._feature_expander(features.float())
        x = self._dense_stack(x)
        return self._rating_out(x)

    def get_rating(self, features):
        return self(features).reshape(1)

    def rank_ratings(self, 
                     tokens,
                     top_count=None,
                     largest=True
    ):
        if top_count is None:
            top_count = len(tokens)

        top_count = min(top_count, len(tokens))
        with torch.no_grad():
            
            features = self._clip_model.encode_text(tokens)
            ratings = self(features).view(-1)
        

            top_ratings, top_labels = (
                ratings.float().cpu().topk(top_count, dim=-1, largest=largest)
            )
            top_words = [
                tokens[top_labels[idx].numpy()] for idx in range(top_count)
            ]
            
            return top_words, top_ratings


    def improve_rating(self, features, target_rating=100, max_diff=0.05, per_step=0.001, verbose=False):
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.view(1, -1)
            features = features / features.norm(dim=-1, keepdim=True)
            before_rating = self.get_rating(features)
            out_features = torch.clone(features)
            cosine_change = 0
            if verbose:
                print("Start rating", before_rating)
            while self.get_rating(out_features)[0] <= target_rating and cosine_change < max_diff:
                dx = torch.autograd.functional.jacobian(self.get_rating, out_features, create_graph=True).reshape(out_features.shape)
                out_features += per_step * dx / dx.norm(dim=-1, keepdim=True)
                out_features = out_features / out_features.norm(dim=-1, keepdim=True)

                cosine_change = abs(1.0 - (features.unsqueeze(0) @ out_features.T))
                

            if verbose:
                print("End Rating", self(out_features))
                print("Cosine Diff of Improve Features", cosine_change)
            return out_features.squeeze(0)
