import typing

import ai.stabledisco.constants as sdconsts
import ai.stabledisco.utils as sdutils
import open_clip
import PIL
import torch
from ai.stabledisco.encodedtext import EncodedText


class ClipModel(torch.nn.Module):
    _TRUNCATE_LEN = 74

    def __init__(self, clip_model: torch.nn.Module, preprocess: torch.nn.Module, name: str) -> None:
        super().__init__()
        self._name = name
        self._model = clip_model
        self._preprocess = preprocess
        self._embedded_end = None

    def forward(self, x_input):
        return self._model(x_input)

    def get_top_percentile(self, image_features, encoded_text, perc, verbosity=1):
        top_k = int(len(encoded_text) / 100 * perc)
        return self.rank_similarity(image_features, encoded_text, top_count=top_k, verbosity=verbosity)

    def get_n_most_disim(self, encoded_baseline, encoded_test_array, n, verbosity=1):
        if type(encoded_baseline) is not list:
            encoded_baseline = [encoded_baseline]
        most_disim = encoded_baseline

        text_features = self.get_features(encoded_test_array, verbosity=verbosity)
        with torch.no_grad():
            for _ in range(min(n, len(encoded_test_array))):
                most_disim_features = self.get_features(most_disim, verbosity=0)
                similarity = torch.zeros((1, len(encoded_test_array)), device=text_features.device)
                for i in range(most_disim_features.shape[0]):
                    similarity += most_disim_features[i].unsqueeze(0) @ text_features.T
                similarity /= most_disim_features.shape[0]

                _, top_labels = similarity.float().cpu().topk(1, dim=-1, largest=False)
                encoded_pos = top_labels[0][0]
                most_disim.append(encoded_test_array[encoded_pos])
                encoded_test_array = torch.cat(
                    tuple((encoded_test_array[:encoded_pos], encoded_test_array[encoded_pos + 1 :])), axis=0
                )
                text_features = torch.cat((text_features[:encoded_pos], text_features[encoded_pos + 1 :]))

            return most_disim[1:]

    def cosine_similarity(self, baseline_features, encoded_test_array, end_idx=-1, verbosity=1):

        with torch.no_grad():
            if isinstance(encoded_test_array[0], torch.Tensor):
                if encoded_test_array[0].is_floating_point():
                    text_features = encoded_test_array
                else:
                    text_features = self.features_from_tokens(encoded_test_array, end_idx=end_idx, verbosity=verbosity)
            else:
                text_features = self.features_from_encoded(encoded_test_array, verbosity=verbosity)

            baseline_features = baseline_features
            
            if len(text_features.shape) == 3:
                text_features = text_features.squeeze(0)

            baseline_features /= baseline_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = torch.zeros(
                (1, len(encoded_test_array)),
                dtype=torch.half,
                device=baseline_features.device,
            )
            for i in range(baseline_features.shape[0]):
                similarity += baseline_features[i].unsqueeze(0) @ text_features.T
            similarity /= baseline_features.shape[0]
            return similarity.squeeze(0)

    def max_cosine_similarity(self, baseline_features, encoded_test_array, verbosity=1):
        with torch.no_grad():
            if isinstance(encoded_test_array[0], torch.Tensor):
                text_features = self.features_from_tokens(encoded_test_array, verbosity=verbosity)
            else:
                text_features = self.features_from_encoded(encoded_test_array, verbosity=verbosity)

            baseline_features = baseline_features.half()
            baseline_features /= baseline_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = torch.zeros(
                (1, len(encoded_test_array)),
                dtype=torch.half,
                device=baseline_features.device,
            )
            max_sim = torch.zeros(text_features.shape[0])
            for i in range(baseline_features.shape[0]):
                similarity = (baseline_features[i].unsqueeze(0) @ text_features.T).squeeze(0)
                for j in range(text_features.shape[0]):
                    max_sim[j] = torch.maximum(max_sim[j], similarity[j])
            return max_sim

    def rank_similarity(
        self,
        baseline_features: torch.Tensor,
        encoded_test_array: typing.Union[torch.tensor, typing.List[EncodedText]],
        top_count=None,
        largest=True,
        end_idx=-1,
        verbosity=1,
    ):
        if not top_count:
            top_count = len(encoded_test_array)

        top_count = min(top_count, len(encoded_test_array))
        similarity = self.cosine_similarity(
            baseline_features, encoded_test_array, end_idx=end_idx, verbosity=verbosity
        ).unsqueeze(0)

        top_probs, top_labels = similarity.float().cpu().topk(top_count, dim=-1, largest=largest)
        top_words = [encoded_test_array[top_labels[0][i].numpy()] for i in range(top_count)]
        top_probs = [top_probs[0][i].numpy() for i in range(top_count)]
        return top_words, top_probs

    def get_encoded_sim(self, text_a: str, text_b: str):
        features_a = self.features_from_encoded(text_a, verbosity=0)
        encoded_b = text_b
        return self.rank_similarity(features_a, [encoded_b], verbosity=0)[1][0]

    def get_text_sim(self, text_a: str, text_b: str):
        features_a = self.features_from_encoded(EncodedText.from_text(text_a), verbosity=0)
        encoded_b = EncodedText.from_text(text_b)
        return self.rank_similarity(features_a, [encoded_b], verbosity=0)[1][0]

    # TODO: Merge to one function checking type
    def features_from_text(self, text, step_size=15000, verbosity=1, cuda=True):
        if type(text) is not list:
            text = [text]
        encoded_text = EncodedText.from_text_list(text)
        return self.features_from_encoded(encoded_text, step_size=step_size, verbosity=verbosity, cuda=cuda)

    def get_features(self, encoded, verbosity=1, normalize=True, cuda=True):
        if type(encoded) is str:
            encoded = open_clip.tokenize(encoded).reshape(-1, sdconsts.prompt_token_len)
            if cuda:
                encoded.cuda()
        elif type(encoded) is not list and type(encoded) is not torch.Tensor:
            encoded = [encoded]

        if type(encoded[0]) == torch.Tensor:
            return self.features_from_tokens(encoded, normalize=normalize, verbosity=verbosity, cuda=cuda)
        
        return self.features_from_encoded(encoded, normalize=normalize, verbosity=verbosity, cuda=cuda)

    def features_from_encoded(self, encoded_text, step_size=15000, verbosity=1, normalize=True, cuda=True):
        if type(encoded_text) is not list:
            encoded_text = [encoded_text]
        tokens = torch.stack([encoded.get_tokens() for encoded in encoded_text])
        return self.features_from_tokens(
            tokens, step_size=step_size, normalize=normalize, verbosity=verbosity, cuda=cuda
        )

    def features_from_tokens(self, tokens, step_size=15000, verbosity=1, normalize=True, cuda=True, end_idx=-1):
        if type(tokens) is list:
            tokens = torch.stack(tuple(tokens))
        tokens = sdutils.change_rev(tokens, False).view(tokens.shape)

        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        if end_idx == -1:
            end_idx = tokens.argmax(dim=-1).reshape(-1)
        
        def local_encode_func(tokens):
            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)
            elif len(tokens.shape) == 3 and tokens.size(0) == 1:
                tokens = tokens.squeeze(0)

            features = self._model.encode_text(tokens.cuda())
            
            if len(features.shape) == 3: 
                features = features[torch.arange(tokens.shape[0]), end_idx]
          
            if normalize:
                features = sdutils.norm_t(features)
                
            return features


        verbosity = 0
        with torch.no_grad():
            if len(tokens) <= step_size:
                text_features = local_encode_func(torch.stack(tuple(tokens)))
                if not cuda:
                    text_features = text_features.cpu()
                return text_features

            text_features = torch.tensor([], dtype=torch.half).cuda()

            if verbosity > 0:
                print(f"Encoding {len(tokens)} entries")

            start = torch.IntTensor([0]).cuda()
            step = torch.IntTensor([step_size]).cuda()
            end = torch.IntTensor().cuda()

            for idx in range(0, len(tokens), step_size):
                if verbosity > 1:
                    print(f"Finished {idx} of {len(tokens)} entries")
                    print(f"Encoding {start} to {end}")
                torch.add(start, step, out=end)

                new_features = local_encode_func(tokens[start:end])
                text_features = torch.cat((text_features, new_features))
                torch.add(start, step, out=start)
            if verbosity > 0:
                print(f"Finished {len(tokens)} of {len(tokens)} entries")

            if normalize:
                text_features /= text_features.norm(dim=-1, keepdim=True)

        if not cuda:
            text_features = text_features.cpu()

        return text_features

    def _features_from_uniform_end_tokens(self, tokens, end_idx):
        # The operation is much faster if all share the same end idx
        model = self._model.model
        x = model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        x = x + model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self._model.ln_final(x)
        x = x.permute(1, 0, 2)[end_idx]
        x = x
        

        # x = x[idxs, arg_maxs]
        x = x @ self._model.text_projection

        return x

    # TODO: Combine the two below
    def encode_images_features(self, imgs: typing.List[PIL.Image.Image], normalize=False) -> torch.Tensor:
        with torch.no_grad():
            images = self._preprocess(imgs).cuda()
            image_features = self._model.encode_image(images)
            if normalize:
                image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_image_features(self, img: PIL.Image.Image, normalize=False) -> torch.Tensor:
        images = self._preprocess(img).unsqueeze(0).cuda()
        with torch.no_grad():
            image_features = self._model.encode_image(images)
        if normalize:
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def partial_image_encode(self, img):
        x = self._preprocess(img).cuda().to(self._model.dtype)
        x = self._model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self._model.visual.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self._model.visual.positional_embedding.to(x.dtype)
        x = self._model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self._model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x

    def embed_partial_tokens(self, tokens):
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        return self._model.token_embedding(tokens).type(self._model.dtype)  # [batch_size, n_ctx, d_model]

    def add_to_embedded(self, embeded_tokens, new_tokens):
        if len(embeded_tokens.shape) == 1:
            embeded_tokens = embeded_tokens.unsqueeze(0)

        if len(new_tokens.shape) == 1:
            new_tokens = new_tokens.unsqueeze(0)
        x = self._model.token_embedding(new_tokens).type(self._model.dtype)  # [batch_size, n_ctx, d_model]
        return torch.cat(embeded_tokens, x, axis=1)

    def encode_embeded(self, tokens, embeded_tokens):
        while len(tokens.shape) < 2:
            tokens = tokens.unsqueeze(0)

        while len(embeded_tokens.shape) < 3:
            embeded_tokens = embeded_tokens.unsqueeze(0)

        rem_len = sdconsts.prompt_token_len - embeded_tokens.size(1)
        ends = self._get_embedded_end(rem_len)

        x = torch.cat((embeded_tokens, ends), axis=1)

        x = x + self._model.positional_embedding.type(self._model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self._model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self._model.ln_final(x).type(self._model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        if tokens.size() != sdconsts.prompt_token_len:
            end_idxs = tokens.size(1)
        else:
            end_idxs = tokens.argmax(dim=-1)

        x = x[torch.arange(x.shape[0]), end_idxs] @ self._model.text_projection

        return x

    def _get_embedded_end(self, suffix_len):
        # TODO: remove after testing
        self._embedded_end = None
        if self._embedded_end is None:
            # Put the end token embedding in front, remove the start
            empty_tokens = open_clip.tokenize("").cuda()[0]
            self._embedded_end = self.embed_partial_tokens(empty_tokens)
            self._embedded_end[0, 0] = self._embedded_end[0, 1]
            self._embedded_end[0, 1] = self._embedded_end[0, -1]

        return self._embedded_end[:, :suffix_len, :]

    def encode_text(self, text, truncate=False):
        max_len = 77
        if isinstance(text, str):
            tokens = open_clip.tokenize(text, truncate=truncate).cuda()
        elif isinstance(text, torch.Tensor):
            tokens = text
        elif isinstance(text, EncodedText):
            tokens = text.get_tokens()
        with torch.no_grad():
            tokens = tokens.reshape(-1, max_len)
            encoded_text = self._model.encode_text(tokens)[:, -1]

        return encoded_text / encoded_text.norm(dim=-1, keepdim=True)

    def get_name(self):
        return self._name

    def get_model(self) -> torch.nn.Module:
        return self._model

    def get_preprocess(self) -> torch.nn.Module:
        return self._preprocess
