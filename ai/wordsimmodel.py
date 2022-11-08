import random
from functools import lru_cache
from typing import List, Tuple

import gensim
import numpy as np
import spacy

nlp = spacy.load("en_core_web_md")


class WordSimModel:
    def __init__(self, corpus_strs):
        class MJCorpus:
            def __init__(self, sentences):
                self._sentences = sentences

            def __iter__(self):
                for sentence in self._sentences:
                    yield WordSimModel.process_sentence(sentence)

        self._gem_model = gensim.models.Word2Vec(MJCorpus(corpus_strs), window=3, vector_size=300)

    def replace_words(
        self,
        orig_sentence,
        rnn_model,
        *args,
        max_percent_replace=0.5,
        rnn_passes=3,
        sub_passes=30,
        verbose=False,
        cutoff_pairs: List[Tuple[str, int]] = None,
        **kwargs,
    ):
        if not cutoff_pairs:
            cutoff_pairs = []
        protected = {word for word, val in cutoff_pairs if val > 0.99}

        curr_sentence = orig_sentence
        sorted_sentences = []
        predictions = []
        for iter_num in range(rnn_passes):
            if verbose:
                print(f"rnn pass {iter_num}")

            sentences_set = {curr_sentence}
            for _ in range(sub_passes):
                sentence = curr_sentence
                sentences_set.add(sentence)
                to_replace_ceil = int(max_percent_replace * len(curr_sentence))
                to_replace_cnt = random.randint(1, to_replace_ceil)
                split_sentence = sentence.split(" ")
                all_idx = [idx for idx in range(len(split_sentence)) if split_sentence[idx] not in protected]
                np.random.shuffle(all_idx)
                to_replace = all_idx[:to_replace_cnt]
                replacement_words = []
                for replace_idx in to_replace:
                    try:
                        banned = []
                        if replace_idx > 0:
                            banned.append(split_sentence[replace_idx - 1])
                        if replace_idx < len(split_sentence) - 1:
                            banned.append(split_sentence[replace_idx + 1])

                        replacement = self.get_weighted_random_replacement_match(
                            split_sentence, replace_idx, banned=banned
                        )
                        replacement_words.append((replace_idx, replacement))
                    except KeyError:
                        pass

                for _ in range(3):
                    random.shuffle(replacement_words)
                    split_sentence = sentence.split(" ")
                    for idx, word in replacement_words:
                        split_sentence[idx] = word
                        sentences_set.add(" ".join(split_sentence))

            sorted_sentences, predictions = rnn_model.sort_command_strs(list(sentences_set))
            curr_sentence, prediction = sorted_sentences[0], predictions[0]

            if verbose:
                print(f"Prediction: {prediction}")
                print(f"Before: {orig_sentence}")
                print(f"After: {curr_sentence}")

        return sorted_sentences, predictions.flatten()

    def add_and_drop_word(
        self,
        sentence,
        rnn_model,
        *args,
        num_outer_runs=1,
        drop_chance=0.25,
        max_drop=20,
        max_drop_percent=0.2,
        add_chance=0.75,
        max_add=30,
        add_match_drop=False,
        cutoff_pairs: List[Tuple[str, int]] = None,
        **kwargs,
    ):

        if not cutoff_pairs:
            cutoff_pairs = []
        curr_sentence = sentence
        protected = [word for word, val in cutoff_pairs if val == 1.0]
        sentences = [sentence]
        predictions = [0]
        for _ in range(num_outer_runs):
            sentences = [curr_sentence]
            num_to_drop = 0
            if random.random() < drop_chance:
                drop_ceil = min(max_drop, int(len(sentence) * max_drop_percent))
                num_to_drop = random.randint(0, drop_ceil)

            num_to_add = 0
            if random.random() < add_chance:
                add_ceil = max_add
                if add_match_drop:
                    add_ceil = max(max_add, num_to_drop)
                num_to_add = random.randint(0, add_ceil)

            # TODO
            try:
                sentences += self.drop_words(curr_sentence, num_to_drop, protected_words=protected)
            except Exception:
                pass
            try:
                sentences += self.add_words(curr_sentence, num_to_add, *args, **kwargs)
            except Exception:
                pass

            sentences, predictions = rnn_model.sort_command_strs(list(set(sentences)))
            curr_sentence = sentences[0]

        return sentences, predictions.flatten()

    def add_words(self, sentence, num_to_add, *args, **kwargs):
        split_sentence = sentence.split(" ")
        changed_sentences = []
        for _ in range(num_to_add):

            insertion_idx = random.randint(0, len(split_sentence) - 1)

            end_insertion_check = insertion_idx - 1 if insertion_idx != 0 else len(split_sentence) - 1
            variants = self.get_words_to_variants(sentence, *args, **kwargs)
            while split_sentence[insertion_idx] not in variants and insertion_idx != end_insertion_check:
                insertion_idx = (insertion_idx + 1) % len(split_sentence)

            sub_words, sub_probs = list(zip(*variants[split_sentence[insertion_idx]]))
            sub_probs /= np.sum(sub_probs)
            replacement = np.random.choice(sub_words, p=sub_probs)
            split_sentence = split_sentence[:insertion_idx] + [replacement] + split_sentence[insertion_idx:]
            changed_sentences.append(" ".join(split_sentence))

        return changed_sentences

    # TODO(Benjamin Sergent): Weight which words to drop by estimated importance
    def drop_words(self, sentence, num_to_drop, protected_words=None):
        if not protected_words:
            protected_words = []
        changed_sentences = []
        split_sentence = sentence.split(" ")
        # TODO: Change to to_drop to keep protected without reducing drops
        all_idx = np.arange(len(split_sentence)).tolist()
        np.random.shuffle(all_idx)

        parts = get_parts(sentence)
        for idx, part in parts:
            if "root" in part:
                protected_words.append(split_sentence[idx])
        protected_idx = []
        for idx, word in enumerate(sentence.split()):
            if word in protected_words:
                protected_idx.append(protected_idx)

        for num_remove in range(1, num_to_drop + 1):
            to_keep = all_idx[: len(split_sentence) - num_remove] + protected_idx
            changed_sentence = " ".join([split_sentence[idx] for idx in to_keep])
            changed_sentences.append(changed_sentence)

        return changed_sentences

    # TODO(Benjamin Sergent): Function to estimate importance of words in a sentence

    def predict_output_word(self, *args, **kwargs):
        return self._gem_model.predict_output_word(*args, **kwargs)

    def get_weighted_random_replacement(self, word, banned: List[str] = None, *args, **kwargs):
        if banned is None:
            banned = []
        variants = [word for word in self.most_similar(word, *args, **kwargs) if word not in banned]
        sub_words, sub_probs = list(zip(*variants))
        sub_probs /= np.sum(sub_probs)
        return np.random.choice(sub_words, p=sub_probs)

    def get_weighted_random_replacement_match(self, sentence_words, idx, banned: List[str] = None):
        # TODO: Change sentencewords to imply tuples
        sub_words, sub_probs = self.get_weighted_random_replacement_options(tuple(sentence_words), idx, tuple(banned))
        return np.random.choice(sub_words)

    @lru_cache(maxsize=2048)
    def get_weighted_random_replacement_options(self, sentence_words, idx, banned: Tuple[str] = None):
        if banned is None:
            banned = []
        sentence_words = list(sentence_words)
        sentence = " ".join(sentence_words)
        orig_parts = get_parts(sentence)
        word = sentence_words[idx]
        word_vars = self.most_similar(word)
        variants = [(replacement, sim) for replacement, sim in word_vars if replacement not in banned]

        sub_words = []
        sub_probs = []
        for replacement, prob in variants:
            sentence_words[idx] = replacement
            replacement_sentence = " ".join(sentence_words)
            rep_parts = get_parts(replacement_sentence)

            if orig_parts == rep_parts:
                sub_words.append(replacement)
                sub_probs.append(prob)

        if len(sub_probs) == 0:
            return [word], [1]

        sub_probs /= np.sum(sub_probs)
        return sub_words, sub_probs

    def get_words_to_variants(
        self,
        sentence,
        *args,
        base_threshold=0.7,
        cutoff_pairs: List[Tuple[str, float]] = None,
        **kwargs,
    ):
        if cutoff_pairs is None:
            cutoff_pairs = []

        sentence = self.process_sentence(sentence)
        words_to_varients = {}
        for word in sentence:
            try:
                threshold = base_threshold
                for words, cuttoff in cutoff_pairs:
                    if word in words:
                        threshold = cuttoff
                        break

                subs = self.most_similar(word, *args, base_threshold=threshold, **kwargs)
                if subs:
                    words_to_varients[word] = subs
            except Exception:
                pass
        return words_to_varients

    @lru_cache(maxsize=2048)
    def most_similar(self, word, *args, base_threshold=0.0, **kwargs):
        # TODO: Cuttofs
        return [
            (sub, sim) for sub, sim in self._gem_model.wv.most_similar(word, *args, **kwargs) if sim > base_threshold
        ]

    def similarity(self, word_a, word_b):
        try:
            sim = 1 - self._gem_model.wv.distance(word_a, word_b)
        except Exception:
            sim = 0

        return sim

    @classmethod
    def load(cls, gem_model_file):
        self = cls.__new__(cls)
        self._gem_model = gensim.models.Word2Vec.load(gem_model_file)
        return self

    def save(self, gem_model_file):
        self._gem_model.save(gem_model_file)


@lru_cache(maxsize=2048)
def get_parts(txt, add_words=False):
    txt = txt.replace(r"[\[#+()-\!]]", " , ")
    doc = nlp(txt)
    ret = []
    ret_idx = 0
    for sentence in doc.sents:
        for word in sentence:
            new_entry = ""
            if add_words:
                new_entry = f"{word}_"
            ret.append(f"{new_entry}{word.pos_}_{word.dep_}".lower())
            ret_idx += 1

    root_idx = 0
    for chunk in doc.noun_chunks:
        for tok in chunk:
            if tok == chunk.root:
                root_idx += 1
                ret[tok.i] += "_root"

        root_idx += 1
    return ret
