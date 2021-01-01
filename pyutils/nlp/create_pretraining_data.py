# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
import json
import datasets
import logging
import collections
import random
import re
from src.config.config_pretrained import *
from src.pretrain.roberta import tokenization
import jieba
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

# dynamic repeat
# max 512
# 4 tokens

# TODO(cmrc2018): BibTeX citation
_CITATION = """\
@inproceedings{cui-emnlp2019-cmrc2018,
    title = {A Span-Extraction Dataset for {C}hinese Machine Reading Comprehension},
    author = {Cui, Yiming  and
      Liu, Ting  and
      Che, Wanxiang  and
      Xiao, Li  and
      Chen, Zhipeng  and
      Ma, Wentao  and
      Wang, Shijin  and
      Hu, Guoping},
    booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
    month = {nov},
    year = {2019},
    address = {Hong Kong, China},
    publisher = {Association for Computational Linguistics},
    url = {https://www.aclweb.org/anthology/D19-1600},
    doi = {10.18653/v1/D19-1600},
    pages = {5886--5891}}
"""

# TODO(cmrc2018):
_DESCRIPTION = """\
A Span-Extraction dataset for Chinese machine reading comprehension to add language
diversities in this area. The dataset is composed by near 20,000 real questions annotated
on Wikipedia paragraphs by human experts. We also annotated a challenge set which
contains the questions that need comprehensive understanding and multi-sentence
inference throughout the context.
"""


class Cmrc2018(datasets.GeneratorBasedBuilder):
    """TODO(cmrc2018): Short description of my dataset."""

    # TODO(cmrc2018): Set up version.
    VERSION = datasets.Version("0.1.0")

    def _info(self):
        # TODO(cmrc2018): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(cmrc2018): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = {
            "train": self.config.data_files["train"],
            "dev": self.config.data_files["dev"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(cmrc2018): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        yield id_, {
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }


class LmModelDatasetProcessor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
        self.rng = random.Random(self.args.random_seed)
        self.vocab_words = list(self.tokenizer.vocab.keys())
        self.masked_lm_instance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
        pass

    def _truncate_sequence(self, sequence):
        sequence = self.tokenizer.tokenize(sequence)
        if len(sequence) > self.args.max_seq_length - 2:
            sequence = sequence[:self.args.max_seq_length - 2]
        return sequence

    def _add_special_prefix(self, example):
        segment = self._truncate_sequence(example["context"])
        seq_cws = jieba.lcut(example["context"])
        seq_cws_dict = {x: 1 for x in seq_cws}
        new_segment, i = [], 0
        while i < len(segment):
            # adding English word directly.
            if not re.findall("[\u4E00-\u9FA5]", segment[i]):
                new_segment.append(segment[i])
                i += 1
                continue
            has_add = False
            for length in range(3, 0, -1):
                if i + length > len(segment):
                    continue
                word = "".join(segment[i:i + length])
                if word in seq_cws_dict:
                    new_segment.append(segment[i])
                    for index in range(1, length):
                        new_segment.append("##" + segment[i + index])
                    i += length
                    has_add = True
                    break
            if not has_add:
                new_segment.append(segment[i])
                i += 1
        return {"new_segment": new_segment}

    def _create_masked_lm_predictions(self, example):
        """Creates the predictions for the masked LM objective."""
        tokens = example["new_segment"]
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (args.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        self.rng.shuffle(cand_indexes)  # shuffle for 'for index_set in cand_indexes'
        output_tokens = [t[2:] if re.findall('##[\u4E00-\u9FA5]', t) else t for t in tokens]  # remove '##' for chinese word
        label_tokens = output_tokens.copy()
        label_ids = [-100] * len(label_tokens)
        num_to_predict = min(self.args.max_predictions_per_seq, max(1, int(round(len(tokens) * self.args.masked_lm_prob))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_token = None
                # 80% of the time, replace with [MASK]
                if self.rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if self.rng.random() < 0.5:
                        masked_token = tokens[index][2:] if re.findall('##[\u4E00-\u9FA5]', tokens[index]) else tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = self.vocab_words[self.rng.randint(0, len(self.vocab_words) - 1)]
                output_tokens[index] = masked_token
                assert label_tokens[index] == tokens[index][2:] if re.findall('##[\u4E00-\u9FA5]', tokens[index]) else tokens[index]
                label_ids[index] = self.tokenizer.convert_tokens_to_ids([label_tokens[index]])[0]
                masked_lms.append(self.masked_lm_instance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)
        # logging.info('%s' % (tokens))
        # logging.info('%s' % (output_tokens))
        return {"input_tokens": output_tokens,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_labels": masked_lm_labels,
                "label_tokens": label_tokens,
                "label_ids": label_ids}

    def _encode_plus(self, example):
        input_tokens, label_ids = example["input_tokens"], example["label_ids"]
        assert len(input_tokens) <= self.args.max_seq_length - 2
        assert len(label_ids) <= self.args.max_seq_length - 2
        input_tokens = ["[CLS]"] + input_tokens + ["[SEP]"]
        label_ids = [-100] + label_ids + [-100]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        input_segment = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= self.args.max_seq_length

        if len(input_ids) < self.args.max_seq_length:
            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids = input_ids + padding
            input_mask = input_mask + padding
            input_segment = input_segment + padding
            label_ids = label_ids + [-100] * len(padding)
        assert len(input_ids) == self.args.max_seq_length
        assert len(input_mask) == self.args.max_seq_length
        assert len(input_segment) == self.args.max_seq_length
        assert len(label_ids) == self.args.max_seq_length

        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(example["masked_lm_labels"])
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        if len(masked_lm_positions) < self.args.max_predictions_per_seq:
            padding_num = self.args.max_predictions_per_seq - len(masked_lm_positions)
            masked_lm_positions = masked_lm_positions + [0] * padding_num
            masked_lm_ids = masked_lm_ids + [0] * padding_num
            masked_lm_weights = masked_lm_weights + [0.0] * padding_num
        assert len(masked_lm_positions) == self.args.max_predictions_per_seq
        assert len(masked_lm_ids) == self.args.max_predictions_per_seq
        assert len(masked_lm_weights) == self.args.max_predictions_per_seq
        assert len(list(filter(lambda x: x != -100, label_ids))) == len(list(filter(lambda x: x != 0, masked_lm_ids)))

        return {"input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": input_segment,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights": masked_lm_weights,
                "label_ids": label_ids}

    def create_training_instances(self, dataset):
        print("_add_special_prefix")
        dataset = dataset.map(self._add_special_prefix)

        print("_create_masked_lm_predictions")
        dataset_train_list, dataset_dev_list = [], []
        for _ in range(self.args.dupe_factor):
            dataset_tmp = dataset.map(self._create_masked_lm_predictions).shuffle(seeds={"train": self.args.random_seed, "validation": self.args.random_seed})
            dataset_train_list.append(dataset_tmp["train"])
            dataset_dev_list.append(dataset_tmp["validation"])
        dataset_train_origin_size, dataset_dev_origin_size = len(dataset["train"]), len(dataset["validation"])
        dataset["train"] = datasets.concatenate_datasets(dataset_train_list)
        dataset["validation"] = datasets.concatenate_datasets(dataset_dev_list)
        assert len(dataset["train"]) == dataset_train_origin_size * self.args.dupe_factor
        assert len(dataset["validation"]) == dataset_dev_origin_size * self.args.dupe_factor

        print("_encode_plus")
        dataset = dataset.map(self._encode_plus)
        return dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, default=INPUT_ROBERTA_CMRC2018_TRAIN_FILE,
                        help="Input train file.")
    parser.add_argument("--dev_file", type=str, default=INPUT_ROBERTA_CMRC2018_DEV_FILE,
                        help="Input dev file.")
    parser.add_argument("--output_file", type=str, default=OUTPUT_ROBERTA_FILE,
                        help="Output TF example file (or comma-separated list of files).")
    parser.add_argument("--vocab_file", type=str, default=VOCAB_ROBERTA_FILE,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--do_whole_word_mask", type=bool, default=True,
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length.")
    parser.add_argument("--max_predictions_per_seq", type=int, default=40,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for data generation.")
    parser.add_argument("--dupe_factor", type=int, default=10,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--masked_lm_prob", type=float, default=0.10,
                        help="Masked LM probability.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of creating sequences which are shorter than the maximum length.")
    args = parser.parse_args()
    dataset = datasets.load_dataset(ROBERTA_LOADING_SCRIPT,
                                    data_files={"train": args.train_file, "dev": args.dev_file},
                                    cache_dir=DATA_ROBERTA_CACHE)
    processor = LmModelDatasetProcessor(args)
    dataset = processor.create_training_instances(dataset)
    columns = ["input_ids", "attention_mask", "token_type_ids", "label_ids"]
    dataset.set_format(type="torch", columns=columns)
    if not os.path.exists(OUTPUT_ROBERTA_PATH):
        os.makedirs(OUTPUT_ROBERTA_PATH)
    torch.save(dataset, OUTPUT_ROBERTA_FILE)
    pass
