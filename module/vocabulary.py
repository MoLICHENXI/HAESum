#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tools.logger import *

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
    def __init__(self, vocab_file, max_size):
        
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for w in [PAD_TOKEN, UNKNOWN_TOKEN,  START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        with open(vocab_file, 'r', encoding='utf8') as vocab_f: 
            cnt = 0
            for line in vocab_f:
                cnt += 1
                pieces = line.split("\t")
                w = pieces[0]
                if w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    logger.error('Duplicated word in vocabulary file Line %d : %s' % (cnt, w))
                    continue
                
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                
                self._count += 1
                
                if max_size != 0 and self._count >= max_size:
                    logger.info("[INFO] max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break
        logger.info("[INFO] Finished constructing vocabulary of %i total words. Last word added: %s", self._count, self._id_to_word[self._count-1])

    # word -> id
    def word2id(self, word):
        """
        Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV.
        """
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    # id -> word
    def id2word(self, word_id):
        """
        Returns the word (string) corresponding to an id (integer).
        """
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    # size
    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    # word list
    def word_list(self):
        """Return the word list of the vocabulary"""
        return self._word_to_id.keys()