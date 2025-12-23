from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from nltk.corpus import wordnet

# import wn
import os
import dill
from collections import defaultdict
from tqdm import tqdm

from src.utils.wsd import pos_map, expand_raganato_path


class SenseInventory(ABC):
    @abstractmethod
    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        pass

    @abstractmethod
    def get_definition(self, sense: str, instance_id: str = None) -> str:
        pass

    @abstractmethod
    def get_id_from_sense(self, sense: str) -> int:
        pass

    @abstractmethod
    def get_sense_from_id(self, id: int) -> str:
        pass

    @abstractmethod
    def get_num_senses(self) -> int:
        pass

    @abstractmethod
    def get_synid(self, synset):
        pass

    def get_gloss_count(gloss):
        pass


# WORDNET


class WordNetSenseInventory(SenseInventory):
    def __init__(self, wn_candidates_path: str, corpora_names: List[str]):
        self.corpora_names = corpora_names
        self.lemmapos2senses = dict()
        self.sense2id = dict()
        self.id2sense = dict()
        self.synid2examples = None
        self.gloss_counts = defaultdict(int)
        self.gloss2sense = dict()
        self._load_lemmapos2senses(wn_candidates_path)
        self._load_sense2id_and_id2sense(wn_candidates_path)
        self._load_synid2examples()
        self._load_gloss_counts()
        self._load_gloss2sense()

    def _load_gloss2sense(self):
        gloss2sense_path = "gloss2sense.pkl"
        if os.path.exists(gloss2sense_path):
            with open(gloss2sense_path, "rb") as f:
                self.gloss2sense = dill.load(f)
        else:
            with open(gloss2sense_path, "wb") as f:
                for sense, _ in tqdm(self.sense2id.items(), desc="create gloss2sense"):
                    synset = wordnet.lemma_from_key(sense).synset()
                    self.gloss2sense[synset.definition()] = sense
                dill.dump(self.gloss2sense, f)

    def _load_gloss_counts(self):
        gloss_counts_path = "gloss_counts.pkl"
        if os.path.exists(gloss_counts_path):
            with open(gloss_counts_path, "rb") as f:
                self.gloss_counts = dill.load(f)
        else:
            for corpora_name in self.corpora_names:
                _, corpora_gold = expand_raganato_path(corpora_name)
                with open(corpora_gold, "r") as f:
                    for line in tqdm(f, desc=corpora_gold):
                        inst, *labels = line.strip().split()
                        for label in labels:
                            synset = wordnet.lemma_from_key(label).synset()
                            self.gloss_counts[synset.definition()] += 1
            with open(gloss_counts_path, "wb") as f:
                dill.dump(self.gloss_counts, f)

    def _load_lemmapos2senses(self, wn_candidates_path: str):
        with open(wn_candidates_path) as f:
            for line in f:
                lemma, pos, *senses = line.strip().split("\t")
                self.lemmapos2senses[(lemma, pos)] = senses

    def _load_sense2id_and_id2sense(self, wn_candidates_path: str):
        ind = 0
        with open(wn_candidates_path) as f:
            for _, line in enumerate(f):
                lemma, pos, *senses = line.strip().split("\t")
                for sense in senses:
                    if sense not in self.sense2id.keys():
                        self.sense2id[sense] = ind
                        ind += 1
        self.id2sense = {v: k for k, v in self.sense2id.items()}

    def _load_synid2examples(self):
        synid2example_path = "examples/b_synid2example.pkl"
        if os.path.exists(synid2example_path):
            with open(synid2example_path, "rb") as f:
                self.synid2examples = dill.load(f)

    def get_gloss_count(self, gloss: str):
        if gloss in self.gloss_counts:
            return self.gloss_counts[gloss]
        else:
            return 0

    def get_sense_from_gloss(self, gloss: str):
        return self.gloss2sense[gloss]

    def get_possible_senses(self, lemma: str, pos: str) -> List[str]:
        return self.lemmapos2senses.get((lemma, pos), [])

    # @lru_cache(maxsize=None)
    def get_definition(self, sense: str, instance_id: str = None):
        synset = wordnet.lemma_from_key(sense).synset()
        tgt_definition = synset.definition()

        rel_infos = []
        synset = wordnet.lemma_from_key(sense).synset()
        synid = self.get_synid(synset)
        if self.synid2examples is not None and synid in self.synid2examples:
            examples = self.synid2examples[synid]
            if instance_id is not None:
                examples = [item[0] for item in examples if item[1] != instance_id]
                if len(examples) == 0:
                    for example in synset.examples()[:1]:
                        rel_infos.append(example)
                else:
                    rel_infos.append(examples[0])
        else:
            for example in synset.examples()[:1]:
                rel_infos.append(example)

        return tgt_definition, rel_infos

    def get_sense_from_id(self, ind: int) -> str:
        return self.id2sense[ind]

    def get_id_from_sense(self, sense: str) -> int:
        return self.sense2id[sense]

    def get_num_senses(self) -> int:
        return len(self.sense2id)

    def get_synid(self, synset):
        pos = synset.pos()
        if pos == "s":
            pos = "a"
        return f"{pos}{synset.offset()}"
