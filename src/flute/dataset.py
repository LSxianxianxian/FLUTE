from typing import List, Dict
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import torch
from sentence_transformers import SentenceTransformer, util  # 用于计算句子相似度
import conf

class SimpleTextDataset:
    def __init__(self, file_path: str, tokenizer_name: str, max_len: int, example_sentences: dict, dataset_name: str = None, is_train: bool = False):
        self.data = []
        self.example_sentences = example_sentences or {} # 示例句
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # 加载 tokenizer
        self.max_len = max_len
        self.dataset_name = dataset_name  # 添加 dataset_name 属性
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 轻量级嵌入模型
        self.category_examples = {0: [], 1: [], 2: [], 3: []}  # 存储不同类别的示例句


        # 读取训练数据
        #with open(file_path, "r", encoding="utf-8") as f:
           # for line in f:
              #  sentence, label = line.strip().split("\t")
             #   self.data.append((sentence, int(label)))
        # 读取数据集并分类存储示例句
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sentence, label = line.strip().split('\t')
                label = int(label)
                self.data.append((sentence, label))
                if len(self.category_examples[label]) < 10:  # 仅存储前 10 个示例句
                    self.category_examples[label].append(sentence)

        if is_train:
            self.data = self.balance_data(self.data)

    def balance_data(self, data):
        label_0 = [d for d in data if d[1] == 0]
        label_1 = [d for d in data if d[1] == 1]
        label_2 = [d for d in data if d[1] == 2]
        label_3 = [d for d in data if d[1] == 3]

        random.seed(42)
        label_0_sampled = random.sample(label_0, len(label_1))  # 平衡为与label_1相同数量

        balanced = label_0_sampled + label_1 + label_2 + label_3
        random.shuffle(balanced)
        print(f"✅ Balanced data size: {len(balanced)}")
        return balanced


    def find_most_similar_example(self, sentence, label):
        """
        从该类别的 10 个示例句中选取最相似的一个，确保不会选择目标句子本身，并且示例句属于相同类别
        """
        if len(self.category_examples[label]) == 0:
            return ""

        example_sentences = [s for s in self.category_examples[label] if s != sentence]

        if len(example_sentences) == 0:
            return random.choice(self.category_examples[label])

        sentence_embedding = self.sentence_model.encode(sentence, convert_to_tensor=True)
        example_embeddings = self.sentence_model.encode(example_sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(sentence_embedding, example_embeddings)[0]
        best_match_idx = similarities.argmax().item()

        return example_sentences[best_match_idx]


    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]

        #example_metaphor = self.find_most_similar_example(sentence, 1)
        #example_metaphor_1 = self.find_most_similar_example(sentence, 1)
        #example_metaphor_2 = self.find_most_similar_example(sentence, 1)
        #example_metaphor_3 = random.choice(self.example_sentences[1])
        #example_metonymy = self.find_most_similar_example(sentence, 2)
        #example_simile = self.find_most_similar_example(sentence, 3)

        # 选取最相关的示例句
        #example_literal = self.example_sentences[0][0]  # 取第一个示例
        #example_metaphor = f"Metaphor (figurative, no 'like/as'): {example_metaphor}"
        #example_metaphor = self.example_sentences[1][0]
        #example_metonymy = self.example_sentences[2][0]
        #example_simile = self.example_sentences[3][0]

        formatted_sentence = (
            f"[CLS] {sentence} [SEP] "
            #f"Label 1 (Metaphor) Example: {example_metaphor} [SEP] "
            #f"Label 1 (Metaphor) Example2: {example_metaphor_1} [SEP] "
            #f"Label 1 (Metaphor) Example3: {example_metaphor_2} [SEP] "
            #f"Label 2 (metonymy) Example4: {example_metonymy} [SEP] "
            #f"Label 3 (simile) Example4: {example_simile} [SEP] "
            #f"Label 0 (literal) Example: {example_literal} [SEP]"
        )
        #formatted_sentence = sentence
        #formatted_sentence = (
        #    f"[CLS] Sentence: {sentence} [SEP] "
        #    f"Label Options: 0 = Literal, 1 = Metaphor, 2 = Metonymy, 3 = Simile [SEP] "
         #   f"Examples -> "
         #   f"(0) Literal: {example_literal} [SEP] "
         #   f"(1) Metaphor: {example_metaphor} [SEP] "
         #   f"(2) Metonymy: {example_metonymy} [SEP] "
         #   f"(3) Simile: {example_simile} [SEP] "
         #   f"Please classify the sentence above."
        #)
        #formatted_sentence = (
            #f"[CLS] Sentence: {sentence} [SEP] "
           # f"Compare the sentence above with the following examples: [SEP] "
           # f"Literal: {example_literal} [SEP] "
           # f"Metaphor: {example_metaphor} [SEP] "
           # f"Metonymy: {example_metonymy} [SEP] "
          #  f"Simile: {example_simile} [SEP] "
          #  f"Which category is the sentence most similar to? Choose from 0 (Literal), 1 (Metaphor), 2 (Metonymy), 3 (Simile)."
        #)

        # Tokenize the formatted sentence
        inputs = self.tokenizer(
            formatted_sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return input_ids, attention_mask, label