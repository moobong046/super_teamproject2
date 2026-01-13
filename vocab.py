import json
import pickle
import re
import os
import nltk
from collections import Counter
from pathlib import Path

# 경로 설정
CAPTION_DIR = Path("./data/captions")
TRAIN_JSON = CAPTION_DIR / "train_3.json"  # 오직 train 데이터만 사용
SAVE_PATH = "vocab_3.pkl"

NLTK_DATA_PATH = "./nltk_data"
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.insert(0, NLTK_DATA_PATH)

for resource in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}', paths=[NLTK_DATA_PATH])
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

class Vocabulary:
    def __init__(self, freq_threshold=1): # <UNK> 감소를 위해 1로 하향 조정
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        # 1. 소문자화 및 특수 문자 제거 (정규표현식)
        # 알파벳과 숫자만 남기고 나머지는 공백으로 대체
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return nltk.word_tokenize(text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4 

        print(f"[*] {len(sentence_list)}개의 Train 문장으로부터 단어 수집 중...")
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    if word not in self.stoi:
                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1
        
        # 데이터 카드용 통계 출력
        self.display_stats(frequencies)

    def display_stats(self, frequencies):
        """기술 보고서 및 데이터 카드용 통계 시각화 보조"""
        print("-" * 40)
        print(f"📊 Vocabulary Statistics")
        print(f"  - Total Vocab Size (including special tokens): {len(self)}")
        print(f"  - Total Unique Tokens found: {len(frequencies)}")
        print(f"  - Tokens kept (freq >= {self.freq_threshold}): {len(self.stoi) - 4}")
        print(f"  - Top 5 Common Words: {frequencies.most_common(5)}")
        print("-" * 40)

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"[SUCCESS] {path} 저장 완료.")

if __name__ == "__main__":
    if not TRAIN_JSON.exists():
        print(f"[ERROR] {TRAIN_JSON} 파일이 없습니다.")
    else:
        with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            train_captions = [item['caption'] for item in data]

        # freq_threshold를 1로 설정하여 모든 train 단어 수용
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocabulary(train_captions)
        save_vocab(vocab, SAVE_PATH)
        
        print(f"[*] 최종 구축된 단어 사전 크기(vocab_size): {len(vocab)}")
        # 테스트
        sample = "A red flower, in the garden!"
        print(f"\n[TEST] 원문: {sample}")
        print(f"[TEST] 정제 후 토큰화: {vocab.tokenizer(sample)}")