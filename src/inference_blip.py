import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
import sys
import glob
import random
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import re
import nltk

# ==========================================
# 1. ★ 핵심 수정: Vocabulary 클래스 직접 정의 ★
# ==========================================
# vocab_3.pkl 파일이 "이 클래스가 메인 화면에 있어야 해!"라고 요구하기 때문입니다.
class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        # 정규식으로 특수문자 제거 후 토큰화
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return nltk.word_tokenize(text)

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

# ==========================================
# 2. 설정
# ==========================================
JIN_FOLDER = "."   # 현재 폴더
IMAGE_FOLDER = "train"  # 이미지 폴더
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# NLTK 데이터 다운로드 (토크나이저 에러 방지)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ==========================================
# 3. 파일 및 모듈 로드
# ==========================================
vocab_files = glob.glob("*.pkl")
model_files = glob.glob("*.pth")

if not vocab_files or not model_files:
    print("🚨 [에러] .pkl 또는 .pth 파일이 현재 폴더에 없습니다.")
    sys.exit()

VOCAB_PATH = vocab_files[0]
MODEL_PATH = model_files[0]
print(f"✅ 파일 로드: {VOCAB_PATH}, {MODEL_PATH}")

# model.py 불러오기
try:
    from model import CNNtoRNN 
except ImportError:
    print("🚨 [에러] 'model.py' 파일이 없습니다. 현재 폴더에 복사해주세요!")
    sys.exit()

# ==========================================
# 4. 기능 함수 (후가공 & 생성)
# ==========================================
def process_multimodal_output(raw_caption):
    text = raw_caption.replace('<start>', '').replace('<end>', '').replace('<pad>', '').strip()
    NOISE_KEYWORDS = ['melon', 'professional', 'photo', 'stock', 'image', 'high', 'resolution', 'vector', 'illustration']
    STOP_WORDS = ['and', 'with', 'a', 'the', 'in', 'on', 'at', 'of']
    
    if '.' in text: text = text.split('.')[0].strip()
    words = text.split()
    cutoff = len(words)
    for i, w in enumerate(words):
        if "".join(filter(str.isalnum, w.lower())) in NOISE_KEYWORDS:
            cutoff = i; break
    words = words[:cutoff]
    while words and words[-1].lower() in STOP_WORDS: words.pop()
    
    final_eng = " ".join(words) + "." if words else text + "."
    try: final_kor = GoogleTranslator(source='auto', target='ko').translate(final_eng)
    except: final_kor = "번역 실패"
    return final_eng, final_kor

def generate_caption_step_by_step(model, image, vocab, max_len=20):
    result_caption = []
    with torch.no_grad():
        features = model.encoderCNN(image).unsqueeze(1)
        states = None 
        _, states = model.decoderRNN.rnn(features, states)
        
        # <START> 토큰으로 시작
        start_token = vocab.stoi.get("<START>", 1)
        inputs = model.decoderRNN.embed(torch.tensor([start_token]).to(device)).unsqueeze(1)
        
        for _ in range(max_len):
            hiddens, states = model.decoderRNN.rnn(inputs, states)
            outputs = model.decoderRNN.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            word = vocab.itos[predicted.item()]
            
            if word == "<END>": break
            result_caption.append(word)
            inputs = model.decoderRNN.embed(predicted).unsqueeze(1)
            
    return " ".join(result_caption)

# ==========================================
# 5. 메인 실행
# ==========================================
if __name__ == "__main__":
    # 1. Vocab 로드 (이제 에러 안 남!)
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"📖 단어 사전 로드 완료 (크기: {len(vocab)})")

    # 2. 모델 설정 로드
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # 설정값 (기본값 + 저장된 값)
    cfg = checkpoint.get('config', {})
    embed_size = cfg.get('embed_size', 300)
    hidden_size = cfg.get('hidden_size', 512)
    num_layers = cfg.get('num_layers', 1)
    encoder_type = cfg.get('encoder_type', 'mobilenet_v2')
    decoder_type = cfg.get('decoder_type', 'lstm')

    # 3. 모델 생성 & 가중치 적용
    model = CNNtoRNN(embed_size, hidden_size, len(vocab), num_layers, encoder_type, decoder_type).to(device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()

    # 4. 이미지 추론
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_files = glob.glob(os.path.join(IMAGE_FOLDER, "**/*.jpg")) + glob.glob(os.path.join(IMAGE_FOLDER, "**/*.png"))
    
    if img_files:
        samples = random.sample(img_files, min(20, len(img_files))) # 20개 확인
        
        plt.figure(figsize=(20, 10))
        # 5열 그리드 계산
        cols = 5
        rows = (len(samples) + cols - 1) // cols
        
        print("\n🚀 결과 생성 중...\n")
        for i, img_path in enumerate(samples):
            try:
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                raw = generate_caption_step_by_step(model, img_tensor, vocab)
                eng, kor = process_multimodal_output(raw)
                
                plt.subplot(rows, cols, i+1)
                plt.imshow(image)
                plt.axis("off")
                
                # 텍스트 줄바꿈
                wrapped_text = f"[Eng] {eng[:40]}\n[Kor] {kor[:30]}"
                plt.title(wrapped_text, fontsize=9, loc='left', color='blue')
                print(f"[{i+1}] {eng}")
                
            except Exception as e:
                print(f"에러: {e}")
                
        plt.tight_layout()
        plt.show()
    else:
        print("🚨 이미지가 없습니다.")