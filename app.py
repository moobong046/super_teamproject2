import os
import torch
import pickle
import gradio as gr
import nltk
import re
from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator

# Model Lead의 아키텍처 파일 임포트
from model_arch import CNNtoRNN 

# [필수] Vocabulary 클래스 정의
class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
    def __len__(self):
        return len(self.itos)

# 경로 설정
DEVICE = torch.device("cpu")
BLIP_PATH = "/app/models/blip_v5" 
MOBILE_MODEL_PATH = "/app/models/mobilenet_lstm/best_model.pth"
VOCAB_PATH = "/app/models/mobilenet_lstm/vocab_3.pkl"

# 모델 로드 (전달받은 로컬 파일 기반)
print("⏳ 시퀀셜 담당자 파일로 BLIP 동기화 중...")
# 담당자가 준 폴더 내의 config와 vocab을 사용하도록 강제 지정
blip_processor = BlipProcessor.from_pretrained(BLIP_PATH)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_PATH).to(DEVICE)

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

# MobileNet 로드
checkpoint = torch.load(MOBILE_MODEL_PATH, map_location=DEVICE)
mobile_model = CNNtoRNN(300, 512, len(vocab), 1, 'mobilenet_v2', 'lstm').to(DEVICE)
mobile_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
mobile_model.eval()

# 가비지 절단 패턴 (PM님 정제 로직 반영)
CUT_PATTERNS = ['a photo', 'an image', 'the photo', 'a high', 'this is', 'with a a', 'of a a', 'and a', 'in a']
STOP_WORDS = ['in', 'the', 'at', 'with', 'and', 'of', 'showing', 'its', 'from', 'to', 'for', 'by', 'a', 'an', 'unk']

def refine_final_text(raw_text):
    # 1. 특수 토큰 제거
    text = raw_text.lower().strip()
    text = text.replace('unk', '').replace('<unk>', '')
    
    # 2. 첫 마침표 절단
    if '.' in text:
        text = text.split('.')[0].strip()
    
    # 3. 가비지 패턴 절단
    words = text.split()
    full_sentence = " ".join(words)
    cutoff_idx = len(words)
    for pattern in CUT_PATTERNS:
        if pattern in full_sentence:
            start_pos = full_sentence.find(pattern)
            curr = 0
            for idx, w in enumerate(words):
                curr = full_sentence.find(w, curr)
                if curr >= start_pos:
                    cutoff_idx = min(cutoff_idx, idx)
                    break
                curr += len(w)
    
    final_words = words[:cutoff_idx]
    
    # 4. 불용어 제거 및 중복 방지
    clean_res = []
    for w in final_words:
        if w not in STOP_WORDS and (not clean_res or w != clean_res[-1]):
            clean_res.append(w)
    
    eng = " ".join(clean_res).strip().capitalize()
    if eng: eng += "."
    else: eng = "Analysis complete."
    
    try:
        kor = GoogleTranslator(source='auto', target='ko').translate(eng)
    except:
        kor = "번역 엔진 연결 지연"
        
    return eng, kor

def predict(img):
    if img is None: return "이미지를 업로드하세요.", ""

    # BLIP 추론
    inputs = blip_processor(img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(
            **inputs, 
            max_new_tokens=40,
            num_beams=3, # 적절한 빔 서치
            repetition_penalty=1.4
        )
    b_raw = blip_processor.decode(out[0], skip_special_tokens=True)
    b_eng, b_kor = refine_final_text(b_raw)

    # MobileNet 추론
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    it = trans(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = mobile_model.encoderCNN(it).unsqueeze(1)
        states = None
        _, states = mobile_model.decoderRNN.rnn(feats, states)
        token = torch.tensor([vocab.stoi["<START>"]]).to(DEVICE)
        m_tokens = []
        for _ in range(20):
            emb = mobile_model.decoderRNN.embed(token).unsqueeze(1)
            h, states = mobile_model.decoderRNN.rnn(emb, states)
            outputs = mobile_model.decoderRNN.linear(h.squeeze(1))
            pred = outputs.argmax(1)
            word = vocab.itos[pred.item()]
            if word == "<END>": break
            m_tokens.append(word)
            token = pred
            
    m_eng, m_kor = refine_final_text(" ".join(m_tokens))

    return f"영문: {b_eng}\n한글: {b_kor}", f"영문: {m_eng}\n한글: {m_kor}"

# Gradio 인터페이스
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Sequential Model (BLIP v5)"),
        gr.Textbox(label="Model Lead (MobileNet+LSTM)")
    ],
    title="이미지 캡셔닝 최종 통합 데모"
).launch(server_name="0.0.0.0", server_port=9090)