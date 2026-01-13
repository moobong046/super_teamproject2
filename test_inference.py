import os
import torch
import json
import random
import pickle
import matplotlib.pyplot as plt
import textwrap
from PIL import Image
from torchvision import transforms
from pathlib import Path

# 기존 프로젝트 모듈 임포트
from model import CNNtoRNN
from data_loader import Vocabulary

# 1. 경로 및 설정
MODEL_PATH = "./checkpoints/best_mobilenet_v2_lstm_lr0.00014, bs64_AdamW.pth" # 실제 파일명으로 수정 필요
VOCAB_PATH = "vocab_3.pkl"
TEST_IMG_DIR = "./data/images/test"
TEST_JSON = "./data/captions/test_3.json"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def load_test_resources():
    # 단어 사전 로드
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    # 모델 로드 (checkpoint 저장 방식에 맞춰 구성)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNNtoRNN(
        embed_size=checkpoint.get('embed_size', 300),
        hidden_size=checkpoint.get('hidden_size', 512),
        vocab_size=checkpoint.get('vocab_size', len(vocab)),
        num_layers=checkpoint.get('num_layers', 1),
        encoder_type=checkpoint.get('encoder_type', 'mobilenet_v2'),
        decoder_type=checkpoint.get('decoder_type', 'lstm')
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 테스트 데이터 로드
    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    return vocab, model, test_data

def get_inference(image_path, model, vocab, transform):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model.encoderCNN(img_tensor)
        # Show & Tell 방식의 추론 시작
        inputs = features.unsqueeze(1)
        states = None
        result_caption = []

        for _ in range(25): # 최대 길이 20
            hiddens, states = model.decoderRNN.rnn(inputs, states)
            outputs = model.decoderRNN.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            
            word = vocab.itos[predicted.item()]
            if word == "<END>":
                break
            if word not in ["<START>", "<PAD>", "<UNK>"]:
                result_caption.append(word)
            
            inputs = model.decoderRNN.embed(predicted).unsqueeze(1)
            
    return " ".join(result_caption), image

def main():
    vocab, model, test_data = load_test_resources()

    # 이미지 전처리 (학습 시와 동일하게, Augmentation 제외)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 1. 랜덤하게 20개 클래스 선택
    all_classes = list(set([item['class'] for item in test_data]))
    selected_classes = random.sample(all_classes, min(20, len(all_classes)))

    # 2. 각 클래스당 1개의 샘플 추출
    samples = []
    for cls in selected_classes:
        cls_samples = [item for item in test_data if item['class'] == cls]
        samples.append(random.choice(cls_samples))

    # 3. 시각화
    fig = plt.figure(figsize=(22, 16))
    plt.subplots_adjust(hspace=0.8, wspace=0.3)

    print(f"[*] {len(samples)}개의 샘플에 대한 추론을 시작합니다...")

    for i, item in enumerate(samples):
        img_path = os.path.join(TEST_IMG_DIR, item['image'])
        pred_caption, original_img = get_inference(img_path, model, vocab, transform)
        
        if original_img is None: continue

        ax = fig.add_subplot(4, 5, i + 1)
        ax.imshow(original_img)
        
        wrapped_gt = textwrap.fill(f"GT: {item['caption']}", width=35)
        wrapped_pred = textwrap.fill(f"Pred: {pred_caption}", width=35)
        
        # 폰트 사이즈 조정 (fontsize=8~9 추천)
        title_text = f"Class: {item['class']}\n{wrapped_gt}\n{wrapped_pred}"
        ax.set_title(title_text, fontsize=9, pad=12, loc='center', color='black')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_name = "test_inference_results.png"
    plt.savefig(save_name, dpi=300)
    print(f"[✔] 결과 시각화가 완료되었습니다: {save_name}")
    plt.show()

if __name__ == "__main__":
    main()