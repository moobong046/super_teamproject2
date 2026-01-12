import os
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# [중요] 기존 프로젝트 구조에 맞게 임포트
from data_loader import Vocabulary, get_loader
from model import CNNtoRNN

# 1. 경로 설정 (여기만 수정하면 됩니다!)
MODEL_PATH = "./checkpoints/best_mobilenet_v2_lstm_lr0.0001, bs32_AdamW.pth"

# 파일명에서 태그 추출 (예: "best_mobilenet_v2_lstm_lr0.0002, bs128_AdamW" 추출)
model_filename = os.path.basename(MODEL_PATH).replace(".pth", "")
# 폴더명 및 파일 접두어로 사용
TAG = model_filename.replace("best_", "").replace("_AdamW", "") 

CONFIG = {
    "vocab_path": "vocab_3.pkl",
    "val_json": "./data/captions/val_3.json",
    "val_img_dir": "./data/images/val",
    "model_path": MODEL_PATH,
    "output_dir": f"./analysis_results_{TAG}"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# 2. 보조 함수
def get_predicted_class(caption, all_classes):
    """생성된 문장에서 클래스 키워드를 추출하여 예측 클래스 결정"""
    caption = caption.lower()
    for cls in all_classes:
        clean_cls = cls.replace('_', ' ').lower()
        if clean_cls in caption:
            return cls
    return "Unknown"


# 3. 통합 실행 함수
def run_full_analysis():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 분석 시작 (Device: {device}) | Tag: {TAG}")

    # --- [STEP 1] 데이터 및 모델 로드 ---
    with open(CONFIG["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    checkpoint = torch.load(CONFIG["model_path"], map_location=device)
    model = CNNtoRNN(
        embed_size=checkpoint.get('embed_size', 300),
        hidden_size=checkpoint.get('hidden_size', 512),
        vocab_size=checkpoint.get('vocab_size', len(vocab)),
        num_layers=checkpoint.get('num_layers', 1),
        encoder_type=checkpoint.get('encoder_type', 'mobilenet_v2'),
        decoder_type=checkpoint.get('decoder_type', 'lstm')
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    val_loader, dataset = get_loader({
        **CONFIG, "img_dir": CONFIG["val_img_dir"], "json_dir": CONFIG["val_json"],
        "batch_size": 1, "shuffle": False, "num_workers": 0, "image_size": (224, 224),
        "norm_mean": (0.485, 0.456, 0.406), "norm_std": (0.229, 0.224, 0.225)
    })

    with open(CONFIG["val_json"], 'r', encoding='utf-8') as f:
        val_raw = json.load(f)
    all_classes = sorted(list(set([item['class'] for item in val_raw])))

    # --- [STEP 2] 추론 루프 ---
    results = []
    chencherry = SmoothingFunction()

    print(f"[*] 총 {len(dataset)}개 샘플 추론 중...")
    for i in tqdm(range(len(dataset))):
        image, _ = dataset[i]
        gt_item = val_raw[i]
        
        img_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encoderCNN(img_tensor)
            inputs = features.unsqueeze(1)
            states = None
            pred_tokens = []
            for _ in range(25):
                hiddens, states = model.decoderRNN.rnn(inputs, states)
                outputs = model.decoderRNN.linear(hiddens.squeeze(1))
                predicted = outputs.argmax(1)
                word = vocab.itos[predicted.item()]
                if word == "<END>": break
                if word not in ["<START>", "<PAD>", "<UNK>"]:
                    pred_tokens.append(word)
                inputs = model.decoderRNN.embed(predicted).unsqueeze(1)
        
        pred_caption = " ".join(pred_tokens)
        pred_class = get_predicted_class(pred_caption, all_classes)
        bleu4 = sentence_bleu([gt_item['caption'].lower().split()], pred_tokens, 
                             weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
        results.append({
            "image_path": gt_item['image'],
            "gt_class": gt_item['class'],
            "pred_class": pred_class,
            "gt_caption": gt_item['caption'],
            "pred_caption": pred_caption,
            "bleu4": bleu4
        })

    df = pd.DataFrame(results)
    
    # 결과 파일 저장
    full_data_path = os.path.join(CONFIG["output_dir"], f"{TAG}_data_full.csv")
    df.to_csv(full_data_path, index=False, encoding='utf-8-sig')

    # --- [STEP 3] 고도화 통계 분석 (Advanced Stats) ---
    print("\n[*] 상세 성능 통계 계산 중...")
    classes = sorted(df['gt_class'].unique())
    
    # 지표 계산
    precision, recall, f1, _ = precision_recall_fscore_support(
        df['gt_class'], df['pred_class'], labels=classes, average=None, zero_division=0
    )
    
    accuracy_list = []
    for cls in classes:
        cls_df = df[df['gt_class'] == cls]
        acc = accuracy_score(cls_df['gt_class'], cls_df['pred_class'])
        accuracy_list.append(acc)

    # BLEU 점수 기반 표준편차
    std_report = df.groupby('gt_class')['bleu4'].agg(['mean', 'std', 'count']).reset_index()
    std_report.columns = ['Class', 'BLEU4_Mean', 'BLEU4_Std', 'Sample_Count']
    
    performance_df = pd.DataFrame({
        'Class': classes,
        'Accuracy': accuracy_list,
        'F1_Score': f1,
        'Precision': precision,
        'Recall': recall
    })
    
    final_report = pd.merge(std_report, performance_df, on='Class').sort_values(by='F1_Score', ascending=False)
    
    # 상세 보고서 저장
    detailed_path = os.path.join(CONFIG["output_dir"], f"{TAG}_detailed_report.csv")
    final_report.to_csv(detailed_path, index=False)

    # --- [STEP 4] 시각화 및 출력 ---
    # Confusion Matrix
    plt.figure(figsize=(22, 18))
    labels = all_classes + ["Unknown"]
    cm = confusion_matrix(df['gt_class'], df['pred_class'], labels=labels)
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=all_classes)
    plt.title(f"Confusion Matrix - {TAG}")
    plt.savefig(os.path.join(CONFIG["output_dir"], f"{TAG}_confusion_matrix.png"))
    plt.close()

    # 상위/하위 10개 출력
    print("\n" + "="*30 + " [TOP 10 CLASSES] " + "="*30)
    print(final_report.head(10)[['Class', 'F1_Score', 'BLEU4_Mean', 'Accuracy']].to_string(index=False))
    
    print("\n" + "="*30 + " [BOTTOM 10 CLASSES] " + "="*30)
    print(final_report.tail(10)[['Class', 'F1_Score', 'BLEU4_Mean', 'Accuracy']].to_string(index=False))

    print(f"\n[✔] 모든 분석이 완료되었습니다.")
    print(f"- 상세 데이터: {full_data_path}")
    print(f"- 통계 보고서: {detailed_path}")
    print(f"- 시각화 결과: {CONFIG['output_dir']} 폴더 내 저장됨")

if __name__ == "__main__":
    run_full_analysis()