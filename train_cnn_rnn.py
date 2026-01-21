import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pickle
import yaml
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

from data_loader import get_loader, Vocabulary
from model import get_model

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    
    # 1. 공통 설정 가져오기
    config = full_config['common'].copy()
    cnn_cfg = full_config['cnn_rnn']
    
    # 2. CNN-RNN 기본 설정 병합
    config.update({k: v for k, v in cnn_cfg.items() if k != 'params'})
    
    # 3. 현재 설정된 encoder_type에 맞는 세부 파라미터 자동 병합
    encoder_type = config['encoder_type']
    if encoder_type in cnn_cfg['params']:
        config.update(cnn_cfg['params'][encoder_type])
        print(f"[INFO] {encoder_type} 전용 파라미터 로드 완료.")
    
    return config

def train():
    CONFIG = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Vocab 로드 (pickle 에러 방지를 위해 Vocabulary 클래스 임포트 상태 확인)
    with open(CONFIG["vocab_path"], "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 데이터 로더 (CONFIG에 공통 경로가 포함되어 있음)
    train_loader, _ = get_loader(
        img_dir=CONFIG["train_img_dir"],
        json_path=CONFIG["train_json"],
        vocab=vocab,
        transform=transform,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        shuffle=True
    )

    val_loader, _ = get_loader(
        img_dir=CONFIG["val_img_dir"],
        json_path=CONFIG["val_json"],
        vocab=vocab,
        transform=transform,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        shuffle=False
    )

    # 모델 생성 (자동 로드된 embed_size, hidden_size 사용)
    model = get_model(
        model_name=CONFIG["model_name"],
        embed_size=CONFIG["embed_size"],
        hidden_size=CONFIG["hidden_size"],
        vocab_size=vocab_size,
        num_layers=CONFIG["num_layers"],
        encoder_type=CONFIG["encoder_type"]
    ).to(device)

    # Weights & Biases 초기화
    run_name = f"CNN_RNN_{CONFIG['encoder_type']}_lr{CONFIG['learning_rate']}"
    wandb.init(project="Image_Captioning", config=CONFIG, name=run_name)
    
    # 체크포인트 저장 경로 설정
    save_path = Path(CONFIG["checkpoint_dir"])
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / f"best_{run_name}.pth"

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    
    # 레이어별 차등 학습률 (통합 모델 구조에 맞춰 접근)
    optimizer = optim.AdamW([
        {'params': model.encoderCNN.backbone.parameters(), 'lr': float(CONFIG["learning_rate"]) * 0.1},
        {'params': model.encoderCNN.linear.parameters(), 'lr': float(CONFIG["learning_rate"])},
        {'params': model.encoderCNN.bn.parameters(), 'lr': float(CONFIG["learning_rate"])},
        {'params': model.decoderRNN.parameters(), 'lr': float(CONFIG["learning_rate"])}
    ], lr=float(CONFIG["learning_rate"]), weight_decay=0.01)

    best_val_loss = float('inf') 

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_train_loss = 0
        for imgs, captions in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()
            
        # 검증 로직 및 모델 저장 (이전과 동일)
        # ... (중략) ...
        print(f"Epoch {epoch} 완성") 

    wandb.finish()

if __name__ == "__main__":
    train()