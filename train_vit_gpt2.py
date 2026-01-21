import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import json
import pickle
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

# 통합된 모듈에서 임포트
from data_loader import get_loader, Vocabulary
from model import get_model

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    config = full_config['common'].copy()
    config.update(full_config['vit_gpt2'])
    return config

def train_pretrained():
    # 1. 설정 로드
    CONFIG = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # 체크포인트 디렉토리 생성
    save_dir = Path(CONFIG["checkpoint_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Vocab 및 데이터 로더 준비
    with open(CONFIG["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_loader, _ = get_loader(
        img_dir=CONFIG["train_img_dir"],
        json_path=CONFIG["train_json"],
        vocab=vocab,
        transform=transform,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )

    # 3. 모델 생성
    model = get_model(
        model_name="vit_gpt2",
        embed_size=CONFIG["embed_size"],
        hidden_size=CONFIG["hidden_size"],
        vocab_size=len(vocab),
        num_layers=CONFIG["num_layers"]
    ).to(device)
    
    # wandb 초기화
    run_name = f"ViT_GPT2_ep{CONFIG['epochs']}_lr{CONFIG['learning_rate']}"
    wandb.init(project="Image_Captioning", config=CONFIG, name=run_name)

    print(f"[*] {run_name} 학습을 시작합니다.")

    # 4. 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]), weight_decay=0.01)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # 5. 학습 루프
    best_loss = float('inf')

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, captions in pbar:
            imgs, captions = imgs.to(device), captions.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, captions[:, :-1]) 
            
            # ViT-GPT2 슬라이싱 (이미지 패치 197개 제외)
            target_captions = captions[:, 1:]
            image_seq_len = 197 
            caption_out = outputs[:, image_seq_len:, :]

            loss = criterion(caption_out.reshape(-1, len(vocab)), target_captions.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # wandb 로그 기록
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss,
            "learning_rate": current_lr
        })
        
        print(f"Epoch {epoch} 완료 | Loss: {avg_loss:.4f} | LR: {current_lr}")
        
        scheduler.step()

        # Best 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = save_dir / CONFIG["save_name"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': CONFIG
            }, save_path)
            print(f"[✔] 최고 성능 갱신! 모델 저장 완료: {save_path}")

    wandb.finish()

if __name__ == "__main__":
    train_pretrained()