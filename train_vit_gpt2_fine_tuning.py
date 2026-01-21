import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import pickle
import wandb
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

def resume_train():
    # 1. 설정 로드
    CONFIG = load_config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # [추가] 체크포인트 폴더 생성
    save_dir = Path(CONFIG["checkpoint_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 데이터 및 Vocab 준비
    with open(CONFIG["vocab_path"], "rb") as f:
        vocab = pickle.load(f)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
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

    # 3. 모델 로드 및 사용자 체크포인트 주입
    model = get_model(
        model_name="vit_gpt2",
        embed_size=CONFIG["embed_size"],
        hidden_size=CONFIG["hidden_size"],
        vocab_size=len(vocab),
        num_layers=CONFIG["num_layers"]
    ).to(device)

    # 불러올 이전 모델 경로 (일반적으로 가장 최근의 베스트 모델)
    base_checkpoint_name = "best_vit_gpt2_lr1e-5, bs32_AdamW.pth" 
    load_path = save_dir / base_checkpoint_name
    
    if load_path.exists():
        print(f"[*] 기존 모델({load_path})을 불러와 보완 학습을 시작합니다.")
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"[!] {load_path} 파일을 찾을 수 없어 구글 가중치 상태로 시작합니다.")

    # [추가] wandb 초기화 및 동적 실행 이름 설정
    run_name = f"ViT_FineTune_ep{CONFIG['epochs']}_bs{CONFIG['batch_size']}_lr{CONFIG['learning_rate']}"
    wandb.init(project="Image_Captioning", config=CONFIG, name=run_name)

    # 4. 보완 학습 설정
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]), weight_decay=0.05)

    # 5. 학습 루프
    best_loss = float('inf')

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Refining Epoch {epoch}")
        
        for imgs, captions in pbar:
            imgs, captions = imgs.to(device), captions.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, captions[:, :-1]) 
            
            caption_out = outputs[:, 197:, :] 
            loss = criterion(caption_out.reshape(-1, len(vocab)), captions[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        
        # [추가] wandb 로그 기록
        wandb.log({
            "refine_epoch": epoch,
            "refine_loss": avg_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch} 완성 | Loss: {avg_loss:.4f}")

        # [수정] 하이퍼파라미터가 포함된 이름으로 모델 저장
        save_file_name = f"refined_vit_gpt2_ep{CONFIG['epochs']}_bs{CONFIG['batch_size']}_lr{CONFIG['learning_rate']}.pth"
        save_path = save_dir / save_file_name

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CONFIG,
                'epoch': epoch,
                'loss': avg_loss
            }, save_path)
            print(f"[✔] 보완된 모델 저장 완료: {save_path}")

    wandb.finish()

if __name__ == "__main__":
    resume_train()