import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from pathlib import Path

from data_loader import get_loader, Vocabulary
from model import CNNtoRNN

# 1. 전역 설정
CONFIG = {
    # 사용할 엔코더 선택: resnet18, mobilenet_v2
    "encoder_type": "mobilenet_v2", 
    
    "vocab_path": "vocab_3.pkl",
    "train_img_dir": "./data/images/train",
    "train_json": "./data/captions/train_3.json",
    "val_img_dir": "./data/images/val",
    "val_json": "./data/captions/val_3.json",
    "save_base_dir": "./checkpoints", 
    
    "optimizer_type": "AdamW",
    "learning_rate": 2e-4,
    "epochs": 30,
    "batch_size": 128,
    "num_workers": 8,
    "shuffle": True,
    "image_size": (224, 224),
    "norm_mean": (0.485, 0.456, 0.406),
    "norm_std": (0.229, 0.224, 0.225)
}

# 2. 엔코더 타입에 따른 최적 디코더 하이퍼파라미터 자동 설정 함수
def get_model_config(encoder_type):
    if encoder_type == "resnet18":
        # 가이드라인의 Show & Tell(NIC) 표준 조합
        return {"decoder_type": "lstm", "embed_size": 512, "hidden_size": 512, "num_layers": 1}
    elif encoder_type == "mobilenet_v2":
        # 경량 모델이므로 임베딩 사이즈를 약간 조정하거나 동일하게 유지하여 비교
        return {"decoder_type": "lstm", "embed_size": 300, "hidden_size": 512, "num_layers": 1}
    else:
        return {"decoder_type": "lstm", "embed_size": 256, "hidden_size": 512, "num_layers": 1}

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 사용 디바이스: {device}")

    # 엔코더에 따른 세부 설정 로드
    m_cfg = get_model_config(CONFIG["encoder_type"])
    CONFIG.update(m_cfg)

    # 데이터 로더
    train_loader, train_dataset = get_loader({**CONFIG, "img_dir": CONFIG["train_img_dir"], "json_dir": CONFIG["train_json"]})
    val_loader, _ = get_loader({**CONFIG, "img_dir": CONFIG["val_img_dir"], "json_dir": CONFIG["val_json"], "shuffle": False})
    
    vocab = train_dataset.vocab
    vocab_size = len(vocab)

    # 모델 생성 (encoder_type과 decoder_type 명시적 전달)
    model = CNNtoRNN(
        embed_size=CONFIG["embed_size"], 
        hidden_size=CONFIG["hidden_size"], 
        vocab_size=vocab_size, 
        num_layers=CONFIG["num_layers"],
        encoder_type=CONFIG["encoder_type"],
        decoder_type=CONFIG["decoder_type"]
    ).to(device)

    run_name = f"{model.encoder_type}_{model.decoder_type}_ep{CONFIG['epochs']}_lr{CONFIG['learning_rate']}_bs{CONFIG['batch_size']}_AdamW"
    model_filename = f"best_{model.encoder_type}_{model.decoder_type}_lr{CONFIG['learning_rate']}, bs{CONFIG['batch_size']}_AdamW.pth"

    wandb.init(project="Image_Captioning", config=CONFIG, name=run_name)
    
    save_path = Path(CONFIG["save_base_dir"])
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / model_filename

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.AdamW([
    # Backbone: 사전 학습된 지식을 유지하기 위해 매우 낮게 설정 (1e-5)
        {'params': model.encoderCNN.backbone.parameters(), 'lr': CONFIG["learning_rate"] * 0.1},
        # New Layers: 처음부터 배우는 부분은 표준 학습률 적용 (1e-4)
        {'params': model.encoderCNN.linear.parameters(), 'lr': CONFIG["learning_rate"]},
        {'params': model.encoderCNN.bn.parameters(), 'lr': CONFIG["learning_rate"]},
        {'params': model.decoderRNN.parameters(), 'lr': CONFIG["learning_rate"]}
    ], lr=CONFIG["learning_rate"], weight_decay=0.01)

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
            
        # Validation
        model.eval()
        total_val_loss, val_correct, val_tokens = 0, 0, 0
        with torch.no_grad():
            for imgs, captions in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                imgs, captions = imgs.to(device), captions.to(device)
                outputs = model(imgs, captions)
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                total_val_loss += loss.item()

                _, predicted = outputs.max(2)
                mask = captions != vocab.stoi["<PAD>"]
                val_correct += (predicted[mask] == captions[mask]).sum().item()
                val_tokens += mask.sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (val_correct / val_tokens) * 100

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch} | Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 저장할 데이터
            save_data = {
                'model_state_dict': model.state_dict(),
                'encoder_type': model.encoder_type,
                'decoder_type': model.decoder_type,
                'embed_size': CONFIG['embed_size'],
                'hidden_size': CONFIG['hidden_size'],
                'num_layers': CONFIG['num_layers'],
                'vocab_size': vocab_size,
                'config': CONFIG
            }
            torch.save(save_data, best_model_path)
            print("  >> Best Model Saved!")

    wandb.finish()

if __name__ == "__main__":
    train()