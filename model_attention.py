import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# 1. Attention 레이어
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [batch_size, num_pixels, encoder_dim]
        # decoder_hidden: [batch_size, decoder_dim]
        att1 = self.encoder_att(encoder_out)     # [batch_size, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch_size, attention_dim]
        # 가중치 계산: 두 정보를 합쳐서 에너지를 구함
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) # [batch_size, num_pixels]
        alpha = self.softmax(att) # 주의 집중 가중치
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) # [batch_size, encoder_dim]
        return attention_weighted_encoding, alpha

# 2. Encoder: 공간 정보를 유지하도록 수정
class EncoderCNN(nn.Module):
    def __init__(self, encoder_type="mobilenet_v2"):
        super(EncoderCNN, self).__init__()
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # 마지막 Pooling과 FC 레이어를 제외하여 7x7 공간 정보를 남김
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.enc_dim = 512
        elif self.encoder_type == "mobilenet_v2":
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.backbone = mobilenet.features
            self.enc_dim = 1280
        else:
            raise ValueError(f"Unsupported encoder: {encoder_type}")

    def forward(self, images):
        # features: [batch_size, enc_dim, 7, 7] (입력 224x224 기준)
        features = self.backbone(images)
        # Attention 입력을 위해 [batch_size, 49, enc_dim] 형태로 변환
        features = features.permute(0, 2, 3, 1) 
        features = features.view(features.size(0), -1, features.size(3))
        return features

# 3. Decoder: LSTMCell을 사용한 Attention 기반 구조
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim=1280):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        
        # LSTMCell: 매 타임스텝마다 직접 루프를 돌며 처리
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        
        # 초기 은닉 상태/셀 상태를 이미지 특징으로부터 결정
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        # 임베딩 [batch_size, seq_len, embed_size]
        embeddings = self.embed(captions)
        
        # 초기 상태 초기화 (이미지 평균 특징값 사용)
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        # 문장 길이만큼 루프 (마지막 <END> 제외)
        decode_lengths = captions.size(1) - 1
        predictions = torch.zeros(batch_size, decode_lengths, self.vocab_size).to(encoder_out.device)
        
        for t in range(decode_lengths):
            # 1. Attention 가중치와 컨텍스트 벡터 계산
            attention_weighted_encoding, _ = self.attention(encoder_out, h)
            
            # 2. 현재 단어 임베딩 + 컨텍스트 벡터 결합
            gate_input = torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1)
            
            # 3. LSTM 스텝 진행
            h, c = self.decode_step(gate_input, (h, c))
            
            # 4. 결과 저장
            preds = self.linear(self.dropout(h))
            predictions[:, t, :] = preds
            
        return predictions

# 4. 통합 모델 (수정된 부분)
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, attention_dim=256, encoder_type="mobilenet_v2"):
        super(CNNtoRNN, self).__init__()
        # num_layers는 LSTMCell 구조상 여기서는 직접 쓰이지 않지만 인터페이스 유지를 위해 남겨둡니다.
        self.encoderCNN = EncoderCNN(encoder_type=encoder_type)
        self.decoderRNN = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=self.encoderCNN.enc_dim
        )

    def forward(self, images, captions):
        # 1. Encoder에서 이미지의 공간 특징 추출 [Batch, 49, 1280]
        features = self.encoderCNN(images)
        # 2. Decoder에서 Attention을 사용해 캡션 생성 시뮬레이션
        outputs = self.decoderRNN(features, captions)
        return outputs

# 기존 get_model 함수 유지 및 수정
def get_model(config, vocab_size):
    return CNNtoRNN(
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        vocab_size=vocab_size,
        num_layers=config["num_layers"],
        attention_dim=config.get("attention_dim", 256), # config에 없을 경우 대비 기본값
        encoder_type=config["encoder_type"]
    )