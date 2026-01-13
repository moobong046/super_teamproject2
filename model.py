import torch
import torch.nn as nn
import torchvision.models as models

# 1. Encoder
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, encoder_type="resnet18"):
        super(EncoderCNN, self).__init__()
        self.encoder_type = encoder_type.lower()
        
        # 가이드라인: CNN은 ResNet-18 사용
        if self.encoder_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # 마지막 FC 레이어를 제외한 모든 층을 백본으로 사용
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            in_features = resnet.fc.in_features
        
        elif self.encoder_type == "mobilenet_v2":
            # MobileNet V2 추가
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            # 마지막 classifier를 제외한 특징 추출기만 사용
            self.backbone = mobilenet.features 
            # MobileNet V2의 최종 채널 수는 1280
            in_features = mobilenet.last_channel 
            # Global Average Pooling 추가 (MobileNet 특징 맵 크기 대응)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {encoder_type}")

        # FC from scratch: 이 부분은 사전 학습되지 않은 상태로 초기화되어 학습됩니다.
        self.linear = nn.Linear(in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        # FC 레이어 가중치 초기화 (from scratch)
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.backbone(images)
        
        if self.encoder_type == "mobilenet_v2":
            features = self.adaptive_pool(features)
            features = features.view(features.size(0), -1)
        else:
            # ResNet의 출력은 [Batch, 512, 1, 1]이므로 평탄화 필요
            features = features.view(features.size(0), -1)
            
        features = self.bn(self.linear(features))
        return features

# 2. Decoder (Show & Tell / NIC 기반)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, decoder_type="lstm"):
        super(DecoderRNN, self).__init__()
        self.decoder_type = decoder_type.lower()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # 가이드라인: NIC(LSTM) 구조 반영
        if self.decoder_type == "lstm":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif self.decoder_type == "gru":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # 캡션에서 <END> 토큰 제외하고 임베딩
        embeddings = self.embed(captions[:, :-1])
        
        # 이미지 특징(features)을 첫 번째 시퀀스 입력으로 결합 [Batch, Seq_Len+1, Embed_Size]
        # Show & Tell 방식: 이미지를 문장의 첫 번째 단어처럼 취급
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        hiddens, _ = self.rnn(embeddings)
        outputs = self.linear(self.dropout(hiddens))
        return outputs

# 3. Full Model
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, encoder_type="resnet18", decoder_type="lstm"):
        super(CNNtoRNN, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoderCNN = EncoderCNN(embed_size, encoder_type=encoder_type)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, decoder_type=decoder_type)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs