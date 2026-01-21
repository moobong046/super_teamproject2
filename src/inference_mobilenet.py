import torch
import pickle
from PIL import Image
from torchvision import transforms
from model import CNNtoRNN  # 리드님이 주신 model.py에서 클래스 임포트

# 1. 설정 및 경로 (도커 환경에 맞춰 설정)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/mobilenet_lstm/best_model.pth"
VOCAB_PATH = "./models/mobilenet_lstm/vocab_3.pkl"

# 2. 모델 및 단어 사전 로드
def load_mobile_model():
    # 단어 사전 로드
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    # 가중치 파일 로드
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 모델 초기화 (train.py에 명시된 하이퍼파라미터 적용)
    model = CNNtoRNN(
        embed_size=300, 
        hidden_size=512, 
        vocab_size=len(vocab), 
        num_layers=1,
        encoder_type="mobilenet_v2"
    ).to(DEVICE)
    
    # 가중치 주입 (save_data 형식에 맞춰 로드)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab

model, vocab = load_mobile_model()

# 3. 이미지 전처리 설정 (train.py 규격 준수)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 4. 추론 함수 (Greedy Search 방식)
def get_mobile_prediction(image_path, max_length=20):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    result_caption = []
    
    with torch.no_grad():
        # 이미지 특징 추출
        features = model.encoderCNN(image)
        states = None
        
        # 첫 번째 입력은 <START> 토큰 (vocab.stoi['<START>'] = 1)
        inputs = torch.tensor([vocab.stoi['<START>']]).unsqueeze(0).to(DEVICE)
        # 이미지 특징을 첫 번째 입력으로 넣기 위해 features 사용 (Show & Tell 방식)
        inputs = model.decoderRNN.embed(inputs)
        inputs = torch.cat((features.unsqueeze(1), inputs), dim=1)

        for _ in range(max_length):
            hiddens, states = model.decoderRNN.rnn(inputs, states)
            outputs = model.decoderRNN.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            
            word = vocab.itos[predicted.item()]
            if word == "<END>":
                break
            
            result_caption.append(word)
            inputs = model.decoderRNN.embed(predicted).unsqueeze(1)
            
    return " ".join(result_caption)