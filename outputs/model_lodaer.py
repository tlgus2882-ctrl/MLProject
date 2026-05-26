import torch
import json
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_classes():
    # 1. 클래스 정보 로드
    with open("model_data/class_info.json", "r", encoding="utf-8") as f:
        class_info = json.load(f)
    classes = class_info["classes"]
    
    # 2. torchvision을 이용한 모델 뼈대 생성 (노트북과 동일한 방식)
    model = models.efficientnet_b3(weights=None)
    
    # 3. 마지막 출력층(Classifier)을 우리 데이터의 클래스 개수에 맞게 변경
    # torchvision의 efficientnet은 마지막 층 구조가 classifier[1] 입니다.
    # 추가된 Dropout 층에 맞게 classifier.4의 in_features를 맞춰주어야 할 수 있습니다.
    # 하지만 로그를 보면 classifier.1 과 classifier.4 가 있습니다.
    # 안전하게 전체 classifier를 덮어씌웁니다. (노트북에서 학습한 구조와 동일하게)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, len(classes))
    )

    
    # 4. 체크포인트에서 가중치 로드
    checkpoint = torch.load("model_data/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    model.eval()
    return model, classes

def predict_ingredients(image_bytes, model, classes, threshold=0.4):
    # 이미지 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_bytes).convert("RGB")
    tensor_img = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.sigmoid(outputs).squeeze()
    
    # Threshold 이상의 확률을 가진 재료만 추출
    detected = []
    
    if probs.dim() == 0: 
        if probs.item() > threshold:
            detected.append(classes[0])
    else:
        for i, prob in enumerate(probs):
            if prob.item() > threshold:
                detected.append(classes[i])
            
    return detected
