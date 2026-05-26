from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
import io
import os # 환경변수 사용을 위해 추가
import anthropic  # Claude API 라이브러리

# model_loader.py 파일에서 정의한 함수들을 가져옵니다.
from model_loader import load_model_and_classes, predict_ingredients

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 웹 서버가 시작될 때 인공지능 모델을 메모리에 한 번만 로드합니다.
model, classes = load_model_and_classes()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def process_image(file: UploadFile = File(...)):
    # 1. 업로드된 파일로부터 이미지 바이트 데이터 읽기
    image_bytes = io.BytesIO(await file.read())
    
    # 2. PyTorch 모델을 사용하여 이미지 속 비건 재료 예측
    detected_ingredients = predict_ingredients(image_bytes, model, classes, threshold=0.4)
    
    # 3. Claude API 호출 (프롬프트 엔지니어링 및 레시피 생성)
    if not detected_ingredients:
        recipe_result = "인식된 비건 재료가 없습니다. 다른 사진으로 다시 시도해 주세요!"
    else:
        prompt = (
            f"너는 전문 비건 요리사야. 다음 비건 재료들을 주재료로 활용한 맛있는 비건 요리 레시피를 3개 추천해줘.\n"
            f"탐지된 재료: {', '.join(detected_ingredients)}\n\n"
            f"각 레시피마다 [요리 이름], [필요한 추가 재료], [간단한 조리 순서]를 줄바꿈을 통해 보기 좋게 정리해서 한국어로 대답해줘."
        )
        
        try:
            # ✅ 수정됨: AsyncAnthropic 사용 (비동기 클라이언트)
            claude_client = anthropic.AsyncAnthropic(api_key="api_key") 
            # 팁: 나중에는 os.environ.get("ANTHROPIC_API_KEY") 처럼 변경하세요!
            
            # ✅ 수정됨: await 추가
            response = await claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            recipe_result = response.content[0].text
            
        except Exception as e:
            recipe_result = f"레시피를 생성하는 과정에서 API 오류가 발생했습니다: {str(e)}"
    
    # 4. 프론트엔드(index.html)로 탐지된 재료 리스트와 레시피 결과 전달
    return {
        "ingredients": detected_ingredients,
        "recipes": recipe_result
    }
