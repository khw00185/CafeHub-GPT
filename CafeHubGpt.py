from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (필요에 따라 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

class ReviewRequest(BaseModel):
    reviews: list[str]

def summarize_reviews_combined(reviews: list[str]) -> str:
    combined_reviews = "\n".join(reviews)
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""다음 리뷰들은 특정 카페에 대한 리뷰야. 이 리뷰들을 종합해서 자연스러운 대화체로 제발 요약해주세요. 강점, 약점, 그리고 전반적인 감정을 강조하고 한글로 작성해주세요. 
                 제발 글자 수 50자 이내로 제공된 리뷰들을 전체적으로 종합적으로 요약해서 딱 두문장만 출력해주세요
                 너무 부정적인 리뷰는 생략해줘.
                 순차적으로 답변하지마.
                 리뷰가 없다면 '작성된 리뷰가 없습니다.'라고 출력해줘 제발.
                 마지막으로 제발 추가적인 설명이나 해설은 포함하지 마.
                 존댓말로 작성 바람.\n{combined_reviews}"""}
            ],
            max_tokens=200,  # 응답 길이
            temperature=0.5,  # 응답의 창의성
            top_p=1,  # 일관된 응답
            frequency_penalty=0.3  # 반복되는 단어 빈도 감소
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError as e:
        raise HTTPException(status_code=429, detail="Rate limit exceeded, please try again later")

@app.post("/summarize_review/")
async def summarize_review_api(request: ReviewRequest):
    summary = summarize_reviews_combined(request.reviews)
    return {"summary": summary} 
