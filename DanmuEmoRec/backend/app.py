from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.database import db
from core.pipeline import RecommendationPipeline

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 启动时加载一次数据库
@app.on_event("startup")
async def startup_event():
    db.load_data("database.json")


@app.get("/recommend")
async def recommend_api(bvid: str):
    if not bvid:
        raise HTTPException(status_code=400, detail="BVID is required")

    try:
        # ✅ 所有的复杂逻辑都封装在 Pipeline 里了
        # 就像工厂流水线一样：输入原材料(bvid) -> 产出成品(list)
        pipeline = RecommendationPipeline(bvid)
        results = pipeline.run()

        return {
            "status": "success",
            "data": results,
            "source": "cache" if pipeline.context.get('vector') == db.find_vector_by_bvid(bvid) else "realtime"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)