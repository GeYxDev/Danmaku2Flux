from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.database import videoSentimentVectorDatabase
from core.pipeline import RecommendationPipeline


# Lifecycle manager
@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Application lifecycle manager
    :param application: Reference of application instances
    :return: None
    """
    print("Initializing database...")
    videoSentimentVectorDatabase.load_video_data("transformer_vector_danmu.json")
    yield
    print("Close the application and release resources...")

# Initialize FastAPI application
app = FastAPI(title="Bilibili Video Emotion Recommender", lifespan=lifespan)

# Configure cross domain resource sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# GET method to obtain recommendation list
@app.get("/recommend")
async def recommend_api(bv: str):
    """
    Video recommendation interface
    :param bv: Recommend similar videos based on this video
    :return: Service response results
    """
    # Exception handling when bv is empty
    if not bv:
        raise HTTPException(status_code=400, detail="BV number cannot be empty")

    # Call the video emotion recommendation system pipeline
    try:
        pipeline = RecommendationPipeline(bv=bv)
        results = pipeline.run()

        # Return the reason for the error when it occurs
        if pipeline.context.get("error") != "":
            return {
                "code": 422,
                "status": "failed",
                "message": pipeline.context["error"],
                "data": []
            }

        # Return results without similar videos
        if not results:
            return {
                "code": 404,
                "status": "success",
                "message": "Calculated successfully but no similar videos found in database.",
                "data": []
            }

        # Return information normally when receiving recommendation results
        return {
            "code": 200,
            "status": "success",
            "data": results
        }

    except Exception as e:
        print(f"Server Error processing {bv}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
