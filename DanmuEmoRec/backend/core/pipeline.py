from .database import videoSentimentVectorDatabase
from .services import (
    DanmakuCrawlerService,
    transformerDataPreprocessor,
    sentimentAnalyzerService,
    transformerRecommendModelService
)


class RecommendationPipeline:
    """
    Video emotion recommendation system processing pipeline
    """
    def __init__(self, bv):
        """
        Initialize recommendation system processing pipeline
        :param bv: The BV number used for recommending reference videos
        """
        self.bv = bv
        # A data structure used to transmit data at various stages of a pipeline
        self.context = {
            "bv": bv,
            "vector": None,
            "recommendations": [],
            "error": ""
        }

    def run(self):
        """
        Start the video emotional recommendation pipeline flow
        :return: Video recommendation results
        """
        print(f"[Pipeline] Start processing task: {self.bv}.")

        # Ensure that the database is loaded
        videoSentimentVectorDatabase.load_video_data()

        # Check database cache
        self._step_check_database()

        # Generate video emotion embedding vectors from raw barrage data
        if self.context["vector"] is None:
            self._step_fetch_and_compute()

        # Vector search for similar videos
        self._step_search()

        # Return video recommendation results
        return self.context["recommendations"]

    def _step_check_database(self):
        """
        Check if there is a vector pointed to by this BV number in the current database
        :return: None
        """
        cached_vector = videoSentimentVectorDatabase.find_vector_by_bv(self.bv)
        if cached_vector:
            print("[Pipeline] Hit database cache, skip real-time calculation.")
            self.context["vector"] = cached_vector

    def _step_fetch_and_compute(self):
        """
        Pipeline flow starting from crawlers
        :return: None
        """
        print("[Pipeline] Failed to hit cache, start real-time calculation process.")

        # Crawler retrieves raw barrage data
        danmaku_list = DanmakuCrawlerService.fetch_danmaku(self.bv)
        if not danmaku_list:
            print("[Pipeline] Crawling failed or no barrage, terminate the process.")
            self.context["error"] = "Failed to obtain danmaku"
            return

        # Preprocessing of raw barrage data
        processed_data = transformerDataPreprocessor.process(danmaku_list, bv=self.bv)
        if not processed_data:
            print("[Pipeline] Preprocessing failed, terminate process.")
            self.context["error"] = "Insufficient danmaku for pre-processing"
            return

        # Perform sentiment analysis on preprocessed dataPerform sentiment analysis on preprocessed data
        sentiment_data = sentimentAnalyzerService.process_video_by_danmu(processed_data)
        if not sentiment_data or "feature_vector" not in sentiment_data:
            print("[Pipeline] Emotional feature extraction failed, terminate process.")
            self.context["error"] = "Unable to extract danmaku emotions"
            return

        raw_feature_vector = sentiment_data["feature_vector"]

        # Convert the original vector into an embedded vector through the model
        # noinspection PyBroadException
        try:
            feature_vector = transformerRecommendModelService.predict(raw_feature_vector)

            # Save the generated embedding vector
            self.context["vector"] = feature_vector
            print("[Pipeline] Real time calculation completed, vector generated.")

        except Exception:
            print("[Pipeline] Model inference failed, terminate process.")
            self.context["error"] = "Unable to extract emotional features from the video"

    def _step_search(self):
        """
        Vector similarity matching
        :return: None
        """
        target_vector = self.context["vector"]

        # Search if there is an embedded vector present
        if target_vector:
            # Call the search function in the database
            results = videoSentimentVectorDatabase.search_similar_vectors(target_vector, top_k=5, exclude_bv=self.bv)
            self.context["recommendations"] = results
            print(f"[Pipeline] Search completed, found {len(results)} related videos.")
        else:
            print("[Pipeline] The vector is empty, unable to perform search.")
            if self.context["error"] == "":
                self.context["error"] = "Vector missing cannot perform search"
