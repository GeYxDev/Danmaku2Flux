import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity


# noinspection PyAttributeOutsideInit
class VideoSentimentVectorDatabase:
    """
    Video emotion vector database loader
    """
    # Store a reference to a unique instance
    _instance = None

    def __new__(cls):
        """
        Control instance creation and limit the number of objects
        """
        if cls._instance is None:
            cls._instance = super(VideoSentimentVectorDatabase, cls).__new__(cls)
            # Store video information and emotion vectors
            cls._instance.data = []
            # Store sentiment vector matrix
            cls._instance.matrix = None
            # Store BV numbers for fast mapping
            cls._instance.id_map = {}
            # Has the marked data been loaded
            cls._instance.loaded = False
        return cls._instance

    def load_video_data(self, json_path="database.json"):
        """
        Load video information and related emotion vectors
        :param json_path: Video related data stored in JSON files
        :return: None
        """
        # If the data has already been loaded, do not reload again
        if self.loaded:
            return

        # Attempt to load and optimize the storage database
        print(f"[Database] Loading data: {json_path}.")
        try:
            if not os.path.exists(json_path):
                print("The database does not exist, initialize an empty database.")
                self.data = []
            else:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)

            # Constructing a video emotion vector matrix and BV index
            if self.data:
                vectors = [item['embedding'] for item in self.data]
                self.matrix = np.array(vectors)
                self.id_map = {item['bv']: idx for idx, item in enumerate(self.data)}
            else:
                self.matrix = np.empty((0, 128))

            # Marking data loading completed
            self.loaded = True
            print(f"[Database] Loading completed, containing {len(self.data)} videos.")

        except Exception as e:
            print(f"[Database] Data and optimization loading failed: {e}")

    def find_vector_by_bv(self, bv):
        """
        Attempt to search for video sentiment vectors in the library based on BV number
        :param bv: The BV number corresponding to the video
        :return: Emotion vector found based on BV number
        """
        idx = self.id_map.get(bv)
        if idx is not None:
            return self.data[idx]['embedding']
        return None

    def search_similar_vectors(self, input_vector, top_k=5, exclude_bv=None):
        """
        Find similar vectors based on the distance between them
        :param input_vector: Input vector used for similarity vector search
        :param top_k: Find the number of similar vectors
        :param exclude_bv: Exclude the BV number corresponding to the current vector
        :return: Several videos with emotional changes that are most similar to the input video
        """
        # When the database is empty or not loaded properly, return an empty result
        if self.matrix is None or len(self.matrix) == 0:
            return []

        # Calculate the cosine similarity between the input vector and other vectors
        target = np.array(input_vector).reshape(1, -1)
        similarities = cosine_similarity(target, self.matrix).flatten()

        # Obtain the index of cosine similarity sorted from small to large
        top_indices = similarities.argsort()[::-1]

        # Return the most similar videos and exclude the input as video
        results = []
        for idx in top_indices:
            item = self.data[idx]
            # noinspection PyTypeChecker
            if item['bv'] == exclude_bv:
                continue

            # noinspection PyTypeChecker
            results.append({
                "title": item['title'],
                "bv": item['bv'],
                "link": f"https://www.bilibili.com/video/{item['bv']}",
                "score": float(similarities[idx])
            })

            if len(results) >= top_k:
                break

        # Return the most similar video information
        return results


# Initialize a global single video emotion vector database loader
videoSentimentVectorDatabase = VideoSentimentVectorDatabase()
