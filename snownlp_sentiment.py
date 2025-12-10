import json
import os
import numpy as np
from snownlp import SnowNLP
from scipy.ndimage import gaussian_filter1d


# noinspection PyMethodMayBeStatic
class SnowNLPSentimentVectorBuilder:
    def __init__(self, smoothing_sigma=2.0):
        """
        Scoring barrage emotions and generating vectors using SnowNLP
        :param smoothing_sigma: smoothing factor
        """
        self.sigma = smoothing_sigma

    def get_danmu_sentiment_score(self, bucket_danmus):
        """
        Calculate the sentiment score for a single bucket
        :param bucket_danmus: Need to predict emotional barrel barrage
        :return: The emotional score of the barrage in the current bucket
        """
        # If there are no bullet comments in the bucket,
        # it is considered that the emotions of the entire bucket are neutral
        if not bucket_danmus:
            return 0.0

        # List for storing barrage emotions
        sentiment_scores = []
        for tokens in bucket_danmus:
            # Skip empty danmu
            if not tokens:
                continue

            # Combine word segmentation into text for a single danmu
            text = "".join(tokens)

            # Attempt to conduct sentiment analysis
            # noinspection PyBroadException
            try:
                s = SnowNLP(text)
                # Map the original ratings from 0 to 1 to -1 to 1
                score = (s.sentiments - 0.5) * 2
                sentiment_scores.append(score)
            except:
                # Skip characters that cannot be processed
                continue

        # If no valid scores were generated return 0.0
        if not sentiment_scores:
            return 0.0

        # Return the mean sentiment of all danmus in this bucket
        return float(np.mean(sentiment_scores))

    def compute_danmu_density(self, raw_bins):
        """
        Calculate the barrage density of each bucket
        :param raw_bins: Bucket list in a single video
        :return: Return bucket density vector
        """
        # Count the number of danmus for each bucket
        counts = np.array([len(bucket) for bucket in raw_bins], dtype=float)

        # Normalize the mapping density to between 0 and 1
        max_val = np.max(counts)
        if max_val > 0:
            normalized_counts = counts / max_val
        else:
            # Dealing with situations where all densities are 0
            normalized_counts = counts

        # Return bucket density vector
        return normalized_counts

    def process_video_by_danmu(self, video_data):
        """
        Generate emotion vectors for each video based on bullet comments
        :param video_data: The data and barrage data of this video
        :return: Return emotional rating related data of the video
        """
        # Read buckets containing bullet comments
        raw_bins = video_data.get('tokenized_bins', [])

        # Calculate the sentiment generation vector for each bucket
        sentiment_raw = []
        for bucket in raw_bins:
            score = self.get_danmu_sentiment_score(bucket)
            sentiment_raw.append(score)
        sentiment_raw = np.array(sentiment_raw)

        # Calculate the density of each bucket to obtain the density vector
        density_raw = self.compute_danmu_density(raw_bins)

        # Apply smoothing to emotion vectors and density vectors
        sentiment_smooth = gaussian_filter1d(sentiment_raw, sigma=self.sigma)
        density_smooth = gaussian_filter1d(density_raw, sigma=self.sigma)

        # Constructing a composite vector of emotions and density
        final_vector = np.concatenate([sentiment_smooth, density_smooth]).tolist()

        # Return processed video information and vectors
        return {
            "title": video_data.get('title'),
            "bv": video_data.get('bv'),
            "duration": video_data.get('duration'),
            "feature_vector": final_vector
        }


if __name__ == "__main__":
    # The folder where the JSON file containing the data is located
    INPUT_DIR = "processed_danmu_data_115"
    # The folder where the JSON file containing the output vector is located
    OUTPUT_DIR = "vector_danmu_data_115"

    # Ensure that the output folder exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize feature vector builder
    builder = SnowNLPSentimentVectorBuilder(smoothing_sigma=1.5)

    # Read the list of JSON files to be processed
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    total_files = len(files)

    print(f"Start vectorizing {total_files} files.")

    # Process each period's data one by one
    for idx, filename in enumerate(files, 1):
        # Composite file input link
        input_path = os.path.join(INPUT_DIR, filename)

        # Composite file output link
        output_path = os.path.join(OUTPUT_DIR, f"vector_{filename}")

        print(f"[{idx}/{total_files}] Processing {filename}.", end="\r")

        try:
            # Attempt to load data
            with open(input_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Attempt to process data
            processed_result = {}
            for vid_key, vid_info in data.items():
                vec_data = builder.process_video_by_danmu(vid_info)
                processed_result[vid_key] = vec_data

            # Attempt to save data
            with open(output_path, 'w', encoding='utf-8') as file:
                # noinspection PyTypeChecker
                json.dump(processed_result, file, ensure_ascii=False)

        except Exception as e:
            print(f"\nError processing {filename}: {e}.")

    print(f"\n\nDone! Vectors saved to {OUTPUT_DIR}.")
