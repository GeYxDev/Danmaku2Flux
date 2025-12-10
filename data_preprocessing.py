import json
import os
import re
import jieba


class TransformerDataPreprocessor:
    def __init__(self, stopwords_path='stopwords.txt', num_bins=100, min_danmu_count=100, min_fill_ratio=0.2):
        """
        Convert raw barrage data into types suitable for BERT and Transformer models
        :param stopwords_path: Stop word list path
        :param num_bins: Number of video clips
        :param min_danmu_count: Minimum number of individual barrages
        :param min_fill_ratio: Minimum density of individual video shards
        """
        self.num_bins = num_bins
        self.min_danmu_count = min_danmu_count
        self.min_fill_ratio = min_fill_ratio

        # Loading stop words
        self.stopwords = self.load_stopwords(stopwords_path)

        # Reserved emotional symbols
        self.protected_symbols = {'!', '！', '?', '？', '~', '～', '6', '2', '3', '5', 'h', 'H', 'w', 'W'}

        # Remove invisible characters or completely garbled characters
        self.noise_pattern = re.compile(r'\s+')

    @staticmethod
    def load_stopwords(stopwords_path):
        """
        Load stop word list
        :param stopwords_path: Stop Word List Path
        :return: Stop word set
        """
        if not os.path.exists(stopwords_path):
            print(f"Warning: Stopwords file {stopwords_path} not found. Running without stopwords filter.")
            return set()
        with open(stopwords_path, 'r', encoding='utf-8') as stopwords:
            return set([stopword.strip() for stopword in stopwords])

    def is_protected(self, word):
        """
        Protect emotional characters in bullet comments
        :param word: Characters under inspection
        :return: Is the character protected
        """
        for char in word:
            if char in self.protected_symbols:
                return True
        return False

    def clean_and_tokenize(self, text):
        """
        Clean the barrage and segment it into words
        :param text: Bullet comments that need to be handled
        :return: Processed barrage
        """
        if not text:
            return []

        # Remove the first and last blank spaces from the text
        text = text.strip()

        # Segment the text into words
        tokens = jieba.lcut(text)

        # Record valid characters
        valid_tokens = []
        for token in tokens:
            # Remove blank tokens
            if not token or self.noise_pattern.fullmatch(token):
                continue

            # Retain characters that are not in the stop word list or in protected words
            if (token not in self.stopwords) or self.is_protected(token):
                valid_tokens.append(token)

        # Return the processed barrage
        return valid_tokens

    def check_danmaku_quality(self, danmu_list, duration):
        """
        Check if the density and quality of the barrage meet the standards
        :param danmu_list: List of bullet comments for a single video
        :param duration: The duration of the corresponding video
        :return: Is the density and quality of the barrage qualified
        """
        # Check if the number of barrages meets the requirements
        if len(danmu_list) < self.min_danmu_count:
            return False, "Low count"

        # Check if the video duration is correct
        if duration <= 0:
            return False, "Zero duration"

        # Check if the barrage density within each segment meets the standard
        filled_bins = set()
        for d in danmu_list:
            # Calculate which video fragment the barrage belongs to
            b_idx = min(int((d['time'] / duration) * self.num_bins), self.num_bins - 1)
            # No empty shards will be recorded
            filled_bins.add(b_idx)

        # Check the empty bucket rate
        if (len(filled_bins) / self.num_bins) < self.min_fill_ratio:
            return False, "Sparse distribution"

        # The quality of the above inspection is recognized
        return True, "OK"

    def run(self, input_file, output_file):
        """
        Batch processing of barrage raw data
        :param input_file: The file containing the original barrage data
        :param output_file: The file to save the processed barrage to
        :return: None
        """
        print(f"Loading raw data from {input_file}.")

        # Attempt to read a period of barrage data
        try:
            with open(input_file, 'r', encoding='utf-8') as danmu_file:
                raw_data = json.load(danmu_file)
        except Exception as e:
            print(f"Error reading input file: {e}.")
            return

        # Store processed data
        processed_data = {}
        # Calculation of video quantity
        total_videos = len(raw_data)
        # Processed video barrage count
        kept_videos = 0

        print(f"Start processing {total_videos} videos.")

        # Process data in video units
        for video_key, info in raw_data.items():
            danmu_list = info.get('danmu_list', [])

            # Ensure that the barrage is sorted by time
            danmu_list.sort(key=lambda x: x['time'])

            # Obtain video duration
            duration = danmu_list[-1]['time'] if danmu_list else 0.0

            # Filter out videos with low barrage quality
            is_valid, reason = self.check_danmaku_quality(danmu_list, duration)
            # Videos that do not meet the standards will be skipped
            if not is_valid:
                continue

            # Initialize N buckets for storing bullet comments
            bins = [[] for _ in range(self.num_bins)]

            # Sort the barrage into corresponding buckets one by one
            for item in danmu_list:
                # Extract barrage information
                content = item.get('content', '')
                timestamp = item.get('time', 0.0)

                # Clear and segmented to the barrage
                tokens = self.clean_and_tokenize(content)

                # Invalid barrage will be skipped
                if not tokens:
                    continue

                # Calculate which bucket the barrage belongs to
                bin_idx = min(int((timestamp / duration) * self.num_bins), self.num_bins - 1)

                # Place the barrage in the corresponding bucket
                bins[bin_idx].append(tokens)

            # Save the processed barrage
            processed_data[video_key] = {
                "title": info.get('title'),
                "bv": info.get('bv'),
                "duration": duration,
                # List[List[List[str]]] for time ordered barrage distribution
                "tokenized_bins": bins
            }

            # Processed video count
            kept_videos += 1

            # Video processing progress prompt
            if kept_videos % 10 == 0:
                print(f"Processed {kept_videos} videos.", end='\r')

        # Save processed barrage data
        print(f"\nSaving processed data to {output_file}.")
        with open(output_file, 'w', encoding='utf-8') as file:
            # noinspection PyTypeChecker
            json.dump(processed_data, file, ensure_ascii=False, indent=None)

        print(f"Done! Kept {kept_videos}/{total_videos} videos.")


if __name__ == "__main__":
    # Enter folder path
    INPUT_DIR = "danmu_data_115"
    # Output folder path
    OUTPUT_DIR = "processed_danmu_data_115"
    # Stop using the vocabulary path
    STOPWORDS_FILE = 'stopwords.txt'

    # Ensure that the output folder exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Initialize the barrage processor
    processor = TransformerDataPreprocessor(stopwords_path=STOPWORDS_FILE, num_bins=100,
                                            min_danmu_count=100, min_fill_ratio=0.2)

    # Retrieve a list of JSON files from the input folder
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    total_files = len(json_files)

    # Check if there is a JSON file available
    if total_files == 0:
        print(f"Error: Cannot find json file in {INPUT_DIR} folder.")
    else:
        print(f"Find {total_files} and prepare to start batch processing.")

        # Process files one by one
        for index, filename in enumerate(json_files, 1):
            # Build input file name
            input_path = os.path.join(INPUT_DIR, filename)

            # Build output file name
            output_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")

            print(f"[{index}/{total_files}] Processing: {filename}.")

            # Call the bullet screen processing function
            processor.run(input_path, output_path)

        print(f"All files have been processed! The results have been saved in the '{OUTPUT_DIR}' folder.")
