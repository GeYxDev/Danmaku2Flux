import json
import os
import glob

def merge_vectors_to_single_file(input_dir, output_file):
    print(f"Scanning directory: {input_dir} .")
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    total_files = len(json_files)
    if total_files == 0:
        print("No JSON files found!")
        return
    all_vectors_data = []
    video_count = 0
    for idx, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for vid_key, content in data.items():
                    title = content.get('title', 'Unknown')
                    bv = content.get('bv', '')
                    vector = content.get('feature_vector', [])
                    if vector and len(vector) == 200:
                        all_vectors_data.append({"title": title, "bv": bv, "vector": vector})
                        video_count += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        if idx % 100 == 0:
            print(f"Processed {idx}/{total_files} files. (Collected {video_count} videos)", end='\r')
    print(f"\n\nExtraction complete!")
    print(f"Total files processed: {total_files}")
    print(f"Total videos collected: {video_count}")
    print(f"Saving to {output_file}.")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(all_vectors_data, f_out, ensure_ascii=False, indent=None)
    print("Done.")

if __name__ == "__main__":
    INPUT_DIR = "vector_danmu_data_115"
    OUTPUT_FILE = "simplified_vector_danmu.json"
    merge_vectors_to_single_file(INPUT_DIR, OUTPUT_FILE)
