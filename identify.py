from deepface import DeepFace
import os

folder_path = '/Users/isaacjieu/Documents/projects/genesis/faces'
file_names = os.listdir(folder_path)
jpg_files = [f for f in file_names if f.endswith('.jpg') and os.path.isfile(os.path.join(folder_path, f))]

results = []

for file in jpg_files:
    dfs = DeepFace.verify(
        enforce_detection=False,
        img1_path = "/Users/isaacjieu/Documents/projects/genesis/test_faces/isaac_test.png",
        img2_path = f"/Users/isaacjieu/Documents/projects/genesis/faces/{file}",
    )

    results.append({
        "name": file,
        "score": 1-dfs["distance"]
    })

print(max(results, key=lambda x:x['score'])['name'])