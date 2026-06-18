import pandas as pd

df = pd.read_excel("data/annotated reviews.xlsx", engine="openpyxl", keep_default_na=False)

# Fix dirty labels
df['label'] = df['label'].replace({'negetive': 0, '\n2': 2})
df['label'] = df['label'].astype(int)
import pandas as pd
import requests

df = pd.read_excel("data/annotated reviews.xlsx", engine="openpyxl", keep_default_na=False)

# Fix dirty labels
df['label'] = df['label'].replace({'negetive': 0, '\n2': 2})
df['label'] = df['label'].astype(int)

def check_and_fix_image_paths(val):
    if not val or str(val).strip() in ("", "nan"):
        return None

    valid_filenames = []
    for u in str(val).split('|'):
        u = u.strip()
        if not u:
            continue
        filename = u.split('/file/')[-1]
        if not filename.endswith('.jpg'):
            filename = filename + '.jpg'
        url = f"https://down-ph.img.susercontent.com/file/{filename.replace('.jpg', '')}"
        try:
            response = requests.head(url, timeout=5)  # head = fast, no download
            if response.status_code == 200:
                valid_filenames.append(filename)
            else:
                print(f"Skipping {filename}: {response.status_code}")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    return ";".join(valid_filenames) if valid_filenames else None

print("Checking image URLs... this may take a while")
df['image_paths'] = df['image_paths'].apply(check_and_fix_image_paths)

# Drop rows where all images were invalid
df = df[df['image_paths'].notna()].reset_index(drop=True)

df.to_csv("data/annotated_reviews.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} rows with valid images to data/annotated_reviews.csv")
# Fix image_paths: extract filename + add .jpg + use ; as separator
def fix_image_paths(val):
    if not val or str(val).strip() in ("", "nan"):
        return val
    filenames = [u.strip().split('/file/')[-1] for u in str(val).split('|')]
    filenames = [f if f.endswith('.jpg') else f + '.jpg' for f in filenames]
    return ";".join(filenames)

df['image_paths'] = df['image_paths'].apply(fix_image_paths)

df.to_csv("data/annotated_reviews.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} rows to data/annotated_reviews.csv")
print(df['image_paths'].head(3))