import json
import os

# --- FILE PATHS (Update these if filenames differ) ---
IMGLIST_1K = "data/benchmark_imglist/imagenet_ln/imagenet_ln_v6.txt"  # Input: The current 1k list
SYNSET_FILE = "data/imagenet_synsets.txt"  # Bridge: Line # corresponds to Old Label #
MAPPING_JSON = "data/potential_mapping_for_rex_v1.json"  # Mapping: Synset -> New 335 Class
OUTPUT_IMGLIST = "data/benchmark_imglist/imagenet_ln/imagenet_ln_335_v1.txt"  # Output: The new list
REPO_JSON = "C:\\Users\Rex Stayer-Suprick\PycharmProjects\pami_osr\\npy_json_files_shuffled\\valid_known_known.json"

def generate_corrected_list():
    # --- STEP 1: Learn the Repo's Labeling Scheme ---
    print(f"Reading {REPO_JSON} to learn target labels...")
    folder_to_target_label = {}

    try:
        with open(REPO_JSON, 'r') as f:
            repo_data = json.load(f)

        # Iterate through entries to find unique Folder -> Label pairings
        # Paths look like: ".../known_known_with_rt/00171/00004.JPEG"
        for entry_id, data in repo_data.items():
            path = data.get("img_path", "")
            label = data.get("label")

            # Extract the parent folder name (e.g., "00171" from path)
            # We assume the structure is .../FolderID/ImageName
            parts = path.replace("\\", "/").split("/")
            if len(parts) >= 2:
                folder_id = parts[-2]  # The folder containing the image

                # Verify it looks like an ID (e.g., "00171")
                if folder_id.isdigit():
                    folder_to_target_label[folder_id] = label

        print(f"Learned {len(folder_to_target_label)} folder-to-label mappings.")
        # Example check:
        if "00004" in folder_to_target_label:
            print(f"   > Verification: Folder '00004' maps to Label {folder_to_target_label['00004']}")

    except Exception as e:
        print(f"Error reading repo json: {e}")
        return

    # --- STEP 2: Map Synsets to those Target Labels ---
    print(f"\nReading {MAPPING_JSON} to link Synsets to Folders...")
    synset_to_target_label = {}

    try:
        with open(MAPPING_JSON, 'r') as f:
            pulkit_mapping = json.load(f)

        mapped_count = 0
        for folder_key, synsets in pulkit_mapping.items():
            # folder_key is like "00001", "00004"

            if folder_key in folder_to_target_label:
                target_label = folder_to_target_label[folder_key]

                for synset in synsets:
                    synset_to_target_label[synset] = target_label
                    mapped_count += 1
            else:
                # This happens if Pulkit's JSON has keys not used in the Repo's known_known set
                pass

        print(f"Linked {mapped_count} Synsets to Target Labels.")

    except Exception as e:
        print(f"Error reading mapping json: {e}")
        return

    # --- STEP 3: Load the 1k Synset Bridge ---
    print(f"\nLoading 1k Synset Bridge from {SYNSET_FILE}...")
    synsets_1k_list = []
    with open(SYNSET_FILE, 'r') as f:
        synsets_1k_list = [line.strip().split()[0] for line in f if line.strip()]

    # --- STEP 4: Generate the List ---
    print(f"\nProcessing {IMGLIST_1K}...")
    kept = 0
    dropped = 0

    with open(IMGLIST_1K, 'r') as f_in, open(OUTPUT_IMGLIST, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) < 2: continue

            img_path = parts[0]
            old_label = int(parts[1])

            # Retrieve Synset
            if old_label < len(synsets_1k_list):
                synset = synsets_1k_list[old_label]

                # Retrieve New Target Label
                if synset in synset_to_target_label:
                    new_label = synset_to_target_label[synset]
                    f_out.write(f"{img_path} {new_label}\n")
                    kept += 1
                else:
                    dropped += 1
            else:
                dropped += 1

    print("-" * 30)
    print(f"Done. Saved to {OUTPUT_IMGLIST}")
    print(f"Kept: {kept}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    generate_corrected_list()