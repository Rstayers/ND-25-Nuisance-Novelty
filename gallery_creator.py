import pandas as pd
from pathlib import Path
import shutil
from torchvision import models

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Nuisance Novelty Gallery</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f9f9f9;
            margin: 20px;
        }}
        h1 {{ text-align: center; }}
        .filters {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .filters button {{
            margin: 0 5px;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
            cursor: pointer;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            text-align: center;
            padding: 8px;
        }}
        .card img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .caption {{
            margin-top: 6px;
            font-size: 14px;
            text-align: left;
        }}
    </style>
</head>
<body>
    <h1>Nuisance Novelty Gallery</h1>
    <div class="filters">
        {filter_buttons}
    </div>
    <div class="gallery" id="gallery">
        {cards}
    </div>

    <script>
        function filterCards(dataset) {{
            let cards = document.getElementsByClassName('card');
            for (let i = 0; i < cards.length; i++) {{
                let db = cards[i].getAttribute('data-dataset');
                if (dataset === 'all' || db === dataset) {{
                    cards[i].style.display = 'block';
                }} else {{
                    cards[i].style.display = 'none';
                }}
            }}
        }}
    </script>
</body>
</html>
"""

CARD_TEMPLATE = """
<div class="card" data-dataset="{dataset}">
    <img src="{filename}" alt="{filename}">
    <div class="caption">
        <div><b>Dataset:</b> {dataset}</div>
        <div><b>GT:</b> {ground_truth} | <b>Pred:</b> {prediction}</div>
        <div><b>Score:</b> {conf:.4f}</div>
        <div><b>Error Type:</b> {error_type}</div>
    </div>
</div>
"""

def make_gallery(csv_path, out_html, class_names=None, copy_images=True):
    df = pd.read_csv(csv_path)

    # Only nuisance novelty rows
    df = df[df["is_nn"] == 1]

    out_dir = Path(out_html).parent
    if copy_images:
        img_out_dir = out_dir / "gallery_imgs"
        img_out_dir.mkdir(parents=True, exist_ok=True)
    else:
        img_out_dir = None

    # Create filter buttons for each dataset
    datasets = sorted(df["dataset"].unique())
    filter_buttons = '<button onclick="filterCards(\'all\')">Show All</button>\n'
    for ds in datasets:
        filter_buttons += f'<button onclick="filterCards(\'{ds}\')">{ds}</button>\n'

    cards = []
    for idx, row in df.iterrows():
        src_path = Path(row["image_path"])
        if copy_images:
            dst_path = img_out_dir / f"{idx}{src_path.suffix}"
            if not dst_path.exists() and src_path.exists():
                shutil.copy(src_path, dst_path)
            filename = f"gallery_imgs/{dst_path.name}"
        else:
            filename = str(src_path)

        gt_idx = int(row["class_id"])
        pred_idx = int(row["pred_class"])
        gt_str = class_names[gt_idx] if (class_names and gt_idx < len(class_names)) else str(gt_idx)
        pred_str = class_names[pred_idx] if (class_names and pred_idx < len(class_names)) else str(pred_idx)

        cards.append(CARD_TEMPLATE.format(
            dataset=row["dataset"],
            filename=filename,
            ground_truth=f"{gt_idx} ({gt_str})",
            prediction=f"{pred_idx} ({pred_str})",
            conf=row["score"],
            error_type=row["error_type"]
        ))

    html = HTML_TEMPLATE.format(cards="\n".join(cards), filter_buttons=filter_buttons)
    Path(out_html).write_text(html, encoding="utf-8")
    print(f"[Gallery saved] {out_html}  (rows: {len(df)})")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/nuisance_runs/nuisance_id_csid_4abda046.csv")
    ap.add_argument("--out_html", type=str, default="results/gallery.html")
    ap.add_argument("--no_copy", action="store_true",
                    help="If set, do not copy images, link directly to dataset paths.")
    args = ap.parse_args()

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    class_names = weights.meta["categories"]

    make_gallery(args.csv, args.out_html, class_names, copy_images=not args.no_copy)
