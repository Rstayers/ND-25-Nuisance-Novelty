#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static HTML gallery for "full/partial nuisance" images from a CSV.

- Writes `nn_gallery.html` with data inline (no external JSON).
- Copies images into `images/` (reuses existing files; stable hashed names).
- Filters in UI: case (Full/Partial/All), detector, backbone, corruption, severity,
  FPR column + value, plus text search.
- Card caption: GT label string + score.

ImageNet-1k labels:
- If torchvision is installed, uses its built-in ImageNet categories automatically.
- Else: uses any string label columns found in CSV; otherwise shows numeric class_id.
"""

import argparse
import os
import sys
import json
import html
import shutil
import hashlib
import re
import pandas as pd
from pathlib import Path

# ---------- File utilities ----------

def stable_name_from_path(src: Path, ext_fallback: str = ".jpg") -> str:
    norm = str(src.as_posix())
    digest = hashlib.md5(norm.encode("utf-8")).hexdigest()
    ext = src.suffix if src.suffix else ext_fallback
    return f"{digest}{ext}"

# ---------- ImageNet label source (no external JSON required) ----------

def get_imagenet1k_labels():
    try:
        import torchvision.models as tvm  # type: ignore
        try:
            labels = tvm.ResNet50_Weights.DEFAULT.meta.get("categories", None)
        except Exception:
            labels = None
        if not labels:
            try:
                labels = tvm.MobileNet_V2_Weights.DEFAULT.meta.get("categories", None)
            except Exception:
                labels = None
        if isinstance(labels, (list, tuple)) and len(labels) == 1000:
            return list(labels)
    except Exception:
        pass
    return None

IMAGENET_LABELS = get_imagenet1k_labels()

# ---------- Label & score coalescing ----------

ALPHA_RE = re.compile(r"[A-Za-z]")

def _looks_like_human_label(val: str) -> bool:
    if not isinstance(val, str):
        val = str(val)
    return bool(ALPHA_RE.search(val))

def _map_imagenet_label_from_index(idx: int):
    if IMAGENET_LABELS is None:
        return None
    if 0 <= idx < 1000:
        return IMAGENET_LABELS[idx]
    if 1 <= idx <= 1000 and 0 <= (idx - 1) < 1000:
        return IMAGENET_LABELS[idx - 1]
    return None

def coalesce_label(row: dict) -> str:
    for key in ["gt_label", "label", "class_name", "gt_class_name", "target", "gt_class"]:
        if key in row and pd.notna(row[key]):
            s = str(row[key]).strip()
            if s and _looks_like_human_label(s):
                return s
    for key in ["class_id", "target", "gt_class", "label", "gt_label"]:
        if key in row and pd.notna(row[key]):
            try:
                idx = int(str(row[key]).strip())
            except Exception:
                continue
            name = _map_imagenet_label_from_index(idx)
            if name is not None:
                return name
            return str(idx)
    return ""

def coalesce_score(row: dict):
    for key in ["score", "confidence", "prob", "logit"]:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                return str(row[key])
    return ""

def normalize_numeric_to_str(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        n = float(val)
        if abs(n - round(n)) < 1e-12:
            return str(int(round(n)))
        return f"{n:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(val)

# ---------- FPR discovery ----------

FPR_COL_REGEX = re.compile(r"^(?:fpr|FPR)[\w@]*", re.ASCII)

def find_fpr_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if FPR_COL_REGEX.match(str(c)):
            cols.append(str(c))
    return sorted(cols, key=lambda x: (len(x), x.lower()))

# ---------- HTML builder ----------

def build_html(page_title: str, records: list) -> str:
    data_json = json.dumps(records, ensure_ascii=False)
    css = r"""
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0d10; color:#d7dde5; }
    header { position:sticky; top:0; background:#12161a; padding:12px; border-bottom:1px solid #243040; z-index:10; }
    h1 { margin:0; font-size:18px; }
    .controls { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; align-items:center; }
    select, input[type="text"], button {
      background:#1e242b; color:#d7dde5; border:1px solid #243040;
      border-radius:6px; padding:6px 10px; font-size:13px;
    }
    button { cursor:pointer; }
    main { padding:16px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:12px; }
    .card { background:#12161a; border:1px solid #243040; border-radius:10px; overflow:hidden; display:flex; flex-direction:column; transition:transform .12s ease, box-shadow .12s ease; }
    .card:hover { transform:translateY(-2px); box-shadow:0 4px 10px rgba(0,0,0,.4); }
    .thumb { aspect-ratio:4/3; background:#0f1317; display:flex; align-items:center; justify-content:center; }
    .thumb img { width:100%; height:100%; object-fit:cover; }
    .meta { padding:8px; font-size:13px; display:flex; flex-direction:column; gap:6px; }
    .chip { background:#1e242b; border:1px solid #243040; border-radius:999px; padding:2px 8px; margin-right:4px; font-size:11px; }
    .row { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
    .label { font-weight:600; }
    .score { opacity:.9; }
    .empty { text-align:center; padding:40px; color:#9aa4af; }
    footer { margin-top:20px; padding:10px; font-size:12px; color:#9aa4af; border-top:1px solid #243040; }
    """
    js = r"""
    const state = {
      all:[], filtered:[],
      caseType:'full',  // 'full' | 'partial' | 'all'
      detector:'all', backbone:'all', corruption:'all', severity:'all',
      fprKey:'all', fprVal:'all',
      search:''
    };

    function unique(arr,key) {
      return Array.from(new Set(arr.map(x => (x[key] ?? '')).filter(v => v !== ''))).sort((a,b) => {
        const na = Number(a), nb = Number(b);
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
        return String(a).localeCompare(String(b));
      });
    }

    function uniqueFromSet(values) {
      return Array.from(new Set(values)).sort((a,b)=>{
        const na=Number(a), nb=Number(b);
        if(!Number.isNaN(na)&&!Number.isNaN(nb)) return na-nb;
        return String(a).localeCompare(String(b));
      });
    }

    function collectFprKeys() {
      const keys = new Set();
      for (const r of state.all) {
        if (r.fprs) for (const k of Object.keys(r.fprs)) keys.add(k);
      }
      return Array.from(keys).sort((a,b)=>a.localeCompare(b));
    }

    function collectFprValuesFor(key) {
      if (key==='all') return [];
      const vals = [];
      for (const r of state.all) {
        if (r.fprs && r.fprs[key]!=null && r.fprs[key]!=='') vals.push(String(r.fprs[key]));
      }
      return uniqueFromSet(vals);
    }

    function hydrateFilters() {
      const dets=['all',...unique(state.all,'detector')];
      const backs=['all',...unique(state.all,'backbone')];
      const corrs=['all',...unique(state.all,'corruption')];
      const sevs =['all',...unique(state.all,'severity')];

      const fprKeys = ['all', ...collectFprKeys()];
      const fprVals = state.fprKey==='all' ? ['all'] : ['all', ...collectFprValuesFor(state.fprKey)];

      document.getElementById('caseType').value    = state.caseType;
      document.getElementById('detector').innerHTML  = dets.map(v=>`<option value="${v}">${v}</option>`).join('');
      document.getElementById('backbone').innerHTML  = backs.map(v=>`<option value="${v}">${v}</option>`).join('');
      document.getElementById('corruption').innerHTML= corrs.map(v=>`<option value="${v}">${v}</option>`).join('');
      document.getElementById('severity').innerHTML  = sevs.map(v=>`<option value="${v}">${v}</option>`).join('');

      document.getElementById('fprKey').innerHTML    = fprKeys.map(v=>`<option value="${v}">${v}</option>`).join('');
      document.getElementById('fprVal').innerHTML    = fprVals.map(v=>`<option value="${v}">${v}</option>`).join('');

      document.getElementById('detector').value  = state.detector;
      document.getElementById('backbone').value  = state.backbone;
      document.getElementById('corruption').value= state.corruption;
      document.getElementById('severity').value  = state.severity;

      if (![...fprKeys].includes(state.fprKey)) state.fprKey='all';
      document.getElementById('fprKey').value    = state.fprKey;

      if (state.fprKey==='all' || !collectFprValuesFor(state.fprKey).includes(state.fprVal)) {
        state.fprVal='all';
      }
      document.getElementById('fprVal').value    = state.fprVal;
    }

    function applyFilters() {
      const d=state.detector, b=state.backbone, c=state.corruption, s=state.severity, fk=state.fprKey, fv=state.fprVal, q=state.search.toLowerCase(), ct=state.caseType;
      state.filtered = state.all.filter(r=>{
        // Case filter
        if (ct==='full'    && r.case!=='full')    return false;
        if (ct==='partial' && r.case!=='partial') return false;

        // Other filters
        if(d!=='all' && r.detector!==d) return false;
        if(b!=='all' && r.backbone!==b) return false;
        if(c!=='all' && r.corruption!==c) return false;
        if(s!=='all' && String(r.severity)!==String(s)) return false;
        if(fk!=='all') {
          const val = r.fprs ? r.fprs[fk] : undefined;
          if (fv!=='all' && String(val)!==String(fv)) return false;
          if (fv==='all' && (val===undefined || val==='')) return false;
        }
        if(q) {
          const hay = `${r.gt_label||''} ${r.score==null?'':r.score} ${r.dataset||''} ${r.corruption||''} ${r.severity||''} ${r.error_type||''} ${r.image_path||''}`.toLowerCase();
          if(!hay.includes(q)) return false;
        }
        return true;
      });
      document.getElementById('count').textContent=state.filtered.length;
      renderGrid();
    }

    function fmtScore(v){
      if (v===null || v===undefined || v==='') return '—';
      const n = Number(v);
      if (Number.isNaN(n)) return String(v);
      return n.toFixed(4).replace(/\.?0+$/,'');
    }

    function renderGrid() {
      const grid=document.getElementById('grid');
      if(!state.filtered.length) { grid.innerHTML='<div class="empty">No images match your filters.</div>'; return; }
      grid.innerHTML = state.filtered.map(r=>{
        const gt = r.gt_label || '—';
        const sc = fmtScore(r.score);
        const img = r.image_path ? `<img src="${r.image_path}" alt="${gt}" loading="lazy" onerror="this.style.display='none'">`
                                 : `<div style="color:#9aa4af;font-size:12px;">(missing image)</div>`;
        return `<a class="card" href="${r.image_path || '#'}" ${r.image_path ? 'target="_blank"' : ''}>
                  <div class="thumb">${img}</div>
                  <div class="meta">
                    <div class="row">
                      <span class="chip">${r.detector||'—'}</span>
                      <span class="chip">${r.backbone||'—'}</span>
                    </div>
                    <div class="row">
                      <span class="label">${gt}</span>
                      <span class="score">• score: ${sc}</span>
                    </div>
                  </div>
                </a>`;
      }).join('');
    }

    function init() {
      const raw = document.getElementById('data-json').textContent;
      try { state.all = JSON.parse(raw); } catch(e){ console.error("Failed to parse data JSON",e); state.all=[]; }
      state.filtered = state.all.slice();
      document.getElementById('total').textContent=state.all.length;
      document.getElementById('count').textContent=state.filtered.length;
      hydrateFilters();
      renderGrid();

      document.getElementById('caseType').addEventListener('change',e=>{state.caseType=e.target.value;applyFilters();});
      document.getElementById('detector').addEventListener('change',e=>{state.detector=e.target.value;applyFilters();});
      document.getElementById('backbone').addEventListener('change',e=>{state.backbone=e.target.value;applyFilters();});
      document.getElementById('corruption').addEventListener('change',e=>{state.corruption=e.target.value;applyFilters();});
      document.getElementById('severity').addEventListener('change',e=>{state.severity=e.target.value;applyFilters();});

      document.getElementById('fprKey').addEventListener('change',e=>{
        state.fprKey=e.target.value;
        state.fprVal='all';
        hydrateFilters();
        applyFilters();
      });
      document.getElementById('fprVal').addEventListener('change',e=>{state.fprVal=e.target.value;applyFilters();});

      document.getElementById('search').addEventListener('input',e=>{state.search=e.target.value;applyFilters();});

      document.getElementById('reset').addEventListener('click',()=>{
        state.caseType='full';
        state.detector='all'; state.backbone='all'; state.corruption='all'; state.severity='all';
        state.fprKey='all'; state.fprVal='all'; state.search='';
        hydrateFilters(); document.getElementById('search').value='';
        applyFilters();
      });
    }

    document.addEventListener('DOMContentLoaded',init);
    """
    return f"""<!doctype html>
    <html><head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{html.escape(page_title)}</title>
    <style>{css}</style>
    </head>
    <body>
    <header>
      <h1>{html.escape(page_title)}</h1>
      <div class="controls">
        <label>Case <select id="caseType">
          <option value="full">Full nuisance</option>
          <option value="partial">Partial nuisance</option>
          <option value="all">All</option>
        </select></label>
        <label>Detector <select id="detector"></select></label>
        <label>Backbone <select id="backbone"></select></label>
        <label>Corruption <select id="corruption"></select></label>
        <label>Severity <select id="severity"></select></label>
        <label>FPR column <select id="fprKey"></select></label>
        <label>FPR value <select id="fprVal"></select></label>
        <input id="search" type="text" placeholder="Search..."/>
        <button id="reset">Reset</button>
        <span><strong id="count">0</strong>/<span id="total">0</span> images</span>
      </div>
    </header>
    <main>
      <div id="grid" class="grid"></div>
      <footer>Generated as static page. Data is embedded inline. Requires images/ folder in same directory.</footer>
    </main>
    <script id="data-json" type="application/json">{data_json}</script>
    <script>{js}</script>
    </body></html>"""

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="gallery_out", help="Output folder")
    ap.add_argument("--title", default="Nuisance Novelty — Image Gallery")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "image_path" not in df.columns:
        print("CSV missing image_path column", file=sys.stderr)
        sys.exit(3)

    fpr_cols = find_fpr_columns(df)

    os.makedirs(args.outdir, exist_ok=True)
    img_dir = Path(args.outdir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def compute_case(row) -> str:
        # full if is_nn == 1 (accept 1/True/"1"/"true"/"yes")
        is_nn_val = row.get("is_nn", None)
        full = False
        if pd.notna(is_nn_val):
            s = str(is_nn_val).strip().lower()
            full = s in {"1","true","yes"} or (s.isdigit() and int(float(s)) == 1)
        partial = str(row.get("error_type","")).strip().lower() == "Partial_Nuisance"
        if full: return "full"
        if partial: return "partial"
        return "other"

    records = []
    for _, row in df.iterrows():
        src = Path(str(row["image_path"]).replace("\\", "/"))
        dst_name = stable_name_from_path(src)
        dst = img_dir / dst_name

        new_path = ""
        try:
            if src.exists():
                if not dst.exists():
                    shutil.copy2(src, dst)
                new_path = f"images/{dst.name}"
        except Exception as e:
            print(f"Warning: failed to copy {src}: {e}", file=sys.stderr)
            new_path = ""

        # FPR fields
        fprs = {}
        for c in fpr_cols:
            if c in df.columns and pd.notna(row.get(c)):
                fprs[c] = normalize_numeric_to_str(row.get(c))

        rec = {
            # Case (new)
            "case":      compute_case(row),  # 'full' | 'partial' | 'other'
            # Filters / metadata
            "dataset":   row.get("dataset", ""),
            "corruption":row.get("corruption", ""),
            "severity":  row.get("severity", ""),
            "detector":  row.get("detector", ""),
            "backbone":  row.get("backbone", ""),
            "error_type":str(row.get("error_type","")).strip(),
            "fprs":      fprs,
            # Display
            "gt_label":  coalesce_label(row),
            "score":     coalesce_score(row),
            "image_path":new_path
        }
        records.append(rec)

    html_path = Path(args.outdir) / "nn_gallery.html"
    html_doc = build_html(args.title, records)
    html_path.write_text(html_doc, encoding="utf-8")

    copied = sum(1 for r in records if r["image_path"])
    print(f"[✓] Wrote {html_path} with {copied} images referenced in {img_dir} (existing files were reused).")
    if IMAGENET_LABELS is None:
        print("[i] torchvision not found; using CSV label strings or numeric class_id.")
    if not fpr_cols:
        print("[i] No FPR-like columns detected. The FPR controls will default to 'all'.")

if __name__ == "__main__":
    main()
