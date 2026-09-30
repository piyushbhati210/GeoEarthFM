# ============================================================
# GEOEARTHFM — PROFESSIONAL STREAMLIT DASHBOARD (research prototype)
# Place at: D:\Piyush Work\GeoEarthFM\dashboard\app.py
# Reuses existing models, checkpoints, preprocessing and outputs.
# ============================================================

from pathlib import Path
import io
import re
import sys
import traceback

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoearthfm.config import *  # noqa: F401,F403
from geoearthfm.models import LandCoverModel, ChangeDetector
from geoearthfm.utils import load_checkpoint, device, normalize_spectral, to_chw

OUT = Path(globals().get("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
TEST_DIR = OUT / "all_test_pairs"
TILE = int(globals().get("IMAGE_SIZE", 64))
BAND_IDX = list(globals().get("STTORM_BAND_INDICES", []))
DEV = device()

DEFAULT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# class key -> (friendly label, group)
LABELS = {
    "annualcrop": ("🌾 Agricultural Crop", "agri"),
    "forest": ("🌳 Forest", "veg"),
    "herbaceousvegetation": ("🌿 Vegetation", "veg"),
    "highway": ("🛣️ Road / Highway", "built"),
    "industrial": ("🏭 Industrial Area", "built"),
    "pasture": ("🐄 Pasture", "veg"),
    "permanentcrop": ("🌱 Permanent Agriculture", "agri"),
    "residential": ("🏠 Residential Area", "built"),
    "river": ("🌊 River / Water", "water"),
    "sealake": ("🌊 Lake / Water", "water"),
}

st.set_page_config(page_title="GeoEarthFM | GeoAI", page_icon="🌍", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{padding-top:1.5rem;padding-bottom:3rem;max-width:1500px}
.hero{background:linear-gradient(135deg,#101b31,#142653);padding:30px;border-radius:18px;
      border:1px solid #263b63;margin-bottom:26px}
.hero h1{font-size:40px;margin-bottom:6px}.hero p{font-size:17px;color:#b9c8e6;margin:0}
.metric-card{background:#121a28;border:1px solid #293a59;border-radius:15px;padding:20px;
             text-align:center;min-height:115px}
.metric-title{color:#9eacc4;font-size:15px;margin-bottom:8px}
.metric-value{color:#fff;font-size:28px;font-weight:700}
.info-card{background:#111a29;border:1px solid #293a59;border-radius:15px;padding:20px;margin-bottom:16px}
.result-box{background:#11251c;border:1px solid #225e3c;border-radius:15px;padding:20px;margin:14px 0}
.warning-box{background:#2a2110;border:1px solid #765b1c;border-radius:15px;padding:18px;margin:14px 0}
.flow{text-align:center;line-height:2.1;font-size:17px}
.section-title{font-size:26px;font-weight:700;margin:18px 0 14px}
</style>""", unsafe_allow_html=True)


class UserError(Exception):
    """Problem the user can fix (shown without a traceback)."""


# ============================================================
# HELPERS
# ============================================================
def render_metric_card(title, value):
    st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div>'
                f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def show_error(msg, exc=None):
    st.error(f"⚠️ {msg}")
    if exc is not None:
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)))


def safe_read_csv(path):
    try:
        return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def find_metric(df, name):
    """Supports long (Metric,Value) and one-row wide CSV formats."""
    if df.empty:
        return None
    try:
        if {"Metric", "Value"} <= set(df.columns):
            m = df[df["Metric"].astype(str).str.strip().str.lower() == name.lower()]
            if not m.empty:
                return float(m.iloc[0]["Value"])
        for c in df.columns:
            if str(c).strip().lower() == name.lower():
                return float(df.iloc[0][c])
    except Exception:
        pass
    return None


def load_metrics():
    df = safe_read_csv(TEST_DIR / "final_overall_summary.csv")
    return {n: find_metric(df, n) for n in ["Accuracy", "Precision", "Recall", "Dice", "IoU"]}


def pct(v):
    return "N/A" if v is None else f"{v * 100:.2f}%"


def lc_info(cls):
    return LABELS.get(re.sub(r"[^a-z]", "", str(cls).lower()), (f"❓ {cls}", "other"))


def interpret_transition(before, after):
    (_, gb), (_, ga) = lc_info(before), lc_info(after)
    if before == after:
        return "Change detected, land-cover class unchanged"
    if ga == "water" and gb != "water":
        return "Possible water expansion / inundation (needs validation)"
    if gb == "water" and ga != "water":
        return "Possible water reduction"
    if gb == "veg" and ga in ("agri", "built"):
        return "Possible vegetation loss / land-use conversion"
    if ga == "built" and gb != "built":
        return "Possible built-up expansion"
    if gb == "built" and ga != "built":
        return "Possible built-up reduction"
    if gb == "agri" and ga == "veg":
        return "Possible agricultural reduction / vegetation regrowth"
    return "Land-cover transition"


def normalize_for_display(arr):
    arr = np.asarray(arr).astype(np.float32)
    lo, hi = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    if hi <= lo:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def to_hwc(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] <= 20 < x.shape[-1]:
        x = np.transpose(x, (1, 2, 0))
    return x


def make_rgb_preview(arr):
    """Quick-look only: first three channels (band order NOT verified as RGB)."""
    arr = to_hwc(arr)
    if arr.ndim == 2:
        g = normalize_for_display(arr)
        return Image.fromarray(np.stack([g] * 3, -1))
    if arr.ndim != 3:
        raise UserError(f"Unsupported array shape {arr.shape}.")
    ch = arr[:, :, :3] if arr.shape[-1] >= 3 else np.repeat(arr[:, :, :1], 3, -1)
    return Image.fromarray(normalize_for_display(ch))


def resize_mask(mask, size):
    img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(size, Image.Resampling.NEAREST)
    return np.array(img) > 0


def create_change_overlay(image, mask):
    base = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype(np.uint8) * 120)).convert("L")
    red = Image.new("RGBA", base.size, (255, 40, 40, 0))
    red.putalpha(alpha)
    return Image.alpha_composite(base, red).convert("RGB")


def png_bytes(mask):
    buf = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255)).save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# MODELS
# ============================================================
@st.cache_resource
def load_models():
    change, change_err = None, None
    ckpt = OUT / "change_best.pt"
    if not ckpt.exists():
        change_err = f"Checkpoint not found: {ckpt}"
    else:
        try:
            change = ChangeDetector().to(DEV)
            load_checkpoint(change, ckpt, map_location=DEV)
            change.eval()
        except Exception as e:
            change, change_err = None, str(e)

    lc, classes, lc_err = None, DEFAULT_CLASSES, None
    ckpt = OUT / "landcover_best.pt"
    if not ckpt.exists():
        lc_err = f"Checkpoint not found: {ckpt}"
    else:
        try:
            lc = LandCoverModel(len(DEFAULT_CLASSES)).to(DEV)
            payload = load_checkpoint(lc, ckpt, map_location=DEV)
            if isinstance(payload, dict) and payload.get("classes"):
                classes = list(payload["classes"])
            lc.eval()
        except Exception as e:
            lc, lc_err = None, str(e)
    return {"change": change, "change_err": change_err, "lc": lc, "classes": classes, "lc_err": lc_err}


# ============================================================
# CHANGE DETECTION
# ============================================================
def prepare_spectral(x):
    x = to_hwc(x)
    if x.ndim != 3:
        raise UserError("Spectral .npy must be a 3-D array (height × width × bands).")
    n = x.shape[-1]
    if BAND_IDX and max(BAND_IDX) < n:
        x = x[:, :, BAND_IDX]
    elif BAND_IDX and n != len(BAND_IDX):
        raise UserError(f"Required spectral bands are missing: file has {n} bands, "
                        f"the model needs band indices {BAND_IDX}.")
    return x


def validate_pair(b, a):
    b, a = to_hwc(b), to_hwc(a)
    if b.shape != a.shape:
        raise UserError("Before and after observations have different dimensions "
                        f"({b.shape} vs {a.shape}). Please upload observations covering the same area.")
    for n, x in (("Before", b), ("After", a)):
        if not np.isfinite(x).all():
            raise UserError(f"{n} image contains NaN or infinite values.")
    return b, a


def run_change_detection(model, before, after):
    x1 = torch.from_numpy(to_chw(normalize_spectral(prepare_spectral(before)))).unsqueeze(0).to(DEV)
    x2 = torch.from_numpy(to_chw(normalize_spectral(prepare_spectral(after)))).unsqueeze(0).to(DEV)
    with torch.no_grad():
        return torch.sigmoid(model(x1, x2))[0, 0].cpu().numpy()


def calculate_change_statistics(prob):
    mask = prob > 0.5
    total, changed = int(mask.size), int(mask.sum())
    # Model-derived confidence: mean predicted probability of the class chosen per pixel.
    conf_change = float(prob[mask].mean()) if changed else None
    conf_nochange = float((1 - prob[~mask]).mean()) if changed < total else None
    return {"mask": mask, "total": total, "changed": changed, "unchanged": total - changed,
            "ratio": changed / max(total, 1) * 100, "conf_change": conf_change,
            "conf_nochange": conf_nochange}


def conf_level(c):
    return "High" if c >= 0.85 else "Moderate" if c >= 0.65 else "Low"


# ============================================================
# LAND COVER
# ============================================================
def lc_transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((TILE, TILE)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def run_landcover_prediction(lc, images):
    """Return softmax probabilities [N, classes] for a list of PIL images."""
    tf, out = lc_transform(), []
    for i in range(0, len(images), 64):
        batch = torch.stack([tf(im.convert("RGB")) for im in images[i:i + 64]]).to(DEV)
        with torch.no_grad():
            logits, _ = lc(batch)
            out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def semantic_transitions(lc, classes, b_img, a_img, mask, min_frac=0.25):
    """Tile-based Before->After land-cover transitions over changed tiles only."""
    H, W = mask.shape
    boxes = [(x, y, x + TILE, y + TILE) for y in range(0, H - TILE + 1, TILE)
             for x in range(0, W - TILE + 1, TILE) if mask[y:y + TILE, x:x + TILE].mean() >= min_frac]
    if not boxes:
        return None
    pb = run_landcover_prediction(lc, [b_img.crop(b) for b in boxes])
    pa = run_landcover_prediction(lc, [a_img.crop(b) for b in boxes])
    agg = {}
    for box, p1, p2 in zip(boxes, pb, pa):
        c1, c2 = int(p1.argmax()), int(p2.argmax())
        px = int(mask[box[1]:box[3], box[0]:box[2]].sum())
        e = agg.setdefault((c1, c2), [0, [], []])
        e[0] += px; e[1].append(p1[c1]); e[2].append(p2[c2])
    rows = []
    for (c1, c2), (px, f1, f2) in agg.items():
        n1, n2 = classes[c1], classes[c2]
        rows.append({"Before": lc_info(n1)[0], "After": lc_info(n2)[0],
                     "Interpretation": interpret_transition(n1, n2),
                     "Approx. area (% of image)": round(px / mask.size * 100, 2),
                     "Before confidence %": round(float(np.mean(f1)) * 100, 1),
                     "After confidence %": round(float(np.mean(f2)) * 100, 1),
                     "_same": c1 == c2})
    return pd.DataFrame(rows).sort_values("Approx. area (% of image)", ascending=False)


# ============================================================
# RESULT RENDERING
# ============================================================
def build_summary_text(r):
    s = r["stats"]
    lines = ["GEOEARTHFM ANALYSIS (research prototype)", "",
             f"Total pixels: {s['total']:,}", f"Changed pixels: {s['changed']:,}",
             f"Unchanged pixels: {s['unchanged']:,}", f"Changed area: {s['ratio']:.2f}% of image",
             "Change confidence (mean model probability on changed pixels): "
             + ("N/A" if s["conf_change"] is None else f"{s['conf_change'] * 100:.1f}%"), ""]
    t = r["transitions"]
    if t is None:
        lines.append("Semantic land-cover transition: not determined from the available data.")
    else:
        lines.append("Land-cover transitions (interpretations, not ground truth):")
        for _, row in t.iterrows():
            lines.append(f"- {row['Before']} -> {row['After']}: {row['Interpretation']} "
                         f"({row['Approx. area (% of image)']}% of image)")
    return "\n".join(lines)


def render_result_card(r):
    s, t = r["stats"], r["transitions"]
    st.markdown("### 🔍 GeoEarthFM Analysis")
    c1, c2, c3 = st.columns(3)
    with c1: render_metric_card("Overall change", f"{s['ratio']:.2f}%")
    with c2: render_metric_card("Changed pixels", f"{s['changed']:,}")
    with c3: render_metric_card("Unchanged pixels", f"{s['unchanged']:,}")

    st.markdown("### 🧠 What changed?")
    diff = t[~t["_same"]] if t is not None else None
    if s["changed"] == 0:
        body = "<h3>No significant change detected.</h3>"
    elif diff is not None and not diff.empty:
        top = diff.iloc[0]
        body = (f"<h2>{top['Before']} → {top['After']}</h2><p>{top['Interpretation']}.</p>"
                f"<p>Approx. changed area for this transition: <b>{top['Approx. area (% of image)']}%</b> of the image.<br>"
                f"Before confidence: {top['Before confidence %']}% | After confidence: {top['After confidence %']}%</p>")
    else:
        body = ("<h3>Change detected, but the semantic land-cover transition could not be determined "
                "from the available data.</h3><p>Provide aligned RGB Before/After images in the optional "
                "section above to enable land-cover comparison.</p>")
    st.markdown(f'<div class="result-box">{body}</div>', unsafe_allow_html=True)

    if t is not None:
        st.markdown("#### Before → After transitions")
        st.dataframe(t.drop(columns="_same"), use_container_width=True, hide_index=True)

    st.markdown("### 🎯 Model confidence")
    if s["conf_change"] is None:
        st.info("Not available: no pixels were flagged as changed.")
    else:
        st.write(f"Change detection confidence: **{s['conf_change'] * 100:.1f}% ({conf_level(s['conf_change'])})** "
                 "— mean model probability over pixels flagged as changed. This is the model's own "
                 "output score, not a calibrated probability.")
        st.progress(min(max(s["conf_change"], 0.0), 1.0))

    st.markdown("### 📊 Change statistics")
    st.dataframe(pd.DataFrame({
        "Statistic": ["Total pixels", "Changed pixels", "Unchanged pixels", "Changed area", "Unchanged area"],
        "Value": [f"{s['total']:,}", f"{s['changed']:,}", f"{s['unchanged']:,}",
                  f"{s['ratio']:.2f}% of image", f"{100 - s['ratio']:.2f}% of image"]}),
        use_container_width=True, hide_index=True)
    st.caption("Spatial resolution is not known for uploaded files, so area is shown as % of image, not km².")

    st.markdown("### ⬇️ Download results")
    d1, d2 = st.columns(2)
    d1.download_button("Change mask (PNG)", png_bytes(r["disp_mask"]), "geoearthfm_change_mask.png", "image/png")
    d2.download_button("Analysis summary (TXT)", build_summary_text(r), "geoearthfm_analysis.txt", "text/plain")

    with st.expander("⚠️ Model limitations"):
        st.markdown(
            "This is a **research prototype**. Results depend on image quality, spectral bands, geographic "
            "region, temporal difference, training data and image alignment.\n\n"
            "Semantic interpretations such as “flooding”, “urban expansion” or “agricultural loss” are "
            "derived from model outputs and must not be treated as ground truth without further validation.")


# ============================================================
# PAGES
# ============================================================
def page_overview():
    st.markdown('<div class="hero"><h1>🌍 GeoEarthFM</h1><p><b>GeoAI Foundation Model for Earth Observation</b><br>'
                'Satellite image understanding, multitemporal change detection, land-cover classification '
                'and Earth observation intelligence.</p></div>', unsafe_allow_html=True)
    st.markdown("### What does GeoEarthFM do?")
    st.write("GeoEarthFM compares satellite observations of the same geographic area across different times "
             "and identifies areas where the land surface has changed.")
    st.markdown('<div class="info-card flow">🛰️ Satellite Image<br>↓<br>GeoEarthFM Representation<br>↓<br>'
                'Land-Cover Understanding<br>↓<br>Temporal Comparison<br>↓<br>Change Detection<br>↓<br>'
                'Human-Readable Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    m = load_metrics()
    cols = st.columns(5)
    for col, (n, label) in zip(cols, [("Accuracy", "Accuracy"), ("Precision", "Precision"),
                                      ("Recall", "Recall"), ("Dice", "Dice / F1"), ("IoU", "IoU")]):
        with col:
            render_metric_card(label, pct(m[n]))
    if m["Accuracy"] is not None and m["Dice"] is not None:
        st.caption("Accuracy is high because unchanged pixels dominate; Dice, IoU, Precision and Recall "
                   "better reflect change-detection quality.")
    pairs = safe_read_csv(TEST_DIR / "all_test_pair_results.csv")
    rc = next((c for c in pairs.columns if "region" in str(c).lower()), None)
    c1, c2, c3 = st.columns(3)
    with c1: render_metric_card("Temporal pairs", f"{len(pairs)}" if not pairs.empty else "N/A")
    with c2: render_metric_card("Regions", f"{pairs[rc].nunique()}" if rc else "N/A")
    with c3: render_metric_card("Evaluation status", "Results loaded" if not pairs.empty else "Not available")


def page_change(models):
    st.markdown("## 🔄 Change Detection Explorer")
    st.write("Upload two observations of the same geographic area. GeoEarthFM compares them and "
             "identifies where the land surface changed.")
    st.info("**Required format:** spectral `.npy` arrays containing the bands the model was trained on. "
            "RGB images (PNG/JPG/TIFF) cannot be used directly with this change detector because it needs "
            "specific spectral bands, so no conversion is attempted.")
    if models["change"] is None:
        show_error(f"Change detection model could not be loaded. {models['change_err']}")
        return

    st.markdown("### Steps 1–2: Upload observations")
    c1, c2 = st.columns(2)
    bf = c1.file_uploader("Before observation (.npy)", type=["npy"], key="before_npy")
    af = c2.file_uploader("After observation (.npy)", type=["npy"], key="after_npy")

    with st.expander("Optional: enable Before → After land-cover transitions"):
        st.write("Upload aligned RGB images of the **same area and same size** as the spectral data. "
                 "The EuroSAT-trained classifier is applied to tiles that the change detector flagged.")
        r1, r2 = st.columns(2)
        brgb = r1.file_uploader("Before RGB", type=["png", "jpg", "jpeg"], key="before_rgb")
        argb = r2.file_uploader("After RGB", type=["png", "jpg", "jpeg"], key="after_rgb")
        use_npy = st.checkbox("No RGB files: use the first three channels of the .npy files (experimental — "
                              "band order is not verified, so results may be unreliable)")

    if not (bf and af):
        st.caption("Upload both files to continue.")
    else:
        try:
            before, after = validate_pair(np.load(io.BytesIO(bf.getvalue())), np.load(io.BytesIO(af.getvalue())))
            st.markdown("### Step 3: Preview")
            p1, p2 = st.columns(2)
            p1.image(make_rgb_preview(before), caption=f"BEFORE {before.shape}", use_container_width=True)
            p2.image(make_rgb_preview(after), caption=f"AFTER {after.shape}", use_container_width=True)
        except UserError as e:
            show_error(str(e))
            return
        except Exception as e:
            show_error("Unable to read these files. Please make sure both are valid `.npy` arrays.", e)
            return

        if st.button("🔍 Analyze Change", type="primary"):
            try:
                with st.spinner("Running GeoEarthFM analysis..."):
                    prob = run_change_detection(models["change"], before, after)
                    stats = calculate_change_statistics(prob)
                    b_prev, a_prev = make_rgb_preview(before), make_rgb_preview(after)
                    disp_mask = resize_mask(stats["mask"], b_prev.size)
                    trans = None
                    if models["lc"] is not None:
                        b_lc = Image.open(io.BytesIO(brgb.getvalue())).convert("RGB") if brgb else (b_prev if use_npy else None)
                        a_lc = Image.open(io.BytesIO(argb.getvalue())).convert("RGB") if argb else (a_prev if use_npy else None)
                        if b_lc is not None and a_lc is not None:
                            if b_lc.size != a_lc.size:
                                raise UserError("Before and after RGB images must have the same size.")
                            trans = semantic_transitions(models["lc"], models["classes"], b_lc, a_lc,
                                                         resize_mask(stats["mask"], b_lc.size))
                st.session_state["result"] = {
                    "stats": stats, "prob": prob, "b": b_prev, "a": a_prev, "disp_mask": disp_mask,
                    "overlay": create_change_overlay(a_prev, disp_mask), "transitions": trans}
            except UserError as e:
                show_error(str(e))
            except Exception as e:
                show_error("Unable to process the uploaded data. Check that Before/After cover the same area, "
                           "use the required format, and contain the required spectral bands.", e)

    r = st.session_state.get("result")
    if r:
        st.markdown("### Steps 4–5: Change map")
        tabs = st.tabs(["Original", "Before", "After", "Change Mask", "Overlay"])
        with tabs[0]:
            o1, o2 = st.columns(2)
            o1.image(r["b"], caption="Before", use_container_width=True)
            o2.image(r["a"], caption="After", use_container_width=True)
        tabs[1].image(r["b"], use_container_width=True)
        tabs[2].image(r["a"], use_container_width=True)
        tabs[3].image(r["disp_mask"], caption="Detected change (white)", use_container_width=True)
        tabs[4].image(r["overlay"], caption="Change overlay on After image", use_container_width=True)
        st.markdown("**Legend:** 🟥 Changed area (translucent overlay) &nbsp;|&nbsp; ⬜ No significant change "
                    "(original image shown). Red marks model-detected change only; it does not encode land-cover type.")
        render_result_card(r)
    if models["lc"] is None:
        st.caption(f"Land-cover model unavailable, so semantic transitions are disabled. {models['lc_err'] or ''}")


def page_landcover(models):
    st.markdown("## 🌱 Land Cover Classification")
    st.write("Upload an RGB satellite image (EuroSAT-style patch) to predict its land-cover category.")
    if models["lc"] is None:
        show_error(f"Land-cover model could not be loaded. {models['lc_err']}")
        return
    up = st.file_uploader("RGB satellite image", type=["png", "jpg", "jpeg"], key="lc_img")
    if not up:
        return
    try:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Input image", width=450)
        probs = run_landcover_prediction(models["lc"], [img])[0]
        classes, order = models["classes"], np.argsort(-probs)
        label = lc_info(classes[order[0]])[0]
        c1, c2 = st.columns(2)
        with c1: render_metric_card("Predicted class", label)
        with c2: render_metric_card("Confidence", f"{probs[order[0]] * 100:.1f}%")
        st.markdown("#### Top 3 predictions")
        st.dataframe(pd.DataFrame({"Class": [lc_info(classes[i])[0] for i in order[:3]],
                                   "Probability %": [round(float(probs[i]) * 100, 1) for i in order[:3]]}),
                     use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"Probability %": probs * 100},
                                  index=[lc_info(c)[0] for c in classes]))
        st.caption("Before/After land-cover comparison is available on the Change Detection page.")
    except Exception as e:
        show_error("Land-cover prediction failed for this image.", e)


def page_region():
    st.markdown("## 📍 Region Analysis")
    df = safe_read_csv(TEST_DIR / "final_region_summary.csv")
    if df.empty:
        df = safe_read_csv(TEST_DIR / "region_summary.csv")
    st.markdown("### Test-set performance by region")
    if df.empty:
        st.info("Not available: region summary CSV not found in outputs/all_test_pairs.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        rc = next((c for c in df.columns if "region" in str(c).lower()), df.columns[0])
        num = [c for c in df.columns if c != rc and pd.api.types.is_numeric_dtype(df[c])]
        if num:
            st.bar_chart(df.set_index(rc)[num])
    st.markdown("### Latest analyzed image")
    r = st.session_state.get("result")
    if not r:
        st.info("Run an analysis on the Change Detection page to inspect an uploaded image here.")
        return
    mask = r["stats"]["mask"]
    rows = [[float(b.mean()) * 100 for b in np.array_split(rb, 4, axis=1)]
            for rb in np.array_split(mask, 4, axis=0)]
    st.write("Change percentage per grid cell (4×4, top-left to bottom-right):")
    st.dataframe(pd.DataFrame(rows, columns=[f"Col {i + 1}" for i in range(4)],
                              index=[f"Row {i + 1}" for i in range(4)]).round(2))
    st.write(f"Changed pixels: **{r['stats']['changed']:,}** | Change: **{r['stats']['ratio']:.2f}%**")
    st.caption("Latitude/longitude and physical area: not available from current input (.npy files carry no georeference).")


def page_temporal():
    st.markdown("## 📅 Temporal Analysis")
    df = safe_read_csv(TEST_DIR / "temporal_interval_summary.csv")
    if df.empty:
        st.info("Not available: temporal_interval_summary.csv not found.")
    else:
        dcol = next((c for c in df.columns if "day" in str(c).lower()), None)
        num = [c for c in df.columns if c != dcol and pd.api.types.is_numeric_dtype(df[c])
               and str(c).lower() in ("dice", "iou", "f1")]
        st.markdown("### Performance vs. time between observations")
        if dcol and num:
            st.line_chart(df.sort_values(dcol).set_index(dcol)[num])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("Short intervals scored lower in the test set, but the trend is not strictly monotonic and "
                   "interval is partly confounded with region.")
    st.markdown("### Land-cover distribution over time")
    st.info("Not available from current input: no dated land-cover time series is stored. Dates are not invented.")


def page_visual():
    st.markdown("## 🖼️ Visual Results")
    if not OUT.exists():
        st.info("Outputs folder not found.")
        return
    files = sorted(OUT.rglob("*.png"))[:500]
    if not files:
        st.info("No result images found in the outputs folder.")
        return
    folders = sorted({str(f.parent.relative_to(OUT)) for f in files})
    folder = st.selectbox("Folder", folders, index=max(folders.index("all_test_pairs/final_analysis"), 0)
                          if "all_test_pairs/final_analysis" in folders else 0)
    here = [f for f in files if str(f.parent.relative_to(OUT)) == folder]
    conf = [f for f in files if "confusion" in f.name.lower()]
    if conf:
        st.caption(f"{len(conf)} confusion-matrix image(s) found; select them by folder/file name.")
    pick = st.selectbox("Image", here, format_func=lambda p: p.name)
    try:
        st.image(str(pick), caption=pick.name, use_container_width=True)
    except Exception as e:
        show_error("This image could not be displayed.", e)


def page_tests():
    st.markdown("## 🧪 Test Results")
    m = load_metrics()
    if all(v is None for v in m.values()):
        st.info("Not available: outputs/all_test_pairs/final_overall_summary.csv not found.")
    else:
        cols = st.columns(5)
        for col, (k, v) in zip(cols, m.items()):
            with col: render_metric_card(k, pct(v))
        tbl = pd.DataFrame({"Metric": list(m), "Value": [v for v in m.values()]}).dropna()
        c1, c2 = st.columns([1, 2])
        c1.dataframe(tbl, use_container_width=True, hide_index=True)
        c2.bar_chart(tbl.set_index("Metric"))
    for title, f in [("Region summary", "final_region_summary.csv"), ("Top 5 temporal pairs", "final_top_5_temporal_pairs.csv"),
                     ("Lowest 5 temporal pairs", "final_lowest_5_temporal_pairs.csv"),
                     ("All pair results", "all_test_pair_results.csv")]:
        d = safe_read_csv(TEST_DIR / f)
        with st.expander(title, expanded=False):
            st.dataframe(d, use_container_width=True, hide_index=True) if not d.empty else st.info(f"Not found: {f}")


def page_about():
    st.markdown("## ℹ️ About GeoEarthFM")
    st.write("GeoEarthFM is a prototype GeoAI system designed to understand Earth observation imagery and "
             "detect changes between satellite observations collected at different times.")
    st.markdown("**Tasks**\n- Land-cover classification\n- Multitemporal change detection\n"
                "- Satellite representation learning\n- Visual change analysis\n- Model evaluation")
    st.markdown("**Data:** STTORM-CD-Floods (change detection) and EuroSAT-based land-cover data.")
    st.markdown("### Real-time inference, not live satellite data")
    st.write("GeoEarthFM currently performs real-time inference on uploaded satellite observations. It does not "
             "automatically retrieve live satellite imagery from external satellite APIs.")
    st.markdown("### Research prototype")
    st.warning("This is a research prototype, not a production satellite monitoring system. A change detector "
               "says *where* something changed; *what* changed comes from the separate land-cover classifier "
               "and is an interpretation that needs validation.")


# ============================================================
# MAIN
# ============================================================
PAGES = {"Overview": "🏠 Overview", "Change Detection": "🔄 Change Detection",
         "Land Cover Classification": "🌱 Land Cover Classification", "Region Analysis": "📍 Region Analysis",
         "Temporal Analysis": "📅 Temporal Analysis", "Visual Results": "🖼️ Visual Results",
         "Test Results": "🧪 Test Results", "About": "ℹ️ About"}

with st.sidebar:
    st.markdown("# 🌍 GeoEarthFM")
    st.markdown("---")
    choice = st.radio("Navigation", list(PAGES), format_func=lambda k: PAGES[k])
    st.markdown("---")
    st.caption(f"Device: {DEV}")

try:
    MODELS = load_models()
    if choice == "Overview": page_overview()
    elif choice == "Change Detection": page_change(MODELS)
    elif choice == "Land Cover Classification": page_landcover(MODELS)
    elif choice == "Region Analysis": page_region()
    elif choice == "Temporal Analysis": page_temporal()
    elif choice == "Visual Results": page_visual()
    elif choice == "Test Results": page_tests()
    else: page_about()
except Exception as e:
    show_error("Something went wrong while rendering this page.", e)

st.markdown("---")
st.caption("GeoEarthFM | Research prototype | Multitemporal Earth Observation")