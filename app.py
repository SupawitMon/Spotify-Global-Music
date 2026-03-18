import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Spotify Popularity Predictor Pro",
    page_icon="🎵",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
METRICS_PATH = "models/metrics.json"
FI_PATH = "outputs/feature_importance.csv"
LOGO_PATH = "assets/spotify_logo.png"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    fi = pd.DataFrame()
    if os.path.exists(FI_PATH):
        fi = pd.read_csv(FI_PATH)

    return model, feature_columns, metrics, fi


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29,185,84,0.10), transparent 25%),
                radial-gradient(circle at top right, rgba(30,215,96,0.08), transparent 20%),
                linear-gradient(180deg, #0a0a0a 0%, #121212 55%, #0b0b0b 100%);
            color: white;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        .hero-box {
            background: linear-gradient(135deg, rgba(29,185,84,0.92), rgba(10,10,10,0.95));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 28px;
            padding: 26px 28px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.35);
            margin-bottom: 14px;
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.20rem;
            line-height: 1.05;
        }

        .hero-subtitle {
            color: rgba(255,255,255,0.92);
            font-size: 1rem;
            margin-top: 0.2rem;
        }

        .glass-card {
            background: rgba(24,24,24,0.90);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 22px;
            padding: 16px 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
        }

        .result-wrap {
            background: linear-gradient(135deg, rgba(29,185,84,0.18), rgba(20,20,20,0.95));
            border: 1px solid rgba(29,185,84,0.35);
            border-radius: 24px;
            padding: 20px;
            margin-top: 10px;
        }

        .pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: white;
            font-size: 0.84rem;
            margin-right: 8px;
            margin-top: 12px;
        }

        .tiny-note {
            color: rgba(255,255,255,0.72);
            font-size: 0.92rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(24,24,24,0.92);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 14px;
            border-radius: 18px;
        }

        div.stButton > button {
            background: #1DB954;
            color: black;
            border: none;
            border-radius: 999px;
            font-weight: 800;
            height: 3.2rem;
            font-size: 1rem;
            width: 100%;
        }

        div.stButton > button:hover {
            background: #22d760;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def popularity_label(score: float) -> str:
    if score >= 85:
        return "🌟 Smash Hit"
    elif score >= 70:
        return "🔥 High Potential"
    elif score >= 55:
        return "🎧 Good Potential"
    elif score >= 40:
        return "🙂 Mid Potential"
    return "🌫️ Low Potential"


def business_comment(score: float) -> str:
    if score >= 85:
        return "เพลงนี้มีโอกาสสูงมากที่จะเป็นกระแส เหมาะกับการโปรโมตเต็มแรง"
    elif score >= 70:
        return "เพลงนี้มีแนวโน้มดีมาก ถ้าปล่อยถูกจังหวะและโปรโมตเหมาะสมมีลุ้นไปไกล"
    elif score >= 55:
        return "เพลงนี้อยู่ระดับกลางค่อนดี มีศักยภาพ แต่ยังต้องอาศัย branding และฐานแฟนช่วย"
    elif score >= 40:
        return "เพลงนี้มีแนวโน้มกลาง ๆ ยังไม่เด่นมากในเชิงตลาดกว้าง"
    return "เพลงนี้อาจเหมาะกับกลุ่มเฉพาะมากกว่าตลาดกว้าง"


def score_color(score: float) -> str:
    if score >= 70:
        return "#1DB954"
    elif score >= 55:
        return "#f59e0b"
    return "#ef4444"


def render_hero(metrics):
    left, right = st.columns([4, 1.25], vertical_alignment="center")

    with left:
        st.markdown('<div class="hero-box">', unsafe_allow_html=True)

        logo_col, text_col = st.columns([0.8, 5], vertical_alignment="center")

        with logo_col:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=62)
            else:
                st.markdown(
                    '<div style="font-size:52px;line-height:1;text-align:center;">🎵</div>',
                    unsafe_allow_html=True
                )

        with text_col:
            st.markdown(
                '<div class="hero-title">Spotify Popularity Predictor</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="hero-subtitle">Predict track popularity from simple metadata with a clean Spotify-style interface.</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <span class="pill">ML Deployment</span>
                <span class="pill">Regression</span>
                <span class="pill">Streamlit App</span>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### About")
        st.write("แอปนี้ใช้ Machine Learning ทำนายความนิยมของเพลงจาก metadata ที่สำคัญ")
        st.write(f"**Best Model:** {metrics.get('best_model', '-')}")
        st.markdown('</div>', unsafe_allow_html=True)


def build_quick_inputs(feature_columns):
    st.markdown('<div class="section-title">⚡ Quick Inputs</div>', unsafe_allow_html=True)

    preset = st.selectbox(
        "Preset",
        ["Balanced Pop", "Viral Single", "Indie Release", "Rap Track"],
        index=0
    )

    preset_map = {
        "Balanced Pop": {
            "artist_popularity": 70,
            "artist_followers": 5_000_000,
            "duration_min": 3.4,
            "album_type": "album",
            "primary_genre": "pop",
            "explicit": 0
        },
        "Viral Single": {
            "artist_popularity": 88,
            "artist_followers": 30_000_000,
            "duration_min": 2.9,
            "album_type": "single",
            "primary_genre": "pop",
            "explicit": 0
        },
        "Indie Release": {
            "artist_popularity": 45,
            "artist_followers": 120_000,
            "duration_min": 4.1,
            "album_type": "album",
            "primary_genre": "indie",
            "explicit": 0
        },
        "Rap Track": {
            "artist_popularity": 82,
            "artist_followers": 12_000_000,
            "duration_min": 3.0,
            "album_type": "single",
            "primary_genre": "rap",
            "explicit": 1
        }
    }

    current = preset_map[preset]

    left, right = st.columns(2)

    with left:
        artist_popularity = st.slider(
            "Artist Popularity",
            min_value=0,
            max_value=100,
            value=int(current["artist_popularity"]),
            help="ระดับความนิยมโดยรวมของศิลปิน"
        )

        artist_followers = st.number_input(
            "Artist Followers",
            min_value=0,
            max_value=200_000_000,
            value=int(current["artist_followers"]),
            step=10_000,
            help="จำนวนผู้ติดตามศิลปิน"
        )

        duration_min = st.slider(
            "Song Duration (minutes)",
            min_value=0.5,
            max_value=10.0,
            value=float(current["duration_min"]),
            step=0.1,
            help="ความยาวเพลง"
        )

    with right:
        album_type = st.selectbox(
            "Album Type",
            ["single", "album", "compilation"],
            index=["single", "album", "compilation"].index(current["album_type"])
        )

        primary_genre = st.selectbox(
            "Primary Genre",
            ["pop", "rap", "rock", "indie", "rnb", "edm", "k-pop", "unknown"],
            index=["pop", "rap", "rock", "indie", "rnb", "edm", "k-pop", "unknown"].index(current["primary_genre"])
        )

        explicit = st.toggle(
            "Explicit Lyrics",
            value=bool(current["explicit"]),
            help="เพลงมีคำหยาบหรือเนื้อหาแรงหรือไม่"
        )

    defaults = {
        "track_number": 1 if album_type == "single" else 3,
        "duration_min": duration_min,
        "explicit": 1 if explicit else 0,
        "artist_popularity": artist_popularity,
        "artist_followers": artist_followers,
        "artist_followers_log": float(np.log1p(artist_followers)),
        "album_total_tracks": 1 if album_type == "single" else 12 if album_type == "album" else 18,
        "release_year": 2024,
        "release_month": 6,
        "album_type": album_type,
        "primary_genre": primary_genre
    }

    row = {}
    for col in feature_columns:
        row[col] = defaults.get(col, 0)

    return pd.DataFrame([row]), preset


def render_metrics(metrics, preset):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", metrics.get("best_model", "-"))
    m2.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}")
    m3.metric("Test R²", f"{metrics.get('test_r2', 0):.3f}")
    m4.metric("Preset", preset)


def render_result(pred: float):
    label = popularity_label(pred)
    color = score_color(pred)

    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])

    with left:
        st.metric("Predicted Popularity", f"{pred:.2f}/100")
        st.progress(int(pred))
        st.write(business_comment(pred))

    with right:
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 20px;
                padding: 18px;
                height: 100%;
            ">
                <div style="font-size:0.95rem;color:rgba(255,255,255,0.75);margin-bottom:8px;">
                    Popularity Level
                </div>
                <div style="font-size:2rem;font-weight:800;color:{color};">
                    {label}
                </div>
                <div style="margin-top:12px;" class="tiny-note">
                    โมเดลนี้ประเมินจาก pattern ของข้อมูลใน dataset
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


def show_feature_importance(fi_df: pd.DataFrame):
    if fi_df.empty:
        return

    top_fi = fi_df.head(8).copy().sort_values("importance")
    st.markdown('<div class="section-title">📊 Top Feature Importance</div>', unsafe_allow_html=True)
    st.bar_chart(top_fi.set_index("feature"))


def main():
    inject_css()

    if not os.path.exists(MODEL_PATH):
        st.error("ยังไม่พบโมเดล กรุณารัน `python train.py` ก่อน")
        st.stop()

    model, feature_columns, metrics, fi_df = load_artifacts()

    render_hero(metrics)

    with st.expander("About this app", expanded=False):
        st.write(
            """
            แอปนี้ใช้ Machine Learning ทำนายความนิยมของเพลงจาก metadata สำคัญ
            เช่น ความนิยมของศิลปิน จำนวนผู้ติดตาม ประเภทอัลบั้ม แนวเพลง และความยาวเพลง
            """
        )
        st.info("ผลลัพธ์เป็นค่าคาดการณ์เพื่อการศึกษา ไม่ใช่ค่าจริงจาก Spotify โดยตรง")

    input_df, preset = build_quick_inputs(feature_columns)
    render_metrics(metrics, preset)

    if st.button("🚀 Predict Popularity"):
        try:
            pred = float(model.predict(input_df)[0])
            pred = max(0.0, min(100.0, pred))

            render_result(pred)

            st.markdown('<div class="section-title">🧾 Input Summary</div>', unsafe_allow_html=True)
            st.dataframe(input_df, use_container_width=True)

            show_feature_importance(fi_df)

            with st.expander("Model Details"):
                st.write(f"Selected features: {', '.join(metrics.get('selected_features', feature_columns))}")
                st.write(
                    f"Cross-validation RMSE: {metrics.get('cv_rmse_mean', 0):.2f} ± "
                    f"{metrics.get('cv_rmse_std', 0):.2f}"
                )

                baseline = metrics.get("baseline_models", [])
                if baseline:
                    st.dataframe(pd.DataFrame(baseline), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
