import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

MODEL_PATH = "models/best_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"
METRICS_PATH = "models/metrics.json"
FI_PATH = "outputs/feature_importance.csv"


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
            background: linear-gradient(180deg, #0f0f0f 0%, #121212 100%);
            color: white;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        .hero {
            background: linear-gradient(135deg, #1db954 0%, #121212 100%);
            border-radius: 24px;
            padding: 28px 30px;
            margin-bottom: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }

        .hero h1 {
            color: white;
            margin: 0;
            font-size: 2.2rem;
        }

        .hero p {
            color: #e8f5ec;
            margin-top: 8px;
            font-size: 1rem;
        }

        .mini-card {
            background: #181818;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 14px 16px;
        }

        .result-box {
            background: linear-gradient(135deg, rgba(29,185,84,0.18), rgba(29,185,84,0.05));
            border: 1px solid rgba(29,185,84,0.35);
            border-radius: 22px;
            padding: 22px;
            margin-top: 10px;
        }

        div[data-testid="stMetric"] {
            background: #181818;
            border: 1px solid rgba(255,255,255,0.06);
            padding: 14px;
            border-radius: 18px;
        }

        div.stButton > button {
            background: #1db954;
            color: black;
            border: none;
            font-weight: 700;
            border-radius: 999px;
            height: 3.2rem;
            font-size: 1rem;
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


def short_comment(score: float) -> str:
    if score >= 85:
        return "เพลงนี้มีแนวโน้มสูงมากที่จะไปได้ดีในตลาดกว้าง"
    elif score >= 70:
        return "เพลงนี้มีโอกาสดังได้ดี ถ้ามีการโปรโมตที่เหมาะสม"
    elif score >= 55:
        return "เพลงนี้มีศักยภาพระดับกลางค่อนดี"
    elif score >= 40:
        return "เพลงนี้มีแนวโน้มกลาง ๆ ยังไม่เด่นมาก"
    return "เพลงนี้อาจเหมาะกับกลุ่มเฉพาะมากกว่าตลาดกว้าง"


def build_user_friendly_input(feature_columns):
    st.subheader("🎚️ Quick Inputs")

    c1, c2 = st.columns(2)

    with c1:
        artist_popularity = st.slider(
            "Artist Popularity",
            0, 100, 70,
            help="ระดับความนิยมโดยรวมของศิลปิน"
        )

        artist_followers = st.number_input(
            "Artist Followers",
            min_value=0,
            max_value=200000000,
            value=5000000,
            step=10000,
            help="จำนวนผู้ติดตามศิลปิน"
        )

        duration_min = st.slider(
            "Song Duration (minutes)",
            0.5, 10.0, 3.4, 0.1,
            help="ความยาวเพลง"
        )

    with c2:
        album_type = st.selectbox(
            "Album Type",
            ["single", "album", "compilation"]
        )

        primary_genre = st.selectbox(
            "Primary Genre",
            ["pop", "rap", "rock", "indie", "rnb", "edm", "k-pop", "unknown"]
        )

        explicit = st.toggle(
            "Explicit Lyrics",
            value=False,
            help="เพลงมีคำหยาบหรือเนื้อหาแรงหรือไม่"
        )

    # ค่าที่เหลือให้ระบบเดาให้
    defaults = {
        "track_number": 1,
        "duration_min": duration_min,
        "explicit": 1 if explicit else 0,
        "artist_popularity": artist_popularity,
        "artist_followers": artist_followers,
        "artist_followers_log": float(np.log1p(artist_followers)),
        "album_total_tracks": 12 if album_type == "album" else 1 if album_type == "single" else 18,
        "release_year": 2024,
        "release_month": 6,
        "album_type": album_type,
        "primary_genre": primary_genre
    }

    row = {}
    for col in feature_columns:
        row[col] = defaults.get(col, 0)

    return pd.DataFrame([row])


def show_feature_importance(fi_df: pd.DataFrame):
    if fi_df.empty:
        return

    top_fi = fi_df.head(8).copy().sort_values("importance")
    st.subheader("📊 What affects prediction most")
    st.bar_chart(top_fi.set_index("feature"))


def main():
    inject_css()

    if not os.path.exists(MODEL_PATH):
        st.error("ยังไม่พบโมเดล กรุณารัน python train.py ก่อน")
        st.stop()

    model, feature_columns, metrics, fi_df = load_artifacts()

    st.markdown(
        """
        <div class="hero">
            <h1>🎵 Spotify Popularity Predictor</h1>
            <p>
                Predict track popularity from simple metadata with a clean Spotify-style interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Best Model", metrics.get("best_model", "-"))
    m2.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}")
    m3.metric("Test R²", f"{metrics.get('test_r2', 0):.3f}")

    with st.expander("About this app", expanded=False):
        st.write(
            """
            แอปนี้ใช้ Machine Learning ทำนายความนิยมของเพลงจาก metadata สำคัญ
            เช่น ความดังของศิลปิน จำนวนผู้ติดตาม ประเภทอัลบั้ม แนวเพลง และความยาวเพลง
            """
        )
        st.info("ผลลัพธ์เป็นค่าคาดการณ์เพื่อการศึกษา ไม่ใช่ค่าจริงจาก Spotify โดยตรง")

    input_df = build_user_friendly_input(feature_columns)

    if st.button("🚀 Predict Popularity", use_container_width=True):
        try:
            pred = float(model.predict(input_df)[0])
            pred = max(0.0, min(100.0, pred))
            level = popularity_label(pred)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            r1, r2 = st.columns([1.2, 1])

            with r1:
                st.metric("Predicted Popularity", f"{pred:.2f}/100")
                st.progress(int(pred))
                st.write(short_comment(pred))

            with r2:
                st.metric("Popularity Level", level)
                st.write("โมเดลนี้ประเมินจากรูปแบบข้อมูลใน dataset")

            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("🧾 Input Summary")
            st.dataframe(input_df, use_container_width=True)

            show_feature_importance(fi_df)

        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
