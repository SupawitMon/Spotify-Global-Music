import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Spotify Track Popularity Predictor Pro",
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
        .main {
            background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(29,185,84,0.18), rgba(139,92,246,0.18));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .result-card {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(29,185,84,0.15), rgba(59,130,246,0.12));
            border: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def normalize_for_display(name: str) -> str:
    return name.replace("_", " ").title()


def get_default_input(feature_columns, preset_name="Balanced Release"):
    presets = {
        "Balanced Release": {
            "track_number": 1,
            "duration_min": 3.4,
            "explicit": 0,
            "artist_popularity": 70,
            "artist_followers": 5000000,
            "artist_followers_log": 15.4,
            "album_total_tracks": 12,
            "release_year": 2023,
            "release_month": 6,
            "album_type": "album",
            "primary_genre": "pop"
        },
        "Viral Pop": {
            "track_number": 2,
            "duration_min": 3.0,
            "explicit": 0,
            "artist_popularity": 88,
            "artist_followers": 30000000,
            "artist_followers_log": 17.2,
            "album_total_tracks": 10,
            "release_year": 2024,
            "release_month": 8,
            "album_type": "single",
            "primary_genre": "pop"
        },
        "Indie Niche": {
            "track_number": 5,
            "duration_min": 4.1,
            "explicit": 0,
            "artist_popularity": 45,
            "artist_followers": 120000,
            "artist_followers_log": 11.7,
            "album_total_tracks": 14,
            "release_year": 2022,
            "release_month": 3,
            "album_type": "album",
            "primary_genre": "indie"
        },
        "Rap Release": {
            "track_number": 1,
            "duration_min": 2.8,
            "explicit": 1,
            "artist_popularity": 82,
            "artist_followers": 12000000,
            "artist_followers_log": 16.3,
            "album_total_tracks": 16,
            "release_year": 2024,
            "release_month": 10,
            "album_type": "single",
            "primary_genre": "rap"
        }
    }

    source = presets.get(preset_name, presets["Balanced Release"])
    result = {}

    for col in feature_columns:
        result[col] = source.get(col, 0 if "genre" not in col and "type" not in col else "unknown")

    return result


def get_feature_help():
    return {
        "track_number": "ลำดับเพลงในอัลบั้ม",
        "duration_min": "ความยาวเพลงหน่วยนาที",
        "explicit": "มีคำหยาบหรือเนื้อหาแรงหรือไม่",
        "artist_popularity": "ความนิยมโดยรวมของศิลปิน",
        "artist_followers": "จำนวนผู้ติดตามศิลปิน",
        "artist_followers_log": "ค่าลอการิทึมของผู้ติดตามศิลปิน",
        "album_total_tracks": "จำนวนเพลงทั้งหมดในอัลบั้ม",
        "release_year": "ปีที่ปล่อยเพลง",
        "release_month": "เดือนที่ปล่อยเพลง",
        "album_type": "ประเภทอัลบั้ม เช่น single หรือ album",
        "primary_genre": "แนวเพลงหลักของศิลปิน"
    }


def build_sidebar(feature_columns, metrics):
    st.sidebar.title("🎛️ Control Panel")

    preset = st.sidebar.selectbox(
        "Choose a preset",
        ["Balanced Release", "Viral Pop", "Indie Niche", "Rap Release"]
    )

    default_input = get_default_input(feature_columns, preset)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Snapshot")
    st.sidebar.write(f"**Best Model:** {metrics.get('best_model', '-')}")
    st.sidebar.write(f"**Test RMSE:** {metrics.get('test_rmse', 0):.2f}")
    st.sidebar.write(f"**Test R²:** {metrics.get('test_r2', 0):.3f}")
    st.sidebar.write(f"**CV RMSE:** {metrics.get('cv_rmse_mean', 0):.2f}")

    return preset, default_input


def build_inputs(feature_columns, default_input):
    help_map = get_feature_help()
    user_data = {}

    left, right = st.columns(2)

    range_map = {
        "track_number": (1, 40, 1, 1),
        "duration_min": (0.5, 15.0, 3.5, 0.1),
        "explicit": (0, 1, 0, 1),
        "artist_popularity": (0, 100, 60, 1),
        "artist_followers": (0, 200000000, 1000000, 1000),
        "artist_followers_log": (0.0, 20.0, 13.8, 0.1),
        "album_total_tracks": (1, 40, 10, 1),
        "release_year": (1990, 2030, 2023, 1),
        "release_month": (1, 12, 6, 1)
    }

    for idx, feature in enumerate(feature_columns):
        col = left if idx % 2 == 0 else right
        label = normalize_for_display(feature)
        help_text = help_map.get(feature, "Feature input")

        with col:
            if feature in range_map:
                min_v, max_v, fallback, step_v = range_map[feature]
                current = default_input.get(feature, fallback)

                if isinstance(step_v, int):
                    user_data[feature] = st.number_input(
                        label,
                        min_value=int(min_v),
                        max_value=int(max_v),
                        value=int(current),
                        step=int(step_v),
                        help=help_text
                    )
                else:
                    user_data[feature] = st.slider(
                        label,
                        min_value=float(min_v),
                        max_value=float(max_v),
                        value=float(current),
                        step=float(step_v),
                        help=help_text
                    )
            else:
                user_data[feature] = st.text_input(
                    label,
                    value=str(default_input.get(feature, "")),
                    help=help_text
                )

    return user_data


def popularity_label(score):
    if score >= 85:
        return "🌟 Smash Hit"
    elif score >= 70:
        return "🔥 Strong Potential"
    elif score >= 55:
        return "🎧 Good"
    elif score >= 40:
        return "🙂 Mid"
    else:
        return "🌫️ Low Potential"


def business_interpretation(score):
    if score >= 85:
        return "เพลงนี้มีโอกาสสูงมากที่จะเป็นกระแส เหมาะกับการโปรโมตหนัก"
    elif score >= 70:
        return "เพลงนี้มีแนวโน้มค่อนข้างดี ถ้าดันการตลาดดีมีโอกาสไปไกล"
    elif score >= 55:
        return "เพลงนี้อยู่ระดับกลางค่อนดี ยังต้องอาศัยฐานแฟนและจังหวะปล่อย"
    elif score >= 40:
        return "เพลงนี้มีแนวโน้มกลาง ๆ ยังไม่เด่นมากในเชิง mass market"
    else:
        return "เพลงนี้อาจเหมาะกับกลุ่มเฉพาะมากกว่าตลาดกว้าง"
        

def show_feature_importance(fi_df):
    if fi_df.empty:
        return

    top_fi = fi_df.head(10).copy().sort_values("importance")
    st.subheader("📊 Top Feature Importance")
    st.bar_chart(top_fi.set_index("feature"))


def main():
    inject_css()

    if not os.path.exists(MODEL_PATH):
        st.error("ยังไม่พบโมเดล กรุณารัน `python train.py` ก่อน")
        st.stop()

    model, feature_columns, metrics, fi_df = load_artifacts()
    preset, default_input = build_sidebar(feature_columns, metrics)

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin-bottom:0.2rem;">🎵 Spotify Track Popularity Predictor Pro</h1>
            <p style="margin-top:0.2rem; font-size:1.05rem;">
                Predict track popularity from artist, album, and release metadata with a presentation-ready interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Best Model", metrics.get("best_model", "-"))
    top2.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f}")
    top3.metric("Test R²", f"{metrics.get('test_r2', 0):.3f}")
    top4.metric("Preset", preset)

    with st.expander("📘 About this app", expanded=True):
        st.write(
            """
            แอปนี้ใช้ Machine Learning ทำนายความนิยมของเพลงจาก metadata
            เช่น ความนิยมศิลปิน จำนวนผู้ติดตาม ประเภทอัลบั้ม ช่วงเวลาปล่อย และความยาวเพลง
            """
        )
        st.info(
            "Disclaimer: ผลลัพธ์เป็นค่าคาดการณ์จาก dataset เพื่อการศึกษา ไม่ใช่ค่าจริงจาก Spotify โดยตรง"
        )

    st.subheader("🎚️ Enter Track Metadata")
    user_data = build_inputs(feature_columns, default_input)

    if st.button("🚀 Predict Popularity", use_container_width=True):
        input_df = pd.DataFrame([user_data])

        try:
            if "artist_followers" in input_df.columns and "artist_followers_log" in feature_columns and "artist_followers_log" not in input_df.columns:
                input_df["artist_followers_log"] = np.log1p(input_df["artist_followers"])

            pred = float(model.predict(input_df)[0])
            pred = max(0.0, min(100.0, pred))
            label = popularity_label(pred)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.metric("Predicted Popularity", f"{pred:.2f}/100")
                st.progress(int(pred))
                st.write(business_interpretation(pred))
            with c2:
                st.metric("Popularity Level", label)
                st.write("โมเดลนี้ประเมินจาก pattern ของเพลงใน dataset")
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("🧾 Input Summary")
            st.dataframe(input_df, use_container_width=True)

            show_feature_importance(fi_df)

            with st.expander("🔍 Model Insights"):
                st.write(f"Selected features: {', '.join(metrics.get('selected_features', feature_columns))}")
                st.write(f"Cross-validation RMSE: {metrics.get('cv_rmse_mean', 0):.2f} ± {metrics.get('cv_rmse_std', 0):.2f}")

                baseline = metrics.get("baseline_models", [])
                if baseline:
                    baseline_df = pd.DataFrame(baseline)
                    st.dataframe(baseline_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    import numpy as np
    main()