# ============================================================
# MULTIMODAL EMOTION DETECTION
# Text + Audio Emotion Analysis
# ============================================================

import os
import io
import re
import string
import tempfile

import streamlit as st
import numpy as np
import joblib
import soundfile as sf
import speech_recognition as sr


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emotion Detection System",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Emotion Classification")
st.markdown("**Detect emotions from Text or Audio**")
st.divider()


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse elongated letters: "happpy" -> "happy", "sleepppppppp" -> "sleep".
    # Without this, typos / casual emphasis spelling turn into out-of-vocabulary
    # tokens the model has never seen, which was silently killing confidence
    # on inputs like "i'm happpy today".
    # IMPORTANT: this must match exactly between training (notebook) and
    # inference (this file) — the tokenizer's vocabulary was built from
    # cleaned text, so if you change this you must re-run training so the
    # saved tokenizer/model reflect the new cleaning.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# OGG TO WAV
# ============================================================

def ogg_to_wav(audio_file):
    """Convert an uploaded OGG file (BytesIO/UploadedFile) to a WAV BytesIO buffer."""
    audio_file.seek(0)
    samples, sample_rate = sf.read(
        io.BytesIO(audio_file.getvalue()),
        dtype="int16"
    )

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    return wav_buffer


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

def transcribe_audio(audio_bytes, language):
    """
    Transcribe a WAV audio buffer to text using Google's free speech
    recognition endpoint. `audio_bytes` must be a BytesIO containing a
    valid WAV file (PCM).
    """
    recognizer = sr.Recognizer()

    audio_bytes.seek(0)
    audio_data_bytes = audio_bytes.getvalue()

    if not audio_data_bytes or len(audio_data_bytes) < 44:
        # 44 bytes = minimum size of a valid WAV header. Anything smaller
        # means nothing was actually recorded.
        raise sr.UnknownValueError("No audio data was captured.")

    with sr.AudioFile(io.BytesIO(audio_data_bytes)) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text

    except sr.UnknownValueError:
        raise sr.UnknownValueError("Speech could not be understood.")

    except sr.RequestError as e:
        raise sr.RequestError(f"Google Speech Recognition error: {e}")


# ============================================================
# LOAD NLP MODELS
# ============================================================

@st.cache_resource
def load_nlp_models():
    try:
        if not all([
            os.path.exists(MODEL_PATH),
            os.path.exists(TOKENIZER_PATH),
            os.path.exists(LABEL_ENCODER_PATH)
        ]):
            return None, None, None

        # Try tf_keras first
        try:
            from tf_keras.models import load_model
            model = load_model(MODEL_PATH, compile=False)

        except Exception:
            from tensorflow.keras.layers import InputLayer
            from tensorflow.keras.models import load_model

            class LegacyInputLayer(InputLayer):
                def __init__(self, batch_shape=None, **kwargs):
                    if batch_shape is not None:
                        kwargs["input_shape"] = tuple(batch_shape[1:])
                    super().__init__(**kwargs)

            model = load_model(
                MODEL_PATH,
                custom_objects={"InputLayer": LegacyInputLayer},
                compile=False
            )

        tokenizer = joblib.load(TOKENIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None


# ============================================================
# TEXT EMOTION PREDICTION
# ============================================================

def predict_text_emotion(text, model, tokenizer, label_encoder):
    if not all([model is not None, tokenizer is not None, label_encoder is not None]):
        return None, None

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        cleaned = clean_text(text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")

        probs = model.predict(padded, verbose=0)[0]
        idx = np.argmax(probs)
        emotion = label_encoder.inverse_transform([idx])[0]
        confidence = float(probs[idx])

        return emotion, confidence

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


# ============================================================
# EMOTION EMOJIS
# ============================================================

emotion_emoji = {
    "anger": "🔴",
    "happiness": "💛",
    "love": "❤️",
    "neutral": "⚪",
    "relief": "💚",
    "sadness": "💙",
    "surprise": "😲"
}


# ============================================================
# HELPER: SHOW EMOTION RESULT
# ============================================================

def show_emotion_result(emotion, confidence):
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Detected Emotion", emotion.upper())
    with col_b:
        st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.success(f"**Result:** {emotion_emoji.get(emotion, '😊')} {emotion}")

    # Be honest with the user when the model itself isn't sure — a 7-class
    # softmax has a random baseline of ~14%, so anything well below ~50% is
    # a genuinely weak call, not a confident wrong answer.
    if confidence < 0.5:
        st.warning(
            "⚠️ Low confidence prediction. Try a longer sentence, check for "
            "typos, or treat this result as uncertain."
        )


# ============================================================
# CREATE TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(["📝 Text", "🎤 Audio", "ℹ️ Info"])


# ============================================================
# TAB 1: TEXT EMOTION
# ============================================================

with tab1:
    st.header("Text Emotion Analysis")

    user_text = st.text_area("Enter text to analyze emotion:", height=100, key="text_input")

    if st.button("🔍 Analyze Text", key="text_analyze_btn"):
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Loading model..."):
                model, tokenizer, label_encoder = load_nlp_models()

            if model is None:
                st.error("❌ NLP model files could not be loaded from the 'models/' folder.")
            else:
                emotion, confidence = predict_text_emotion(user_text, model, tokenizer, label_encoder)
                if emotion:
                    show_emotion_result(emotion, confidence)


# ============================================================
# TAB 2: AUDIO EMOTION
# ============================================================

with tab2:
    st.header("Audio Emotion Analysis")

    st.info(
        "🎤 Record your voice or upload an audio file. The system will "
        "convert speech to text and predict emotion."
    )

    # --------------------------------------------------------
    # Make sure this Streamlit version actually supports live
    # microphone recording (st.audio_input needs Streamlit >= 1.35).
    # Older versions silently don't have the attribute, which used
    # to crash this whole tab with an AttributeError.
    # --------------------------------------------------------
    HAS_AUDIO_INPUT = hasattr(st, "audio_input")

    recognition_language = st.selectbox(
        "Speech language:",
        options=["en-US", "en-GB", "hi-IN", "fr-FR", "es-ES"],
        format_func=lambda code: {
            "en-US": "English (US)",
            "en-GB": "English (UK)",
            "hi-IN": "Hindi",
            "fr-FR": "French",
            "es-ES": "Spanish"
        }[code]
    )

    col1, col2 = st.columns(2)

    # ========================================================
    # OPTION 1: UPLOAD AUDIO
    # ========================================================

    with col1:
        st.subheader("📁 Option 1: Upload Audio File")

        audio_file = st.file_uploader(
            "Upload audio file:",
            type=["wav", "ogg", "mp3", "m4a"],
            key="uploaded_audio"
        )

        if audio_file is not None:
            st.success("✅ Audio file uploaded!")
            st.audio(audio_file)

            if st.button("🔍 Analyze Uploaded Audio", key="audio_upload_btn"):
                with st.spinner("Processing audio..."):
                    model, tokenizer, label_encoder = load_nlp_models()

                if model is None:
                    st.error("❌ NLP model files could not be loaded from the 'models/' folder.")
                else:
                    try:
                        if audio_file.name.lower().endswith(".wav"):
                            audio_file.seek(0)
                            wav_audio = io.BytesIO(audio_file.getvalue())

                        elif audio_file.name.lower().endswith(".ogg"):
                            wav_audio = ogg_to_wav(audio_file)

                        else:
                            st.error("For best compatibility, please upload a WAV or OGG file.")
                            st.stop()

                        text = transcribe_audio(wav_audio, recognition_language)

                        st.write("### 📝 Transcribed Text")
                        st.info(text)

                        emotion, confidence = predict_text_emotion(text, model, tokenizer, label_encoder)
                        if emotion:
                            show_emotion_result(emotion, confidence)

                    except sr.UnknownValueError:
                        st.error("❌ Could not understand the speech. Please speak clearly and try again.")

                    except sr.RequestError as e:
                        st.error(f"❌ Speech API error: {e}")

                    except Exception as e:
                        st.error(f"❌ Error processing audio: {e}")

    # ========================================================
    # OPTION 2: LIVE MICROPHONE
    # ========================================================

    with col2:
        st.subheader("🎤 Option 2: Record Live Audio")

        if not HAS_AUDIO_INPUT:
            st.error(
                "❌ Your installed Streamlit version doesn't support live "
                "microphone recording (`st.audio_input`). Please upgrade:\n\n"
                "`pip install -U streamlit` (needs Streamlit 1.35 or newer)."
            )
        else:
            st.write("Click the microphone button, speak clearly, then stop recording.")
            st.caption(
                "⚠️ The browser will ask for microphone permission the first "
                "time — click **Allow**. If you're accessing this app over a "
                "plain `http://` address on a remote server (not `localhost`), "
                "most browsers block microphone access entirely; the app must "
                "be served over **HTTPS** for recording to work."
            )

            # ----------------------------------------------------
            # OFFICIAL STREAMLIT AUDIO INPUT
            # ----------------------------------------------------
            try:
                live_audio = st.audio_input(
                    "🎙️ Record your voice",
                    sample_rate=16000,
                    key="live_microphone"
                )
            except Exception as e:
                live_audio = None
                st.error(f"❌ Could not initialize the microphone widget: {e}")

            # ----------------------------------------------------
            # RECORDING RECEIVED
            # ----------------------------------------------------
            if live_audio is not None:

                audio_value = live_audio.getvalue()

                if not audio_value or len(audio_value) < 44:
                    st.warning(
                        "⚠️ No audio was captured. Make sure your microphone "
                        "is selected/working and try recording again."
                    )
                else:
                    st.success("✅ Recording captured successfully!")
                    st.audio(live_audio)

                    # --------------------------------------------
                    # SAVE RECORDING (optional, best-effort)
                    # --------------------------------------------
                    # NOTE: many deployment environments (Streamlit
                    # Community Cloud, Docker containers, etc.) run with a
                    # read-only filesystem for the app directory. Writing
                    # to PROJECT_DIR there raises a PermissionError, which
                    # used to crash this entire tab right after recording
                    # and looked like "recording isn't allowed". Saving to
                    # disk isn't actually required for transcription (it
                    # already works from the in-memory bytes below), so
                    # this is now optional and wrapped in a try/except.
                    try:
                        recorded_audio_path = os.path.join(
                            tempfile.gettempdir(), "recorded_audio.wav"
                        )
                        with open(recorded_audio_path, "wb") as f:
                            f.write(audio_value)
                    except Exception:
                        pass  # Saving a local copy is a nice-to-have, not required.

                    # --------------------------------------------
                    # ANALYZE RECORDING
                    # --------------------------------------------
                    if st.button("🔍 Analyze Recorded Audio", key="analyze_recorded_audio_btn"):
                        with st.spinner("🎤 Converting speech to text..."):
                            model, tokenizer, label_encoder = load_nlp_models()

                        if model is None:
                            st.error("❌ NLP model files could not be loaded from the 'models/' folder.")
                        else:
                            try:
                                audio_bytes = io.BytesIO(audio_value)
                                text = transcribe_audio(audio_bytes, recognition_language)

                                st.write("### 📝 Transcribed Text")
                                st.info(text)

                                emotion, confidence = predict_text_emotion(
                                    text, model, tokenizer, label_encoder
                                )
                                if emotion:
                                    show_emotion_result(emotion, confidence)

                            except sr.UnknownValueError:
                                st.error("❌ Could not understand your recording.")
                                st.info("💡 Speak clearly for 3–10 seconds and try again.")

                            except sr.RequestError as e:
                                st.error(f"❌ Speech recognition error: {e}")

                            except Exception as e:
                                st.error(f"❌ Error processing recording: {e}")

            else:
                st.info("🎤 Click the microphone button above to record your voice.")


# ============================================================
# TAB 3: INFORMATION
# ============================================================

with tab3:
    st.header("Project Information")

    st.subheader("📊 System Overview")
    st.write(
        """
        This emotion detection system analyzes:

        - **Text:** NLP-based emotion classification using LSTM
        - **Audio:** Speech-to-text + emotion analysis
        """
    )

    st.subheader("🛠️ Technologies")
    st.write(
        """
        - **TensorFlow/Keras:** Deep learning model
        - **Speech Recognition:** Speech-to-text
        - **Streamlit:** Web interface
        - **SoundFile:** Audio file processing
        - **NumPy:** Numerical processing
        - **Joblib:** Model/tokenizer loading
        """
    )

    st.subheader("✅ Features Implemented")
    st.write(
        """
        ✓ Text emotion classifier

        ✓ Speech-to-text transcription

        ✓ Live microphone recording

        ✓ Audio file upload

        ✓ Emotion prediction

        ✓ Confidence score

        ✓ Streamlit web interface
        """
    )

    st.subheader("🚀 Future Improvements")
    st.write(
        """
        - Implement ensemble model combining modalities
        - Add confidence thresholds
        - Add emotion explanation
        - Add real-time emotion visualization
        - Deploy the application
        """
    )

    st.subheader("📁 Project Structure")
    st.code(
        """
Emotion_Classification_NLP/
│
├── app1.py
│
├── models/
│   ├── emotion_model.keras
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
│
├── notebooks/
│   └── Emotion_Classification.ipynb
│
├── dataset/
│   └── Emotion_dataset.csv
│
└── requirements.txt
        """,
        language="text"
    )

    st.subheader("📌 Usage Instructions")
    st.write(
        """
        1. Open the **Audio** tab.
        2. Select your speech language.
        3. Under **Record Live Audio**, click the microphone button.
        4. Allow microphone permission if requested.
        5. Speak clearly for 3–10 seconds.
        6. Stop the recording.
        7. Listen to the recorded audio.
        8. Click **Analyze Recorded Audio**.
        9. The application will convert your speech to text.
        10. The NLP model will predict the emotion.
        11. The application will show the detected emotion and confidence.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("✨ Emotion Classification System | Built with Streamlit & TensorFlow")
