# Intelligent Travel Fashion Suggestion System — Personalized Outfit Planning

A deep-learning and computer-vision assistant that suggests what to wear based on **who you are**, **what you're already wearing**, and **where you're going**. It combines a webcam clothing classifier, real-time colour detection, live weather data and Google Gemini to produce personalised outfit recommendations, with typed or voice input throughout.

---

## Features

| Module | What it does | How |
|---|---|---|
| **Age & gender outfit suggestion** | Recommends a top/bottom pairing for seven age bands (under 3 → 45+) and Male / Female / Unisex | Rule-based lookup; input by keyboard or microphone |
| **Colour-combination suggestion** | Given a colour you're wearing, lists colours that go with it | Curated colour-pairing dictionary; voice or text input |
| **Live colour detection** | Point the webcam at a garment; after 15 s the colour locks and complementary colours appear on screen | OpenCV, HSV hue ranges on the centre of the frame |
| **Clothing-type recognition** | Classifies a garment from the webcam into one of 23 categories and suggests items to pair with it | Fine-tuned MobileNetV2 (`fashion_classifier.h5`) |
| **Colour + clothing combined** | Detects colour and garment type together, then asks Gemini for a full outfit | OpenCV + CNN + `gemini-1.5-pro` |
| **Gemini fashion chat** | Free-form fashion questions by voice or text | `google-generativeai` |
| **Weather-aware travel outfits** | Enter (or say) a destination city; current weather picks an outfit category (Hot & Sunny, Cold & Snowy, Rainy, Mild, Windy) | OpenWeatherMap API |

## Clothing classifier

The image model is trained on a 23-class apparel dataset (≈44k images, split 80/20 into **35,284 training** and **8,812 validation** images):

```
Men:   Denim, Jacket_Vests, Pants, Shorts, Shorts_Polos, Suiting, Sweaters, Sweatshirts, Tees_Tank
Women: Graphic_tees, Blouse_Shirts, Cardigans, Denim, Dresses, Jacket_Coats, Jumpsuits,
       Leggins, Pants, Shorts, Skirts, Sweaters, Sweatshirts, Tops
```

Three approaches were compared in the notebook:

| Model | Setup | Best validation accuracy |
|---|---|---|
| **MobileNetV2 (transfer learning)** | ImageNet weights frozen; GAP → Dense 512 → Dense 256 → softmax; augmentation, ReduceLROnPlateau, EarlyStopping | **≈ 55.8 %** (20 epochs) |
| Custom CNN | 3 × Conv/MaxPool → Dense 256 | ≈ 50 % |
| Deeper CNN + BatchNorm | 4 conv blocks, class-weighted loss | did not converge (≈ 3 %) |

MobileNetV2 is the model saved as `fashion_classifier.h5` and used by the webcam modules. Many classes are visually close (e.g. *Sweaters* vs *Sweatshirts*, men's vs women's *Denim*), which limits top-1 accuracy; see **Future work**.

## Tech stack

Python · TensorFlow / Keras · TensorFlow Hub · OpenCV · scikit-learn · NumPy · Matplotlib · SpeechRecognition · Google Generative AI (Gemini) · OpenWeatherMap API · Flask

## Getting started

```bash
git clone <this-repo-url>
cd Intelligent-Travel-Fashion-Suggestion-System-Personalized-Outfit-Planning-

pip install tensorflow tensorflow_hub opencv-python numpy matplotlib scikit-learn \
            SpeechRecognition pyaudio google-generativeai requests flask
```

Set your API keys as environment variables (the notebook reads them with `os.environ`):

```bash
export GEMINI_API_KEY="your-gemini-key"
export OPENWEATHER_API_KEY="your-openweathermap-key"
```

Then open the notebook:

```bash
jupyter notebook "Project_II (1).ipynb"
```

**Notes**

- The webcam and microphone modules need a local machine (they won't run in Colab or a headless server).
- To retrain the classifier, change `dataset_path` / `data_dir` to a folder containing one sub-folder per class.
- Training on CPU is slow (≈ 1 hour per epoch); a GPU is strongly recommended.

## Project structure

```
.
├── Project_II (1).ipynb   # All modules: rule-based suggestions, CV, CNN training, Gemini, weather
└── README.md
```

## Future work

- Fine-tune the upper MobileNetV2 / EfficientNet layers instead of freezing the whole backbone
- Merge near-duplicate classes or add a hierarchical (category → gender) head
- Replace fixed HSV hue ranges with K-means dominant-colour extraction for more robust colour naming
- Package the pipeline as a Streamlit or Flask web app
- Add outfit image generation from the Gemini suggestion
