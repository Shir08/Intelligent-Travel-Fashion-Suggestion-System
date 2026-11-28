# Intelligent Travel Fashion Suggestion System

By [Shirley Jane](https://shirley-jane-portfolio.vercel.app) · [LinkedIn](https://www.linkedin.com/in/shirley-jane-sam-manohar-nargunam-99878a27b/)

An outfit assistant that suggests what to wear based on who you are, what you are already wearing and where you are going. It combines a webcam clothing classifier, live colour detection, current weather and Google Gemini, and you can type or speak to it.

## What it does

| Module | What it does | How |
|---|---|---|
| Age and gender suggestion | Suggests a top and bottom for seven age bands (under 3 to 45+) and Male, Female or Unisex | Rule-based lookup, keyboard or microphone input |
| Colour combinations | Given a colour you are wearing, lists colours that go with it | A hand-made colour-pairing dictionary |
| Live colour detection | Point the webcam at a garment and after 15 seconds the colour locks and matching colours appear | OpenCV, HSV hue ranges on the centre of the frame |
| Clothing recognition | Classifies a garment from the webcam into one of 23 categories and suggests what to pair with it | Fine-tuned MobileNetV2 (`fashion_classifier.h5`) |
| Colour and clothing together | Detects both, then asks Gemini for a full outfit | OpenCV, the CNN and `gemini-1.5-pro` |
| Gemini fashion chat | Free-form fashion questions by voice or text | `google-generativeai` |
| Weather-aware travel outfits | Enter or say a destination city and the current weather picks an outfit type (hot and sunny, cold and snowy, rainy, mild, windy) | OpenWeatherMap API |

## The clothing classifier

I trained the image model on a 23-class apparel dataset of about 44,000 images, split 80/20 into 35,284 training and 8,812 validation images.

```
Men:   Denim, Jacket_Vests, Pants, Shorts, Shorts_Polos, Suiting, Sweaters, Sweatshirts, Tees_Tank
Women: Graphic_tees, Blouse_Shirts, Cardigans, Denim, Dresses, Jacket_Coats, Jumpsuits,
       Leggins, Pants, Shorts, Skirts, Sweaters, Sweatshirts, Tops
```

I compared three approaches in the notebook:

| Model | Setup | Best validation accuracy |
|---|---|---|
| MobileNetV2 (transfer learning) | ImageNet weights frozen, then pooling, Dense 512, Dense 256 and softmax, with augmentation, ReduceLROnPlateau and EarlyStopping | about 55.8% after 20 epochs |
| Custom CNN | 3 Conv/MaxPool blocks then Dense 256 | about 50% |
| Deeper CNN with BatchNorm | 4 conv blocks, class-weighted loss | did not converge (about 3%) |

MobileNetV2 is the model saved as `fashion_classifier.h5` and used by the webcam modules. Many classes look alike (Sweaters and Sweatshirts, or men's and women's Denim), which holds top-1 accuracy down. The future work list below covers what I would try next.

## Tech stack

Python, TensorFlow/Keras, TensorFlow Hub, OpenCV, scikit-learn, NumPy, Matplotlib, SpeechRecognition, Google Generative AI (Gemini), OpenWeatherMap API, Flask

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

- The webcam and microphone modules need a local machine. They will not run in Colab or on a headless server.
- To retrain the classifier, change `dataset_path` / `data_dir` to a folder containing one sub-folder per class.
- Training on a CPU is slow, around an hour per epoch, so a GPU is strongly recommended.

## Project structure

```
.
├── Project_II (1).ipynb   # all the modules: rule-based suggestions, computer vision, CNN training, Gemini, weather
└── README.md
```

## Future work

- Fine-tune the upper MobileNetV2 or EfficientNet layers instead of freezing the whole backbone
- Merge near-duplicate classes, or add a head that predicts category first and gender second
- Replace the fixed HSV hue ranges with K-means dominant-colour extraction
- Package it as a Streamlit or Flask web app
- Generate an outfit image from the Gemini suggestion

## License

Released under the [MIT License](LICENSE).
