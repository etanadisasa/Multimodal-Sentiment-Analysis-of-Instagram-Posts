# Multimodal Sentiment Analysis of Instagram Posts

## Author

**Etana Disasa**
Master of Science in Data Science
[edisasa@regis.edu](mailto:edisasa@regis.edu)

## Project Overview

This practicum project develops an interactive tool for **multimodal sentiment analysis** of Instagram posts. It combines **natural language processing (NLP)** for captions and comments with **computer vision** for analyzing images to extract sentiment and emotions. The tool integrates these modalities to provide a comprehensive sentiment evaluation of Instagram content.

### Features

* Extract captions, comments, and images from Instagram posts
* Perform sentiment analysis on captions and comments using **BERT**
* Detect emotions and facial expressions in images using **ResNet50** and **MobileNetV2**
* Aggregate and display weighted sentiment and emotion scores
* Interactive interface for visualization and exploration of results

## Requirements

* Python 3.10+
* Libraries (install via `pip`):

```
pip install -r requirements.txt
```

## Usage Instructions

1. **Login to Instagram** (required for private accounts):

```
instaloader --login YOUR_USERNAME
```

* Enter your password and 2FA code if prompted.
* This creates a session file that the project scripts can reuse.

2. **Run the main script**:

```
python Code/sentiment-processor.py
```

* Follow the prompts to input Instagram post URLs or usernames.

3. **View Results**:

* Sentiment scores and emotion visualizations will be displayed in the console or GUI depending on the interface used.

## Project Structure
The app and utility script, and all other necessary files were in the same folder when code developed. 
Please place all codes and necessary files/folders in the same project folder.
```
project-folder/
│
├── Code/                    # Python scripts and utilities
│   ├── sentiment-processor.py
│   └── socialmedia-utils.py
│
├── Report/                  # Practicum report files and license
│   └── LICENSE.txt
│
├── Presentation/            # Slides or presentation files
│
├── NSFW/                    # NSFW model data (make sure you download your version here
├── requirements.txt         # List of Python dependencies
├── yolov8n.pt               # YOLO model file can be downloaded and be stored here
└── README.md                # Project overview and instructions
```

## Licenses and Acknowledgments

All libraries and pretrained models used in this project are **open-source** and used under their respective licenses. No models were modified or redistributed outside the scope of this academic project. See `Report/LICENSE.txt` for full details.

## Contact

For questions or inquiries, please contact **Etana Disasa** at [edisasa@regis.edu](mailto:edisasa@regis.edu).
