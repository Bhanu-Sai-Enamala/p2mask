# Local Face PII Redaction Tool

This Streamlit app redacts personally identifiable facial regions in uploaded images by covering the eyes and mouth with semi-transparent black rectangles.

The app runs entirely locally. It uses MediaPipe Face Mesh for offline face landmark detection, OpenCV and NumPy for image preparation, and Pillow for in-memory image redaction and export.

## Installation

Use Python 3.8 or newer, then install the dependencies:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended so the pinned local computer-vision dependencies do not affect other Python projects.

## Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL in your browser, upload a JPG, JPEG, or PNG image, adjust the opacity and padding sliders, and click **Process Image**. The redacted image can be downloaded as a PNG.

## How It Works

1. The uploaded image is read in memory with Pillow.
2. MediaPipe Face Mesh detects up to 10 faces and returns 468 facial landmarks per face.
3. The app extracts landmark groups around the left eye, right eye, and mouth.
4. Each landmark group is converted into a padded bounding box.
5. Pillow draws semi-transparent black rectangles over those boxes.
6. The final image is serialized directly to PNG bytes for download.

## Privacy Note

All processing happens on your machine. The app does not call external APIs, does not upload images to third-party services, and does not save uploaded or redacted images to disk. Streamlit usage telemetry is disabled in `.streamlit/config.toml`.

## Testing Checklist

- Single face detection works
- Multiple face detection works
- Eyes are properly covered
- Mouth is properly covered
- Opacity slider works
- Padding slider works
- Download produces a valid image file
- No external API calls are made
- Different image sizes are handled
- Images with no faces are handled gracefully
