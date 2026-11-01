# Business Card Image Processing

This project was developed using Python and OpenCV as part of my summer internship work. It processes images of business cards by checking their quality, detecting card boundaries, correcting perspective, and enhancing image contrast.

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Pytest

## Features

- Image quality validation (blur, brightness, glare and resolution)
- Image resizing and grayscale conversion
- Gaussian blur and Canny edge detection
- Contour detection and area filtering
- Quadrilateral detection and corner ordering
- Perspective transformation to flatten business cards
- CLAHE-based contrast enhancement

## Project Structure

`main.py` - Main program

`quality_check.py` - Image quality validation

`card_detection.py` - Card detection and image processing

`preprocess.py` - Reserved for preprocessing

`utils.py` - Reserved for utility functions

`tests/` - Automated tests

`images/` - Input and output image directories

## How to Run

**1. Clone the repository and open the project directory.**

**2. Create and activate a virtual environment:**

`python -m venv .venv`

`.\.venv\Scripts\Activate.ps1`

**3. Install dependencies:**

`python -m pip install -r requirements.txt`

**4. Place a business card image in `images/input/` named `card1.jpeg`.**

**5. Run the application:**

`python main.py`

## Testing

Automated tests are written using Pytest.

Run the tests using:

`python -m pytest -v`

## Current Limitations

- Perspective transformation uses fixed output dimensions.
- OCR and automatic text extraction are not implemented.

## Future Improvements

- Improve glare detection accuracy.
- Add OCR to extract information from business cards.
- Test the pipeline on a larger variety of card images.
