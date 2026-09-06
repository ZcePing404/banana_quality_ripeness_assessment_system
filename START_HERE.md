# Updated Banana Quality Assessment

Use Python 3.11 or 3.12 in a virtual environment.

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Hard-refresh the browser after replacing an older project.

## Features
- Single image, multiple image and folder upload; each batch file is analysed independently.
- Restored original two-point reference line: enable calibration, enter its known length in cm, then click its two endpoints. The measure icon displays estimated bounding-box length and width. This is spatial scaling, not perspective rectification or curved banana length. Recalibrate after changing image/crop. Batch/video never reuse another image's reference.
- Preprocessing is the first Processing Details row: aspect-preserving resize, CLAHE and Gaussian filtering. Images are not stretched or cropped by the display and have white surrounding space. Black pixels inside binary masks are actual mask data.
- Final Assessment appears at the top of the right column beside Image Input; desktop columns stretch to the same overall height. Narrow screens stack vertically.
- Download PDF report exports the current image, batch findings (including failures), or the latest 200 analysed video frames. Batch reports include metrics; a single-image report also includes the final image. Export is user initiated, with report content generated automatically.
- Video file input and live webcam analysis show frame results and measured analysis FPS. Browser-supported video formats are accepted. Sequential sampled frames prevent an inference backlog; this does not promise every source frame or original playback FPS. Uploaded playback pauses during inference. Camera permission requires localhost or HTTPS. Stop ends the loop and releases the webcam.

## Validation and limitations
Five bundled images passed the analysis endpoint sequentially. Calibration scale, preprocessing output, invalid image errors and PDF generation were checked. The provided trained models are retained; TensorFlow inference requires the dependencies above. This verification environment did not have TensorFlow installed, so model predictions were reported as unavailable rather than fabricated. Image decoding/segmentation and PDF export still function if model loading fails.

The original merge removed calibration HTML and backend handling while leaving JavaScript references. Missing canvas/size elements interrupted the script and could make every batch item fail before a request. Restored elements and backend handling resolve that path. HTTP and individual file errors are now clearer. Images above 16 MB are rejected individually.

Four automated regression tests passed. JavaScript syntax and referenced HTML IDs were checked. Browser visual and live-webcam verification could not run because the browser download timed out; test those locally before demonstration.
