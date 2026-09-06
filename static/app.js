let cameraStream = null;
let capturedBlob = null;
let currentFileName = "camera_capture.jpg";

// Current image/crop state.
let originalImageBlob = null;
let activeImageBlob = null;
let cropSourceImage = null;
let cropSelection = null;
let cropDragging = false;
let cropStartPoint = null;

// Live-analysis queue. Only one request is allowed at a time; repeated slider
// movement is coalesced into the newest pending settings instead of creating a
// backlog of expensive Flask requests.
let analysisInFlight = false;
let pendingAnalysis = false;
let pendingIncludeRipeness = false;
let analysisTimer = null;
let analysisEpoch = 0;

const settingIds = [
    "kmeans_k",
    "banana_score_threshold",
    "banana_opening",
    "banana_closing",
    "blemish_k",
    "blemish_chroma_threshold",
    "blemish_opening",
    "blemish_closing",
];

const bananaSettingIds = new Set([
    "kmeans_k",
    "banana_score_threshold",
    "banana_opening",
    "banana_closing",
]);

// ---------------------------------------------------------------------
// Settings: range + numeric input synchronisation
// ---------------------------------------------------------------------

function decimalsForStep(step) {
    const text = String(step);
    return text.includes(".") ? text.split(".")[1].length : 0;
}

function clampToRange(range, value) {
    const min = Number(range.min);
    const max = Number(range.max);
    const step = Number(range.step) || 1;
    let numeric = Math.max(min, Math.min(max, Number(value)));

    // Snap to the same legal values as the slider.
    numeric = min + Math.round((numeric - min) / step) * step;
    const decimals = decimalsForStep(step);
    return Number(numeric.toFixed(decimals));
}

function syncNumberFromRange(id) {
    const range = document.getElementById(id);
    const number = document.getElementById(`${id}_number`);
    if (range && number) number.value = range.value;
}

function normaliseNumberInput(id, { analyse = true } = {}) {
    const range = document.getElementById(id);
    const number = document.getElementById(`${id}_number`);
    if (!range || !number) return false;

    const raw = String(number.value).trim();
    if (raw === "" || !Number.isFinite(Number(raw))) return false;

    const value = clampToRange(range, Number(raw));
    range.value = String(value);
    number.value = range.value;

    if (analyse) queueAnalysis(bananaSettingIds.has(id));
    return true;
}

function bindSetting(id) {
    const range = document.getElementById(id);
    const number = document.getElementById(`${id}_number`);
    if (!range || !number) return;

    syncNumberFromRange(id);

    range.addEventListener("input", () => {
        syncNumberFromRange(id);
        queueAnalysis(bananaSettingIds.has(id));
    });

    // Block scientific notation, signs and other characters that a native
    // number field may otherwise accept. Decimal point is allowed only for
    // decimal controls.
    number.addEventListener("keydown", (event) => {
        const allowedNavigation = new Set([
            "Backspace", "Delete", "ArrowLeft", "ArrowRight", "ArrowUp",
            "ArrowDown", "Home", "End", "Tab", "Enter",
        ]);
        if (allowedNavigation.has(event.key) || event.ctrlKey || event.metaKey) return;

        const isDigit = /^\d$/.test(event.key);
        const decimalAllowed = range.dataset.kind === "decimal";
        const isDecimalPoint = event.key === "." && decimalAllowed && !number.value.includes(".");

        if (!isDigit && !isDecimalPoint) event.preventDefault();
    });

    number.addEventListener("input", () => {
        const decimalAllowed = range.dataset.kind === "decimal";
        let cleaned = number.value.replace(/[^0-9.]/g, "");

        if (!decimalAllowed) {
            cleaned = cleaned.replace(/\./g, "");
        } else {
            const firstDot = cleaned.indexOf(".");
            if (firstDot !== -1) {
                cleaned = cleaned.slice(0, firstDot + 1) + cleaned.slice(firstDot + 1).replace(/\./g, "");
            }
        }

        if (number.value !== cleaned) number.value = cleaned;
        if (cleaned === "" || cleaned === ".") return;

        const numeric = Number(cleaned);
        if (!Number.isFinite(numeric)) return;
        if (numeric < Number(range.min) || numeric > Number(range.max)) return;

        range.value = String(numeric);
        number.value = range.value;
        queueAnalysis(bananaSettingIds.has(id));
    });

    number.addEventListener("blur", () => normaliseNumberInput(id));
    number.addEventListener("change", () => normaliseNumberInput(id));
}

for (const id of settingIds) bindSetting(id);

document.getElementById("reset-settings").addEventListener("click", () => {
    for (const id of settingIds) {
        const range = document.getElementById(id);
        if (!range) continue;
        range.value = range.dataset.default;
        syncNumberFromRange(id);
    }
    queueAnalysis(true, true);
});

// ---------------------------------------------------------------------
// Source selection and upload
// ---------------------------------------------------------------------

const uploadPanel = document.getElementById("upload-panel");
const cameraPanel = document.getElementById("camera-panel");
const imageFile = document.getElementById("image-file");
const uploadMeta = document.getElementById("upload-meta");
const uploadName = document.getElementById("upload-name");
const uploadSize = document.getElementById("upload-size");

for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener("change", () => {
        const useCamera = radio.value === "camera" && radio.checked;
        uploadPanel.classList.toggle("hidden", useCamera);
        cameraPanel.classList.toggle("hidden", !useCamera);
    });
}

function prettyBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

imageFile.addEventListener("change", async () => {
    const file = imageFile.files[0];
    if (!file) {
        clearUpload();
        return;
    }

    analysisEpoch += 1;
    currentFileName = file.name;
    originalImageBlob = file;
    activeImageBlob = file;
    capturedBlob = null;

    uploadMeta.classList.remove("hidden");
    uploadName.textContent = file.name;
    uploadSize.textContent = prettyBytes(file.size);

    await initialiseCropEditor(file);
    queueAnalysis(true, true);
});

function clearUpload() {
    analysisEpoch += 1;
    imageFile.value = "";
    uploadMeta.classList.add("hidden");
    currentFileName = "banana.jpg";
    originalImageBlob = null;
    activeImageBlob = null;
    capturedBlob = null;
    pendingAnalysis = false;
    pendingIncludeRipeness = false;
    clearTimeout(analysisTimer);
    resetCropEditorState();
    resetAnalysisUI();
    setStatus("", "info");
}

document.getElementById("clear-upload").addEventListener("click", clearUpload);

// ---------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------

const video = document.getElementById("camera-video");
const canvas = document.getElementById("camera-canvas");
const startCamera = document.getElementById("start-camera");
const captureCamera = document.getElementById("capture-camera");
const stopCamera = document.getElementById("stop-camera");

startCamera.addEventListener("click", async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        video.srcObject = cameraStream;
        captureCamera.disabled = false;
        stopCamera.disabled = false;
        setStatus("Camera ready.", "info");
    } catch (error) {
        setStatus(`Could not access camera: ${error.message}`, "error");
    }
});

captureCamera.addEventListener("click", () => {
    if (!cameraStream || video.videoWidth === 0) {
        setStatus("Camera frame is not ready yet.", "error");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        if (!blob) {
            setStatus("Could not capture the camera frame.", "error");
            return;
        }

        analysisEpoch += 1;
        capturedBlob = blob;
        originalImageBlob = blob;
        activeImageBlob = blob;
        currentFileName = "camera_capture.jpg";
        await initialiseCropEditor(blob);
        queueAnalysis(true, true);
    }, "image/jpeg", 0.95);
});

stopCamera.addEventListener("click", stopCameraStream);

function stopCameraStream() {
    if (cameraStream) {
        for (const track of cameraStream.getTracks()) track.stop();
        cameraStream = null;
    }
    video.srcObject = null;
    captureCamera.disabled = true;
    stopCamera.disabled = true;
}

window.addEventListener("beforeunload", stopCameraStream);

// ---------------------------------------------------------------------
// Optional crop editor (the only post-selection image preview in Input)
// ---------------------------------------------------------------------

const cropEditor = document.getElementById("crop-editor");
const cropCanvas = document.getElementById("crop-canvas");
const cropContext = cropCanvas.getContext("2d");
const cropStatus = document.getElementById("crop-status");
const applyCropButton = document.getElementById("apply-crop");
const resetCropButton = document.getElementById("reset-crop");

function resetCropEditorState() {
    cropSourceImage = null;
    cropSelection = null;
    cropDragging = false;
    cropStartPoint = null;
    cropCanvas.width = 1;
    cropCanvas.height = 1;
    cropEditor.classList.add("hidden");
    cropStatus.textContent = "Not cropped";
    cropStatus.classList.remove("applied");
}

function blobToImage(blob) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        const url = URL.createObjectURL(blob);
        image.onload = () => {
            URL.revokeObjectURL(url);
            resolve(image);
        };
        image.onerror = () => {
            URL.revokeObjectURL(url);
            reject(new Error("Could not open the image for cropping."));
        };
        image.src = url;
    });
}

async function initialiseCropEditor(blob) {
    try {
        cropSourceImage = await blobToImage(blob);
        const maxWidth = 760;
        const maxHeight = 520;
        const scale = Math.min(
            1,
            maxWidth / cropSourceImage.naturalWidth,
            maxHeight / cropSourceImage.naturalHeight
        );

        cropCanvas.width = Math.max(1, Math.round(cropSourceImage.naturalWidth * scale));
        cropCanvas.height = Math.max(1, Math.round(cropSourceImage.naturalHeight * scale));
        cropSelection = { x: 0, y: 0, width: cropCanvas.width, height: cropCanvas.height };
        cropEditor.classList.remove("hidden");
        cropStatus.textContent = "Not cropped";
        cropStatus.classList.remove("applied");
        drawCropEditor();
    } catch (error) {
        setStatus(error.message, "error");
    }
}

function drawCropEditor() {
    if (!cropSourceImage) return;

    cropContext.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
    cropContext.drawImage(cropSourceImage, 0, 0, cropCanvas.width, cropCanvas.height);
    if (!cropSelection) return;

    cropContext.save();
    cropContext.fillStyle = "rgba(10, 24, 18, 0.40)";
    cropContext.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
    cropContext.clearRect(cropSelection.x, cropSelection.y, cropSelection.width, cropSelection.height);

    const sx = cropSelection.x / cropCanvas.width * cropSourceImage.naturalWidth;
    const sy = cropSelection.y / cropCanvas.height * cropSourceImage.naturalHeight;
    const sw = cropSelection.width / cropCanvas.width * cropSourceImage.naturalWidth;
    const sh = cropSelection.height / cropCanvas.height * cropSourceImage.naturalHeight;

    cropContext.drawImage(
        cropSourceImage,
        sx, sy, sw, sh,
        cropSelection.x, cropSelection.y, cropSelection.width, cropSelection.height
    );

    cropContext.strokeStyle = "#2f7d59";
    cropContext.lineWidth = 3;
    cropContext.setLineDash([8, 6]);
    cropContext.strokeRect(
        cropSelection.x + 1.5,
        cropSelection.y + 1.5,
        Math.max(0, cropSelection.width - 3),
        Math.max(0, cropSelection.height - 3)
    );
    cropContext.restore();
}

function canvasPointFromEvent(event) {
    const rect = cropCanvas.getBoundingClientRect();
    return {
        x: Math.max(0, Math.min(cropCanvas.width, (event.clientX - rect.left) * cropCanvas.width / rect.width)),
        y: Math.max(0, Math.min(cropCanvas.height, (event.clientY - rect.top) * cropCanvas.height / rect.height)),
    };
}

cropCanvas.addEventListener("pointerdown", (event) => {
    if (!cropSourceImage) return;
    cropDragging = true;
    cropStartPoint = canvasPointFromEvent(event);
    cropSelection = { x: cropStartPoint.x, y: cropStartPoint.y, width: 0, height: 0 };
    cropCanvas.setPointerCapture(event.pointerId);
    drawCropEditor();
});

cropCanvas.addEventListener("pointermove", (event) => {
    if (!cropDragging || !cropStartPoint) return;
    const point = canvasPointFromEvent(event);
    cropSelection = {
        x: Math.min(cropStartPoint.x, point.x),
        y: Math.min(cropStartPoint.y, point.y),
        width: Math.abs(point.x - cropStartPoint.x),
        height: Math.abs(point.y - cropStartPoint.y),
    };
    drawCropEditor();
});

function finishCropDrag(event) {
    if (!cropDragging) return;
    cropDragging = false;

    if (!cropSelection || cropSelection.width < 12 || cropSelection.height < 12) {
        cropSelection = { x: 0, y: 0, width: cropCanvas.width, height: cropCanvas.height };
    }

    if (event?.pointerId != null) {
        try { cropCanvas.releasePointerCapture(event.pointerId); } catch (_) {}
    }
    drawCropEditor();
}

cropCanvas.addEventListener("pointerup", finishCropDrag);
cropCanvas.addEventListener("pointercancel", finishCropDrag);

applyCropButton.addEventListener("click", async () => {
    if (!cropSourceImage || !cropSelection || !originalImageBlob) {
        setStatus("Choose or capture an image before cropping.", "error");
        return;
    }

    const scaleX = cropSourceImage.naturalWidth / cropCanvas.width;
    const scaleY = cropSourceImage.naturalHeight / cropCanvas.height;
    const sx = Math.max(0, Math.round(cropSelection.x * scaleX));
    const sy = Math.max(0, Math.round(cropSelection.y * scaleY));
    const sw = Math.min(cropSourceImage.naturalWidth - sx, Math.max(1, Math.round(cropSelection.width * scaleX)));
    const sh = Math.min(cropSourceImage.naturalHeight - sy, Math.max(1, Math.round(cropSelection.height * scaleY)));

    const fullArea = cropSourceImage.naturalWidth * cropSourceImage.naturalHeight;
    const cropArea = sw * sh;
    if (cropArea / fullArea > 0.995) {
        activeImageBlob = originalImageBlob;
        cropStatus.textContent = "Full image";
        cropStatus.classList.remove("applied");
        queueAnalysis(true, true);
        return;
    }

    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = sw;
    outputCanvas.height = sh;
    outputCanvas.getContext("2d").drawImage(cropSourceImage, sx, sy, sw, sh, 0, 0, sw, sh);

    const croppedBlob = await new Promise((resolve) => outputCanvas.toBlob(resolve, "image/jpeg", 0.96));
    if (!croppedBlob) {
        setStatus("Could not create the cropped image.", "error");
        return;
    }

    analysisEpoch += 1;
    activeImageBlob = croppedBlob;
    cropStatus.textContent = "Crop applied";
    cropStatus.classList.add("applied");
    queueAnalysis(true, true);
});

resetCropButton.addEventListener("click", async () => {
    if (!originalImageBlob) return;
    analysisEpoch += 1;
    activeImageBlob = originalImageBlob;
    await initialiseCropEditor(originalImageBlob);
    cropStatus.textContent = "Not cropped";
    cropStatus.classList.remove("applied");
    queueAnalysis(true, true);
});

function selectedSource() {
    return document.querySelector('input[name="source"]:checked').value;
}

// ---------------------------------------------------------------------
// Status + modal helpers
// ---------------------------------------------------------------------

function setStatus(message, type = "info") {
    const status = document.getElementById("status");
    status.textContent = message;
    status.className = `status ${type}${message ? "" : " empty"}`;
}

const imageModal = document.getElementById("image-modal");
const imageModalImg = document.getElementById("image-modal-img");
const imageModalCaption = document.getElementById("image-modal-caption");
const imageModalClose = document.getElementById("image-modal-close");

function openImageModal(src, caption) {
    imageModalImg.src = src;
    imageModalImg.alt = caption;
    imageModalCaption.textContent = caption;
    imageModal.classList.remove("hidden");
    imageModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
}

function closeImageModal() {
    imageModal.classList.add("hidden");
    imageModal.setAttribute("aria-hidden", "true");
    imageModalImg.src = "";
    document.body.classList.remove("modal-open");
}

imageModalClose.addEventListener("click", closeImageModal);
imageModal.addEventListener("click", (event) => {
    if (event.target === imageModal) closeImageModal();
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !imageModal.classList.contains("hidden")) closeImageModal();
});

function makeImageCard(src, caption) {
    if (!src) return null;
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    const cap = document.createElement("figcaption");

    img.src = src;
    img.alt = caption;
    img.loading = "eager";
    img.title = "Click to enlarge";
    img.classList.add("expandable-image");
    img.addEventListener("click", () => openImageModal(src, caption));
    cap.textContent = caption;
    figure.append(img, cap);
    return figure;
}

function arrowIcon() {
    const arrow = document.createElement("div");
    arrow.className = "process-arrow";
    arrow.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M5 12h13M14 8l4 4-4 4"/>
        </svg>`;
    return arrow;
}

function renderImageList(containerId, items) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";

    for (const item of items || []) {
        if (!item?.src) continue;
        const card = makeImageCard(item.src, item.caption);
        if (card) container.appendChild(card);
    }
}

function renderMiniPipeline(containerId, items) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";

    (items || []).forEach((item, index) => {
        const card = makeImageCard(item.src, item.caption);
        if (card) {
            card.classList.add("mini-card");
            container.appendChild(card);
        }
        if (index < items.length - 1) container.appendChild(arrowIcon());
    });
}

function renderProbabilityRows(probabilities, compact = false) {
    const wrapper = document.createElement("div");
    wrapper.className = compact ? "probability-list compact" : "probability-list";

    const entries = Object.entries(probabilities || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
    for (const [name, raw] of entries) {
        const probability = Math.max(0, Math.min(1, Number(raw)));
        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
            <span>${name}</span>
            <div class="progress"><div class="progress-fill" style="width:${probability * 100}%"></div></div>
            <strong>${(probability * 100).toFixed(1)}%</strong>`;
        wrapper.appendChild(row);
    }
    return wrapper;
}

function renderRipenessProcessing(data) {
    if (data.ripeness?.skipped) return;

    const container = document.getElementById("top-ripeness-processing");
    const warning = document.getElementById("ripeness-warning");
    container.innerHTML = "";

    if (!data.ripeness?.available) {
        const unavailable = document.createElement("div");
        unavailable.className = "ripeness-unavailable";
        unavailable.textContent = "Ripeness prediction unavailable";
        container.appendChild(unavailable);
        warning.textContent = data.ripeness?.error || "Ripeness classification is unavailable.";
        warning.classList.remove("hidden");
        return;
    }

    warning.classList.add("hidden");

    const inputCard = makeImageCard(data.images.ripeness_segmented, "1. Classifier input");
    if (inputCard) {
        inputCard.classList.add("mini-card", "ripeness-input-card");
        container.appendChild(inputCard);
    }

    const infoStack = document.createElement("div");
    infoStack.className = "ripeness-info-stack";

    const predictionCard = document.createElement("div");
    predictionCard.className = "ripeness-result-card";
    const confidence = data.metrics.ripeness_confidence == null
        ? ""
        : `${(data.metrics.ripeness_confidence * 100).toFixed(1)}% confidence`;
    predictionCard.innerHTML = `
        <span>3. Prediction</span>
        <strong>${data.metrics.ripeness || "—"}</strong>
        <small>${confidence}</small>`;

    const probabilityCard = document.createElement("div");
    probabilityCard.className = "ripeness-probability-card";
    const probabilityTitle = document.createElement("span");
    probabilityTitle.textContent = "2. Class probabilities";
    probabilityCard.append(probabilityTitle, renderProbabilityRows(data.ripeness.probabilities, true));

    infoStack.append(predictionCard, probabilityCard);
    container.appendChild(infoStack);
}

document.getElementById("show-details-toggle").addEventListener("change", (event) => {
    document.getElementById("top-processing-details").classList.toggle("hidden", !event.target.checked);
});

// ---------------------------------------------------------------------
// Live analysis
// ---------------------------------------------------------------------

function hasActiveImage() {
    if (selectedSource() === "upload") return Boolean(imageFile.files[0] && activeImageBlob);
    return Boolean(capturedBlob && activeImageBlob);
}

function queueAnalysis(includeRipeness = true, immediate = false) {
    if (!hasActiveImage()) return;

    pendingAnalysis = true;
    pendingIncludeRipeness = pendingIncludeRipeness || includeRipeness;
    clearTimeout(analysisTimer);

    if (immediate) {
        void flushAnalysisQueue();
    } else {
        analysisTimer = setTimeout(() => void flushAnalysisQueue(), 90);
    }
}

async function flushAnalysisQueue() {
    if (!pendingAnalysis || analysisInFlight || !hasActiveImage()) return;

    pendingAnalysis = false;
    const includeRipeness = pendingIncludeRipeness;
    pendingIncludeRipeness = false;
    analysisInFlight = true;
    const epochAtStart = analysisEpoch;
    const startedAt = performance.now();

    setStatus(includeRipeness ? "Analysing image…" : "Updating analysis…", "loading");

    try {
        const data = await requestAnalysis(includeRipeness);
        if (epochAtStart !== analysisEpoch) return;

        const elapsedSeconds = (performance.now() - startedAt) / 1000;
        renderResults(data, elapsedSeconds);
        setStatus("Analysis updated.", "success");
    } catch (error) {
        if (epochAtStart === analysisEpoch) setStatus(error.message, "error");
    } finally {
        analysisInFlight = false;
        if (pendingAnalysis && hasActiveImage()) {
            requestAnimationFrame(() => void flushAnalysisQueue());
        }
    }
}

async function requestAnalysis(includeRipeness) {
    let blob;
    let filename;

    if (selectedSource() === "upload") {
        const file = imageFile.files[0];
        if (!file || !activeImageBlob) throw new Error("Choose an image first.");
        blob = activeImageBlob;
        filename = file.name;
        currentFileName = file.name;
    } else {
        if (!capturedBlob || !activeImageBlob) throw new Error("Capture a camera frame first.");
        blob = activeImageBlob;
        filename = "camera_capture.jpg";
        currentFileName = filename;
    }

    const form = new FormData();
    form.append("image", blob, filename);
    form.append("include_ripeness", includeRipeness ? "1" : "0");

    for (const id of settingIds) {
        form.append(id, document.getElementById(id).value);
    }

    const response = await fetch("/api/analyse", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "Analysis failed.");
    return data;
}

function renderResults(data, elapsedSeconds) {
    const metrics = data.metrics;
    const blemishValue = Number(metrics.blemish_percentage);
    const blemishText = `${blemishValue.toFixed(2)}%`;

    if (!data.ripeness?.skipped && metrics.ripeness) {
        document.getElementById("metric-ripeness").textContent = metrics.ripeness;
        const ripenessNote = {
            Unripe: "Not yet ripe",
            Ripe: "Ready to eat",
            Overripe: "Very mature",
            Rotten: "Not suitable",
        }[metrics.ripeness] || "Prediction ready";
        document.getElementById("metric-ripeness-note").textContent = ripenessNote;
    }

    document.getElementById("metric-blemish").textContent = blemishText;
    document.getElementById("metric-blemish-note").textContent = blemishValue <= 1
        ? "Very low blemish level"
        : blemishValue <= 5
            ? "Low blemish level"
            : blemishValue <= 10
                ? "Moderate blemish level"
                : "High blemish level";

    document.getElementById("metric-quality").textContent = metrics.surface_quality;
    const qualityText = String(metrics.surface_quality || "").toLowerCase();
    document.getElementById("metric-quality-note").textContent = qualityText.includes("extra")
        ? "Premium surface quality"
        : qualityText.includes("class i")
            ? "High quality surface"
            : qualityText.includes("class ii")
                ? "Acceptable surface quality"
                : qualityText.includes("reject")
                    ? "Below quality threshold"
                    : "Quality grade";

    document.getElementById("processing-time").textContent = `Updated in ${elapsedSeconds.toFixed(2)} s`;

    renderImageList("final-images", [
        { caption: "Final output", src: data.images.overlay },
    ]);

    renderMiniPipeline("top-banana-processing", data.processing.banana);
    renderMiniPipeline("top-blemish-processing", (data.processing.blemish || []).slice(0, 3));
    renderRipenessProcessing(data);

    document.getElementById("processing-placeholder").classList.add("hidden");
    document.getElementById("results-section").classList.remove("hidden");
}

function resetAnalysisUI() {
    document.getElementById("results-section").classList.add("hidden");
    document.getElementById("processing-placeholder").classList.remove("hidden");

    document.getElementById("top-banana-processing").innerHTML = `
        <div class="empty-process"><span>1</span><p>Input</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><span>2</span><p>K-means candidate</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><span>3</span><p>Final mask</p></div>`;

    document.getElementById("top-blemish-processing").innerHTML = `
        <div class="empty-process"><span>1</span><p>Banana mask</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><span>2</span><p>K-means candidate</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><span>3</span><p>Final mask</p></div>`;

    document.getElementById("top-ripeness-processing").innerHTML = `
        <div class="empty-process"><span>1</span><p>Classifier input</p></div>
        <div class="ripeness-info-stack">
            <div class="empty-process"><span>3</span><p>Prediction</p></div>
            <div class="empty-process"><span>2</span><p>Class probabilities</p></div>
        </div>`;

    document.getElementById("ripeness-warning").classList.add("hidden");
}

resetAnalysisUI();
