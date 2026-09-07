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

// Optional 2-point line calibration state. Points are stored in preview-
// canvas coordinates and converted back to the active image's natural pixels
// immediately before the Flask request is sent.
let calibrationSourceImage = null;
let calibrationPoints = [];

// Live-analysis queue. Only one request is allowed at a time; repeated slider
// movement is coalesced into the newest pending settings instead of creating a
// backlog of expensive Flask requests.
let analysisInFlight = false;
let pendingAnalysis = false;
let pendingIncludeRipeness = false;
let analysisTimer = null;
let analysisEpoch = 0;

// Batch-upload state.  Flask still processes one image per request; the
// browser sends selected files sequentially so a large folder does not need
// to be uploaded as one huge request.
let batchFiles = [];
let batchResults = [];
let batchMode = false;
let batchProcessing = false;
let batchRunToken = 0;
let batchNeedsRefresh = false;
let activeBatchIndex = null;
let batchDetailInitialising = false;
let nextBatchItemId = 1;

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

// Mobile-only compact result dock. The user explicitly enables it with a
// checkbox. Closing the dock also clears the checkbox, while the latest result
// is retained so checking it again immediately restores the current preview.
const mobileLivePreview = document.getElementById("mobile-live-preview");
const mobilePreviewImage = document.getElementById("mobile-preview-image");
const mobilePreviewState = document.getElementById("mobile-preview-state");
const mobilePreviewRipeness = document.getElementById("mobile-preview-ripeness");
const mobilePreviewBlemish = document.getElementById("mobile-preview-blemish");
const mobilePreviewQuality = document.getElementById("mobile-preview-quality");
const mobilePreviewToggle = document.getElementById("mobile-live-toggle");
const mobilePreviewClose = document.getElementById("mobile-preview-close");
const mobilePreviewMedia = window.matchMedia("(max-width: 700px)");
let latestMobilePreview = null;

function mobilePreviewEnabled() {
    return Boolean(mobilePreviewToggle?.checked && mobilePreviewMedia.matches);
}

function setMobilePreviewUpdating(isUpdating) {
    if (!mobileLivePreview || !mobilePreviewEnabled() || mobileLivePreview.classList.contains("hidden")) return;
    mobileLivePreview.classList.toggle("is-updating", isUpdating);
    if (mobilePreviewState) {
        mobilePreviewState.textContent = isUpdating ? "Updating…" : "Updated";
    }
}

function updateMobilePreviewContents(data, elapsedSeconds) {
    if (!data?.metrics) return;

    const metrics = data.metrics;
    const blemishValue = Number(metrics.blemish_percentage);

    // Blemish-only updates intentionally skip the ripeness model. Keep the
    // previous ripeness value instead of replacing it with a dash.
    if (!data.ripeness?.skipped && metrics.ripeness && mobilePreviewRipeness) {
        mobilePreviewRipeness.textContent = metrics.ripeness;
    }
    if (mobilePreviewBlemish) {
        mobilePreviewBlemish.textContent = Number.isFinite(blemishValue)
            ? `${blemishValue.toFixed(2)}%`
            : "—";
    }
    if (mobilePreviewQuality) {
        mobilePreviewQuality.textContent = metrics.surface_quality || "—";
    }
    if (mobilePreviewImage && data.images?.overlay) {
        mobilePreviewImage.src = data.images.overlay;
    }
    if (mobilePreviewState) {
        mobilePreviewState.textContent = Number.isFinite(elapsedSeconds)
            ? `Updated ${elapsedSeconds.toFixed(1)} s`
            : "Updated";
    }
}

function showMobilePreview(data, elapsedSeconds) {
    if (!mobileLivePreview || !data?.metrics) return;

    latestMobilePreview = { data, elapsedSeconds };
    updateMobilePreviewContents(data, elapsedSeconds);

    if (!mobilePreviewEnabled()) {
        hideMobilePreview({ clearContent: false });
        return;
    }

    mobileLivePreview.classList.remove("hidden", "is-updating");
    document.body.classList.add("mobile-preview-active");
}

function hideMobilePreview({ clearContent = false } = {}) {
    if (!mobileLivePreview) return;
    mobileLivePreview.classList.add("hidden");
    mobileLivePreview.classList.remove("is-updating");
    document.body.classList.remove("mobile-preview-active");

    if (!clearContent) return;

    latestMobilePreview = null;
    if (mobilePreviewImage) mobilePreviewImage.removeAttribute("src");
    if (mobilePreviewRipeness) mobilePreviewRipeness.textContent = "—";
    if (mobilePreviewBlemish) mobilePreviewBlemish.textContent = "—";
    if (mobilePreviewQuality) mobilePreviewQuality.textContent = "—";
    if (mobilePreviewState) mobilePreviewState.textContent = "Ready";
}

if (mobilePreviewToggle) {
    mobilePreviewToggle.addEventListener("change", () => {
        if (!mobilePreviewToggle.checked) {
            hideMobilePreview({ clearContent: false });
            return;
        }

        if (latestMobilePreview && mobilePreviewMedia.matches) {
            showMobilePreview(latestMobilePreview.data, latestMobilePreview.elapsedSeconds);
        }
    });
}

if (mobilePreviewClose) {
    mobilePreviewClose.addEventListener("click", () => {
        if (mobilePreviewToggle) mobilePreviewToggle.checked = false;
        hideMobilePreview({ clearContent: false });
    });
}

function handleMobilePreviewViewportChange() {
    if (!mobilePreviewMedia.matches) {
        hideMobilePreview({ clearContent: false });
        return;
    }

    if (mobilePreviewEnabled() && latestMobilePreview) {
        showMobilePreview(latestMobilePreview.data, latestMobilePreview.elapsedSeconds);
    }
}

if (typeof mobilePreviewMedia.addEventListener === "function") {
    mobilePreviewMedia.addEventListener("change", handleMobilePreviewViewportChange);
} else if (typeof mobilePreviewMedia.addListener === "function") {
    mobilePreviewMedia.addListener(handleMobilePreviewViewportChange);
}

const ripenessMethodInputs = Array.from(
    document.querySelectorAll('input[name="ripeness_method"]')
);

function selectedRipenessMethod() {
    return (
        document.querySelector('input[name="ripeness_method"]:checked')?.value
        || "baseline"
    );
}

function bananaSettingAffectsRipeness(id) {
    return (
        selectedRipenessMethod() === "segmented"
        && bananaSettingIds.has(id)
    );
}

function updateRipenessMethodUI() {
    const method = selectedRipenessMethod();
    const badge = document.getElementById("ripeness-method-badge");
    const help = document.getElementById("ripeness-method-help");

    if (badge) {
        badge.textContent = (
            method === "baseline"
                ? "Baseline model"
                : "Segmented model"
        );
    }

    if (help) {
        help.textContent = (
            method === "baseline"
                ? "Baseline selected: the ripeness classifier receives the original or cropped image directly."
                : "Segmented selected: the final banana mask is applied first and the classifier receives a background-removed image."
        );
    }
}

for (const input of ripenessMethodInputs) {
    input.addEventListener("change", () => {
        // Invalidate an older in-flight response from the previous model.
        analysisEpoch += 1;
        updateRipenessMethodUI();

        if (activeBatchIndex != null) {
            queueAnalysis(true, true);
        } else if (batchResults.length && batchSharedSettingsToggle?.checked) {
            markBatchSettingsDirty();
        } else {
            queueAnalysis(true, true);
        }
    });
}

updateRipenessMethodUI();
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

    if (analyse) queueAnalysis(bananaSettingAffectsRipeness(id));
    return true;
}

function bindSetting(id) {
    const range = document.getElementById(id);
    const number = document.getElementById(`${id}_number`);
    if (!range || !number) return;

    syncNumberFromRange(id);

    range.addEventListener("input", () => {
        syncNumberFromRange(id);
        queueAnalysis(bananaSettingAffectsRipeness(id));
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
        queueAnalysis(bananaSettingAffectsRipeness(id));
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
    queueAnalysis(
        selectedRipenessMethod() === "segmented",
        true
    );
});

// ---------------------------------------------------------------------
// Source selection and upload
// ---------------------------------------------------------------------

const uploadPanel = document.getElementById("upload-panel");
const cameraPanel = document.getElementById("camera-panel");
const videoPanel = document.getElementById("video-panel");
const imageFile = document.getElementById("image-file");
const folderInput = document.getElementById("folder-input");
const uploadMeta = document.getElementById("upload-meta");
const uploadName = document.getElementById("upload-name");
const uploadSize = document.getElementById("upload-size");

const batchResultsSection = document.getElementById("batch-results-section");
const batchResultsBody = document.getElementById("batch-results-body");
const batchSummaryText = document.getElementById("batch-summary-text");
const batchProgressText = document.getElementById("batch-progress-text");
const batchProgressPercent = document.getElementById("batch-progress-percent");
const batchProgressFill = document.getElementById("batch-progress-fill");
const batchExportPdfButton = document.getElementById("batch-export-pdf");
const batchSharedSettingsToggle = document.getElementById("batch-shared-settings");
const batchSettingsWarning = document.getElementById("batch-settings-warning");

for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener("change", () => {
        if (!radio.checked) return;

        const source = radio.value;
        uploadPanel.classList.toggle("hidden", source !== "upload");
        cameraPanel.classList.toggle("hidden", source !== "camera");
        videoPanel?.classList.toggle("hidden", source !== "video");

        if (source === "camera") {
            stopVideoAnalysis({ silent: true });
            updateCameraAvailability();
        } else {
            stopCameraStream();
        }

        if (source !== "video") {
            stopVideoAnalysis({ silent: true });
        } else {
            // Uploaded-video analysis remains available everywhere, but live
            // camera analysis is disabled on insecure/unsupported connections.
            setVideoButtons(videoRunning);
        }
    });
}

function prettyBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

imageFile.addEventListener("change", async () => {
    const files = validImageFiles(Array.from(imageFile.files || []));
    if (folderInput) folderInput.value = "";

    if (files.length === 0) return;

    if (files.length === 1) {
        await activateSingleUpload(files[0]);
    } else {
        await activateBatchUpload(files, { sourceLabel: "images" });
    }
});

if (folderInput) {
    folderInput.addEventListener("change", async () => {
        const files = validImageFiles(Array.from(folderInput.files || []));
        imageFile.value = "";
        if (files.length === 0) return;
        await activateBatchUpload(files, { sourceLabel: "folder" });
    });
}

function validImageFiles(files) {
    const allowedExtensions = /\.(jpe?g|png|webp)$/i;
    return files.filter((file) => (
        allowedExtensions.test(file.name)
        && (!file.type || file.type.startsWith("image/"))
    ));
}

function displayFileName(file) {
    return file.webkitRelativePath || file.name || "image";
}

function totalFileBytes(files) {
    return files.reduce((total, file) => total + Number(file.size || 0), 0);
}

function updateUploadMeta(message = "") {
    if (!uploadMeta) return;
    if (!batchResults.length && !message) {
        uploadMeta.classList.add("hidden");
        return;
    }

    uploadMeta.classList.remove("hidden");
    uploadName.textContent = message || `${batchResults.length} image${batchResults.length === 1 ? "" : "s"} kept in this session`;
    const files = batchResults.map((item) => item.file).filter(Boolean);
    uploadSize.textContent = files.length ? `${prettyBytes(totalFileBytes(files))} total` : "Session images";
}

async function activateSingleUpload(file) {
    stopVideoAnalysis({ silent: true });
    stopCameraStream();
    if (mobileCameraInput) mobileCameraInput.value = "";
    analysisEpoch += 1;

    const [index] = addFilesToBatch([file]);
    updateUploadMeta(`${batchResults.length} image${batchResults.length === 1 ? "" : "s"} kept in this session`);
    await openBatchDetail(index, { scroll: false });
}

async function activateBatchUpload(files, { sourceLabel = "images" } = {}) {
    stopVideoAnalysis({ silent: true });
    stopCameraStream();
    if (mobileCameraInput) mobileCameraInput.value = "";
    analysisEpoch += 1;

    const indices = addFilesToBatch(files);
    updateUploadMeta(
        sourceLabel === "folder"
            ? `${files.length} new images added from folder · ${batchResults.length} kept in session`
            : `${files.length} new images added · ${batchResults.length} kept in session`
    );
    await startBatchAnalysis(indices);
}

function clearUpload() {
    // Clearing the picker no longer removes the session table.  Rows remain
    // until the user deletes them explicitly or closes/reloadss the page.
    imageFile.value = "";
    if (folderInput) folderInput.value = "";
    uploadMeta.classList.add("hidden");
    setStatus("Image picker cleared. Existing Batch Assessment rows were kept.", "info");
}

document.getElementById("clear-upload").addEventListener("click", clearUpload);

// ---------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------

const liveCameraArea = document.getElementById("live-camera-area");
const cameraFallback = document.getElementById("camera-fallback");
const mobileCameraInput = document.getElementById("mobile-camera-input");
const cameraSelect = document.getElementById("camera-select");
const video = document.getElementById("camera-video");
const canvas = document.getElementById("camera-canvas");
const startCamera = document.getElementById("start-camera");
const captureCamera = document.getElementById("capture-camera");
const stopCamera = document.getElementById("stop-camera");

function liveCameraSupported() {
    // getUserMedia is normally available on HTTPS and on localhost/127.0.0.1.
    // A phone opening http://192.168.x.x:5000 is an insecure context, so the
    // browser may intentionally hide navigator.mediaDevices.
    return Boolean(
        window.isSecureContext &&
        navigator.mediaDevices &&
        typeof navigator.mediaDevices.getUserMedia === "function"
    );
}

function updateCameraAvailability() {
    const supported = liveCameraSupported();
    liveCameraArea.classList.toggle("hidden", !supported);
    cameraFallback.classList.toggle("hidden", supported);

    if (!supported) {
        stopCameraStream();
        const activeSource = document.querySelector('input[name="source"]:checked')?.value;
        if (activeSource === "camera") {
            setStatus(
                "Live preview is unavailable on this connection. Use Take photo to use the phone camera.",
                "info"
            );
        }
    }

    return supported;
}

function buildVideoConstraints(deviceId = "") {
    const videoConstraints = {
        width: { ideal: 1280 },
        height: { ideal: 720 },
    };

    if (deviceId) {
        videoConstraints.deviceId = { exact: deviceId };
    } else {
        // On phones this prefers the rear camera. Desktop browsers simply use
        // their normal/default camera when facingMode is not meaningful.
        videoConstraints.facingMode = { ideal: "environment" };
    }

    return { video: videoConstraints, audio: false };
}

async function refreshCameraList(preferredDeviceId = "") {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== "function") {
        return;
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter((device) => device.kind === "videoinput");
        const activeDeviceId = preferredDeviceId ||
            cameraStream?.getVideoTracks?.()[0]?.getSettings?.().deviceId ||
            cameraSelect.value || "";

        cameraSelect.innerHTML = "";

        if (cameras.length === 0) {
            const option = document.createElement("option");
            option.value = "";
            option.textContent = "Default camera";
            cameraSelect.appendChild(option);
            cameraSelect.disabled = true;
            return;
        }

        cameras.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        cameraSelect.disabled = false;
        if (activeDeviceId && cameras.some((camera) => camera.deviceId === activeDeviceId)) {
            cameraSelect.value = activeDeviceId;
        }
    } catch (_) {
        // Camera selection is optional; the default camera can still be used.
    }
}

async function startCameraWithDevice(deviceId = "") {
    if (!updateCameraAvailability()) return;

    try {
        stopCameraStream();
        cameraStream = await navigator.mediaDevices.getUserMedia(
            buildVideoConstraints(deviceId)
        );
        video.srcObject = cameraStream;

        // Calling play explicitly improves behaviour on some browsers even
        // though the <video> element is also autoplay + playsinline.
        try { await video.play(); } catch (_) {}

        captureCamera.disabled = false;
        stopCamera.disabled = false;
        await refreshCameraList(
            cameraStream.getVideoTracks()[0]?.getSettings?.().deviceId || deviceId
        );
        setStatus("Camera ready. Choose a different camera above if needed.", "info");
    } catch (error) {
        stopCameraStream();
        setStatus(`Could not access camera: ${error.message}`, "error");
    }
}

startCamera.addEventListener("click", async () => {
    await startCameraWithDevice(cameraSelect.value);
});

cameraSelect.addEventListener("change", async () => {
    if (!cameraStream) return;
    await startCameraWithDevice(cameraSelect.value);
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
        const name = `camera_capture_${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
        const file = new File([blob], name, { type: "image/jpeg" });
        const [index] = addFilesToBatch([file]);
        capturedBlob = file;
        updateUploadMeta();
        await openBatchDetail(index, { scroll: false });
    }, "image/jpeg", 0.95);
});

mobileCameraInput.addEventListener("change", async () => {
    const file = mobileCameraInput.files[0];
    if (!file) return;

    analysisEpoch += 1;
    stopCameraStream();
    const [index] = addFilesToBatch([file]);
    capturedBlob = file;
    updateUploadMeta();
    setStatus("Photo captured. Analysing image…", "info");
    await openBatchDetail(index, { scroll: false });
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

if (navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", () => {
        if (liveCameraSupported()) void refreshCameraList();
    });
}

// Configure the camera UI immediately. On mobile HTTP this automatically
// exposes the Take photo fallback instead of throwing a getUserMedia error.
updateCameraAvailability();

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
        await initialiseCalibrationEditor(activeImageBlob);
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
    await initialiseCalibrationEditor(activeImageBlob);
    queueAnalysis(true, true);
});

resetCropButton.addEventListener("click", async () => {
    if (!originalImageBlob) return;
    analysisEpoch += 1;
    activeImageBlob = originalImageBlob;
    await initialiseCropEditor(originalImageBlob);
    await initialiseCalibrationEditor(originalImageBlob);
    cropStatus.textContent = "Not cropped";
    cropStatus.classList.remove("applied");
    queueAnalysis(true, true);
});

// ---------------------------------------------------------------------
// Optional spatial calibration using a 2-point reference line
// ---------------------------------------------------------------------

const calibrationEditor = document.getElementById("calibration-editor");
const calibrationCanvas = document.getElementById("calibration-canvas");
const calibrationContext = calibrationCanvas.getContext("2d");
const calibrationEnable = document.getElementById("calibration-enable");
const calibrationStatus = document.getElementById("calibration-status");
const calibrationPointHelp = document.getElementById("calibration-point-help");
const referenceSizeCm = document.getElementById("reference-size-cm");
const clearCalibrationPoints = document.getElementById("clear-calibration-points");

function resetCalibrationEditorState() {
    calibrationSourceImage = null;
    calibrationPoints = [];
    calibrationCanvas.width = 1;
    calibrationCanvas.height = 1;
    calibrationEditor.classList.add("hidden");
    calibrationEnable.checked = false;
    calibrationStatus.textContent = "Off";
    calibrationStatus.classList.remove("applied");
    calibrationPointHelp.textContent = "0 / 2 reference points selected.";
}

async function initialiseCalibrationEditor(blob) {
    try {
        calibrationSourceImage = await blobToImage(blob);
        const maxWidth = 760;
        const maxHeight = 520;
        const scale = Math.min(
            1,
            maxWidth / calibrationSourceImage.naturalWidth,
            maxHeight / calibrationSourceImage.naturalHeight
        );

        calibrationCanvas.width = Math.max(1, Math.round(calibrationSourceImage.naturalWidth * scale));
        calibrationCanvas.height = Math.max(1, Math.round(calibrationSourceImage.naturalHeight * scale));

        // A crop or a newly selected image changes pixel coordinates, so old
        // calibration points must never be reused.
        calibrationPoints = [];
        calibrationEditor.classList.remove("hidden");
        calibrationStatus.textContent = calibrationEnable.checked ? "Select 2 points" : "Off";
        calibrationStatus.classList.remove("applied");
        calibrationPointHelp.textContent = "0 / 2 reference points selected.";
        drawCalibrationEditor();
    } catch (error) {
        setStatus(`Could not prepare calibration preview: ${error.message}`, "error");
    }
}

function drawCalibrationEditor() {
    if (!calibrationSourceImage) return;

    calibrationContext.clearRect(0, 0, calibrationCanvas.width, calibrationCanvas.height);
    calibrationContext.drawImage(
        calibrationSourceImage,
        0, 0, calibrationCanvas.width, calibrationCanvas.height
    );

    if (calibrationPoints.length === 0) return;

    calibrationContext.save();
    calibrationContext.lineWidth = 3;
    calibrationContext.strokeStyle = "#2f7d59";
    calibrationContext.fillStyle = "#ffffff";
    calibrationContext.font = "bold 14px sans-serif";

    calibrationContext.beginPath();
    calibrationContext.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
    for (let i = 1; i < calibrationPoints.length; i += 1) {
        calibrationContext.lineTo(calibrationPoints[i].x, calibrationPoints[i].y);
    }
    calibrationContext.stroke();

    calibrationPoints.forEach((point, index) => {
        calibrationContext.beginPath();
        calibrationContext.arc(point.x, point.y, 8, 0, Math.PI * 2);
        calibrationContext.fillStyle = "#2f7d59";
        calibrationContext.fill();
        calibrationContext.strokeStyle = "#ffffff";
        calibrationContext.lineWidth = 2;
        calibrationContext.stroke();

        calibrationContext.fillStyle = "#ffffff";
        calibrationContext.fillText(String(index + 1), point.x - 4, point.y + 5);
    });
    calibrationContext.restore();
}

function calibrationCanvasPoint(event) {
    const rect = calibrationCanvas.getBoundingClientRect();
    return {
        x: Math.max(0, Math.min(
            calibrationCanvas.width,
            (event.clientX - rect.left) * calibrationCanvas.width / rect.width
        )),
        y: Math.max(0, Math.min(
            calibrationCanvas.height,
            (event.clientY - rect.top) * calibrationCanvas.height / rect.height
        )),
    };
}

function calibrationReady() {
    return Boolean(
        calibrationEnable.checked
        && calibrationSourceImage
        && calibrationPoints.length === 2
        && Number(referenceSizeCm.value) > 0
    );
}

function calibrationPointsInNaturalPixels() {
    if (!calibrationSourceImage || calibrationPoints.length !== 2) return [];
    const scaleX = calibrationSourceImage.naturalWidth / calibrationCanvas.width;
    const scaleY = calibrationSourceImage.naturalHeight / calibrationCanvas.height;
    return calibrationPoints.map((point) => [
        point.x * scaleX,
        point.y * scaleY,
    ]);
}

calibrationCanvas.addEventListener("pointerdown", (event) => {
    if (!calibrationEnable.checked || !calibrationSourceImage) return;

    // Starting a third click begins a fresh 2-point selection.
    if (calibrationPoints.length >= 2) calibrationPoints = [];
    calibrationPoints.push(calibrationCanvasPoint(event));

    const count = calibrationPoints.length;
    calibrationPointHelp.textContent = count === 2
        ? "2 / 2 selected. Calibrated result will update automatically."
        : `${count} / 2 reference points selected.`;
    calibrationStatus.textContent = count === 2 ? "Ready · analysing" : `Point ${count} / 2`;
    calibrationStatus.classList.toggle("applied", count === 2);
    drawCalibrationEditor();

    if (count === 2) {
        analysisEpoch += 1;
        queueAnalysis(true, true);
    }
});

calibrationEnable.addEventListener("change", () => {
    analysisEpoch += 1;
    calibrationStatus.textContent = calibrationEnable.checked
        ? (calibrationPoints.length === 2 ? "Ready" : "Select 2 points")
        : "Off";
    calibrationStatus.classList.toggle(
        "applied",
        calibrationEnable.checked && calibrationPoints.length === 2
    );
    drawCalibrationEditor();

    // Turning calibration off returns immediately to the normal image path.
    // Turning it on only re-analyses after both points are available.
    if (!calibrationEnable.checked || calibrationReady()) {
        queueAnalysis(true, true);
    }
});

referenceSizeCm.addEventListener("change", () => {
    if (calibrationReady()) {
        analysisEpoch += 1;
        queueAnalysis(true, true);
    }
});

clearCalibrationPoints.addEventListener("click", () => {
    const hadAppliedCalibration = calibrationReady();
    calibrationPoints = [];
    calibrationStatus.textContent = calibrationEnable.checked ? "Select 2 points" : "Off";
    calibrationStatus.classList.remove("applied");
    calibrationPointHelp.textContent = "0 / 2 reference points selected.";
    drawCalibrationEditor();

    if (hadAppliedCalibration) {
        analysisEpoch += 1;
        queueAnalysis(true, true);
    }
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
    const badge = document.getElementById("ripeness-method-badge");

    container.innerHTML = "";

    if (badge && data.ripeness?.method_label) {
        badge.textContent = data.ripeness.method_label;
    }

    if (!data.ripeness?.available) {
        const unavailable = document.createElement("div");
        unavailable.className = "ripeness-unavailable";
        unavailable.textContent = "Ripeness prediction unavailable";
        container.appendChild(unavailable);
        warning.textContent = (
            data.ripeness?.error
            || "Ripeness classification is unavailable."
        );
        warning.classList.remove("hidden");
        return;
    }

    warning.classList.add("hidden");

    const inputCaption = (
        data.ripeness?.input_caption
        || "Classifier input"
    );

    const inputCard = makeImageCard(
        data.images.ripeness_input,
        inputCaption
    );

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
        <span>Prediction · ${data.ripeness.method_label || ""}</span>
        <strong>${data.metrics.ripeness || "—"}</strong>
        <small>${confidence}</small>`;

    const probabilityCard = document.createElement("div");
    probabilityCard.className = "ripeness-probability-card";

    const probabilityTitle = document.createElement("span");
    probabilityTitle.textContent = "Class probabilities";

    probabilityCard.append(
        probabilityTitle,
        renderProbabilityRows(
            data.ripeness.probabilities,
            true
        )
    );

    infoStack.append(
        predictionCard,
        probabilityCard
    );

    container.appendChild(infoStack);
}

document.getElementById("show-details-toggle").addEventListener("change", (event) => {
    document.getElementById("top-processing-details").classList.toggle("hidden", !event.target.checked);
});

// ---------------------------------------------------------------------
// Live analysis
// ---------------------------------------------------------------------

function hasActiveImage() {
    if (activeBatchIndex != null && activeImageBlob) return true;
    if (selectedSource() === "upload") return Boolean(activeImageBlob);
    if (selectedSource() === "camera") return Boolean(capturedBlob && activeImageBlob);
    return false;
}

function queueAnalysis(includeRipeness = true, immediate = false) {
    if (videoRunning) return;
    if (batchProcessing) {
        markBatchSettingsDirty();
        return;
    }

    if (activeBatchIndex != null) {
        const item = batchResults[activeBatchIndex];
        if (item && !batchDetailInitialising) {
            item.draftDirty = true;
            item.status = "Unsaved";
            markBatchSettingsDirty();
            renderBatchResults();
        }
    }

    if (!hasActiveImage()) return;

    setMobilePreviewUpdating(true);
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
        if (epochAtStart === analysisEpoch) {
            setStatus(error.message, "error");
            setMobilePreviewUpdating(false);
        }
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

    if (activeBatchIndex != null) {
        const item = batchResults[activeBatchIndex];
        if (!item || !activeImageBlob) throw new Error("Open a Batch Assessment image first.");
        blob = activeImageBlob;
        filename = item.displayName;
        currentFileName = filename;
    } else if (selectedSource() === "upload") {
        if (!activeImageBlob) throw new Error("Choose an image first.");
        blob = activeImageBlob;
        filename = currentFileName || "banana.jpg";
    } else {
        if (!capturedBlob || !activeImageBlob) throw new Error("Capture a camera frame first.");
        blob = activeImageBlob;
        filename = currentFileName || "camera_capture.jpg";
    }

    return requestAnalysisForBlob(blob, filename, {
        includeRipeness,
        compactResponse: false,
    });
}

async function requestAnalysisForBlob(
    blob,
    filename,
    {
        includeRipeness = true,
        compactResponse = false,
        settingsOverride = null,
        calibrationOverride = null,
        ripenessMethodOverride = null,
    } = {}
) {
    const form = new FormData();
    form.append("image", blob, filename);
    form.append("include_ripeness", includeRipeness ? "1" : "0");
    form.append("ripeness_method", ripenessMethodOverride || selectedRipenessMethod());
    form.append("compact_response", compactResponse ? "1" : "0");

    const calibrationState = calibrationOverride == null
        ? snapshotCalibrationState()
        : explicitCalibrationForRequest(calibrationOverride);
    const useCalibration = !videoRunning && Boolean(calibrationState?.enabled);
    form.append("calibration_enabled", useCalibration ? "1" : "0");
    if (useCalibration) {
        form.append("reference_size_cm", String(calibrationState.referenceSizeCm));
        form.append("calibration_points", JSON.stringify(calibrationState.points));
    }

    const settings = settingsOverride || snapshotAnalysisSettings();
    for (const id of settingIds) {
        form.append(id, settings[id] ?? document.getElementById(id).value);
    }

    const response = await fetch("/api/analyse", {
        method: "POST",
        body: form,
    });

    const data = await response.json().catch(() => ({ok: false, error: `Server returned HTTP ${response.status}. Check the server terminal for details.`}));
    if (!response.ok || !data.ok) {
        throw new Error(data.error || "Analysis failed.");
    }

    return data;
}

function snapshotAnalysisSettings() {
    const snapshot = {};
    for (const id of settingIds) {
        snapshot[id] = document.getElementById(id).value;
    }
    return snapshot;
}

function applyAnalysisSettings(snapshot = {}) {
    for (const id of settingIds) {
        if (snapshot[id] == null) continue;
        const range = document.getElementById(id);
        const number = document.getElementById(`${id}_number`);
        if (!range) continue;
        range.value = String(snapshot[id]);
        if (number) number.value = range.value;
    }
}

function applyRipenessMethod(method) {
    const value = method === "segmented" ? "segmented" : "baseline";
    const input = document.querySelector(`input[name="ripeness_method"][value="${value}"]`);
    if (input) input.checked = true;
    updateRipenessMethodUI();
}

function snapshotCalibrationState() {
    return {
        enabled: calibrationReady(),
        referenceSizeCm: Number(referenceSizeCm.value) || 0,
        points: calibrationReady() ? calibrationPointsInNaturalPixels() : [],
    };
}

async function restoreCalibrationState(snapshot = null) {
    const state = snapshot || { enabled: false, referenceSizeCm: 10, points: [] };
    referenceSizeCm.value = String(Number(state.referenceSizeCm) > 0 ? state.referenceSizeCm : 10);
    calibrationEnable.checked = Boolean(state.enabled);
    calibrationPoints = [];

    if (
        state.enabled
        && calibrationSourceImage
        && Array.isArray(state.points)
        && state.points.length === 2
    ) {
        const scaleX = calibrationCanvas.width / calibrationSourceImage.naturalWidth;
        const scaleY = calibrationCanvas.height / calibrationSourceImage.naturalHeight;
        calibrationPoints = state.points.map((point) => ({
            x: Number(point[0]) * scaleX,
            y: Number(point[1]) * scaleY,
        }));
    }

    const ready = calibrationEnable.checked && calibrationPoints.length === 2;
    calibrationStatus.textContent = ready
        ? "Ready"
        : calibrationEnable.checked
            ? "Select 2 points"
            : "Off";
    calibrationStatus.classList.toggle("applied", ready);
    calibrationPointHelp.textContent = `${calibrationPoints.length} / 2 reference points selected.`;
    drawCalibrationEditor();
}

function makeBatchItem(file) {
    return {
        id: nextBatchItemId++,
        file,
        workingBlob: file,
        displayName: displayFileName(file),
        settings: snapshotAnalysisSettings(),
        ripenessMethod: selectedRipenessMethod(),
        calibration: { enabled: false, referenceSizeCm: Number(referenceSizeCm?.value) || 10, points: [] },
        ok: false,
        data: null,
        draftData: null,
        draftDirty: false,
        error: null,
        status: "Queued",
        elapsedSeconds: 0,
    };
}

function addFilesToBatch(files) {
    const indices = [];
    for (const file of files) {
        batchFiles.push(file);
        batchResults.push(makeBatchItem(file));
        indices.push(batchResults.length - 1);
    }
    batchMode = batchResults.length > 0;
    batchResultsSection?.classList.toggle("hidden", !batchMode);
    updateBatchSummary();
    renderBatchResults();
    return indices;
}

function updateBatchSummary() {
    if (!batchSummaryText) return;
    const total = batchResults.length;
    if (!total) {
        batchSummaryText.textContent = "Upload/capture an image or add a selected video frame to this session.";
        return;
    }
    const complete = batchResults.filter((item) => item.ok && item.data).length;
    batchSummaryText.textContent = `${total} image/frame${total === 1 ? "" : "s"} kept in this session · ${complete} analysed. View a row to crop, calibrate or tune its own settings, then Save. PDF export uses only this table.`;
}

function markBatchSettingsDirty() {
    if (!batchResults.length) return;
    batchNeedsRefresh = true;
    batchSettingsWarning?.classList.remove("hidden");
}

function refreshBatchWarning() {
    const hasUnsaved = batchResults.some((item) => item.draftDirty);
    const sharedNeedsRefresh = Boolean(batchSharedSettingsToggle?.checked && batchNeedsRefresh);
    batchSettingsWarning?.classList.toggle("hidden", !(hasUnsaved || sharedNeedsRefresh));
}

function setBatchProgress(processed, total, label = "") {
    const safeTotal = Math.max(1, total);
    const percent = Math.round(processed / safeTotal * 100);
    if (batchProgressText) batchProgressText.textContent = label || `Processed ${processed} of ${total}`;
    if (batchProgressPercent) batchProgressPercent.textContent = `${percent}%`;
    if (batchProgressFill) batchProgressFill.style.width = `${percent}%`;
}

function setBatchControlsDisabled(disabled) {
    if (batchExportPdfButton) batchExportPdfButton.disabled = disabled || !batchResults.some((item) => item.ok && item.data);
    imageFile.disabled = disabled;
    if (folderInput) folderInput.disabled = disabled;
}

function explicitCalibrationForRequest(state) {
    if (!state?.enabled || !Array.isArray(state.points) || state.points.length !== 2) {
        return { enabled: false, referenceSizeCm: Number(state?.referenceSizeCm) || 10, points: [] };
    }
    return {
        enabled: true,
        referenceSizeCm: Number(state.referenceSizeCm),
        points: state.points.map((point) => [Number(point[0]), Number(point[1])]),
    };
}

async function startBatchAnalysis(indices = null) {
    const targets = (indices || batchResults.map((_, index) => index))
        .filter((index) => batchResults[index]);
    if (!targets.length || batchProcessing) return;

    const runToken = ++batchRunToken;
    batchProcessing = true;
    batchMode = batchResults.length > 0;
    batchNeedsRefresh = false;
    batchSettingsWarning?.classList.add("hidden");
    batchResultsSection?.classList.remove("hidden");

    const useSharedSettings = Boolean(batchSharedSettingsToggle?.checked);
    const sharedSettings = useSharedSettings ? snapshotAnalysisSettings() : null;

    setBatchControlsDisabled(true);
    setBatchProgress(0, targets.length, `Preparing ${targets.length} image${targets.length === 1 ? "" : "s"}…`);

    let processed = 0;
    for (const index of targets) {
        if (runToken !== batchRunToken) break;
        const item = batchResults[index];
        if (!item) continue;

        item.status = "Analysing";
        item.error = null;
        renderBatchResults();

        const settings = useSharedSettings ? sharedSettings : (item.settings || snapshotAnalysisSettings());
        const ripenessMethod = item.ripenessMethod || selectedRipenessMethod();
        const calibration = explicitCalibrationForRequest(item.calibration);
        const blob = item.workingBlob || item.file;
        const startedAt = performance.now();

        setBatchProgress(processed, targets.length, `Analysing ${processed + 1} of ${targets.length}: ${item.displayName}`);
        setStatus(`Analysing ${item.displayName}…`, "loading");

        try {
            const data = await requestAnalysisForBlob(blob, item.file?.name || "image.jpg", {
                includeRipeness: true,
                compactResponse: true,
                settingsOverride: settings,
                calibrationOverride: calibration,
                ripenessMethodOverride: ripenessMethod,
            });
            if (runToken !== batchRunToken) break;
            data.filename = item.displayName;

            item.ok = true;
            item.data = data;
            item.draftData = null;
            item.draftDirty = false;
            item.error = null;
            item.status = "Complete";
            item.elapsedSeconds = (performance.now() - startedAt) / 1000;
            if (useSharedSettings) item.settings = { ...sharedSettings };
        } catch (error) {
            item.ok = false;
            item.error = error.message;
            item.status = "Failed";
            item.elapsedSeconds = (performance.now() - startedAt) / 1000;
        }

        processed += 1;
        renderBatchResults();
        updateBatchSummary();
        setBatchProgress(processed, targets.length);
    }

    batchProcessing = false;
    setBatchControlsDisabled(false);
    renderBatchResults();
    updateBatchSummary();
    setBatchProgress(processed, targets.length, `Completed ${processed} image${processed === 1 ? "" : "s"}`);
    setStatus("Batch assessment updated.", "success");
}

function appendBatchCell(row, content, className = "") {
    if (content instanceof HTMLElement) {
        if (className) content.classList.add(className);
        row.appendChild(content);
        return;
    }
    const cell = document.createElement("td");
    cell.textContent = String(content);
    if (className) cell.classList.add(className);
    row.appendChild(cell);
}

function svgIconButton({ title, className = "", svg, onClick, disabled = false }) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `batch-icon-button ${className}`.trim();
    button.title = title;
    button.setAttribute("aria-label", title);
    button.innerHTML = svg;
    button.disabled = disabled;
    button.addEventListener("click", onClick);
    return button;
}

const EYE_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.8"/></svg>`;
const SAVE_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 4h12l2 2v14H5z"/><path d="M8 4v6h8V4M8 20v-6h8v6"/></svg>`;
const TRASH_SVG = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5"/></svg>`;

function renderBatchResults() {
    if (!batchResultsBody) return;
    batchResultsBody.innerHTML = "";

    batchResults.forEach((item, index) => {
        const row = document.createElement("tr");
        if (index === activeBatchIndex) row.classList.add("is-active");
        if (item.draftDirty) row.classList.add("has-unsaved");

        const imageCell = document.createElement("td");
        imageCell.className = "batch-image-cell";
        const thumb = document.createElement("img");
        thumb.className = "batch-thumbnail";
        thumb.alt = "";
        if (item.data?.images?.overlay) thumb.src = item.data.images.overlay;
        const name = document.createElement("span");
        name.textContent = item.displayName;
        name.title = item.displayName;
        imageCell.append(thumb, name);

        const metrics = item.data?.metrics || {};
        const confidence = metrics.ripeness_confidence == null
            ? "—"
            : `${(Number(metrics.ripeness_confidence) * 100).toFixed(1)}%`;
        const blemish = Number.isFinite(Number(metrics.blemish_percentage))
            ? `${Number(metrics.blemish_percentage).toFixed(2)}%`
            : "—";
        const calibration = item.data?.calibration || {};
        const size = calibration.applied
            ? `${Number(calibration.length_cm).toFixed(1)} × ${Number(calibration.width_cm).toFixed(1)} cm`
            : "N/A";

        appendBatchCell(row, imageCell);
        appendBatchCell(row, item.ok ? (metrics.ripeness || "—") : "—");
        appendBatchCell(row, item.ok ? confidence : "—");
        appendBatchCell(row, item.ok ? blemish : "—");
        appendBatchCell(row, item.ok ? (metrics.surface_quality || "—") : "—");
        appendBatchCell(row, item.ok ? size : "—");
        appendBatchCell(
            row,
            item.draftDirty ? "Unsaved" : item.status,
            item.status === "Failed" ? "batch-status-failed" : item.draftDirty ? "batch-status-unsaved" : "batch-status-success"
        );

        const actionCell = document.createElement("td");
        actionCell.className = "batch-actions-cell";
        const viewButton = svgIconButton({
            title: "View details",
            className: "view",
            svg: EYE_SVG,
            disabled: batchProcessing,
            onClick: () => void openBatchDetail(index),
        });
        const saveButton = svgIconButton({
            title: "Save current details and settings",
            className: "save",
            svg: SAVE_SVG,
            disabled: batchProcessing || index !== activeBatchIndex,
            onClick: () => void saveBatchItem(index),
        });
        const deleteButton = svgIconButton({
            title: "Delete from batch",
            className: "delete",
            svg: TRASH_SVG,
            disabled: batchProcessing,
            onClick: () => deleteBatchItem(index),
        });
        actionCell.append(viewButton, saveButton, deleteButton);
        row.appendChild(actionCell);
        batchResultsBody.appendChild(row);
    });

    batchMode = batchResults.length > 0;
    batchResultsSection?.classList.toggle("hidden", !batchMode);
    if (batchExportPdfButton) batchExportPdfButton.disabled = !batchResults.some((item) => item.ok && item.data);
}

async function saveBatchItem(index) {
    const item = batchResults[index];
    if (!item || index !== activeBatchIndex) {
        setStatus("Open this image with the eye button before saving its settings.", "error");
        return;
    }
    if (analysisInFlight || batchProcessing) {
        setStatus("Wait for the current analysis to finish, then save.", "info");
        return;
    }

    item.settings = snapshotAnalysisSettings();
    item.ripenessMethod = selectedRipenessMethod();
    item.calibration = snapshotCalibrationState();
    item.workingBlob = activeImageBlob || item.workingBlob || item.file;

    const candidate = item.draftData || latestReportResult;
    if (candidate) {
        item.data = mergeRipenessFromPrevious(candidate, item.data);
        item.ok = true;
    }

    item.draftData = null;
    item.draftDirty = false;
    item.status = "Saved";
    item.error = null;
    batchNeedsRefresh = false;

    refreshBatchWarning();
    renderBatchResults();
    updateBatchSummary();

    if (batchSharedSettingsToggle?.checked && batchResults.length > 1) {
        const shared = { ...item.settings };
        batchResults.forEach((other) => {
            other.settings = { ...shared };
        });

        const otherIndices = batchResults
            .map((_, otherIndex) => otherIndex)
            .filter((otherIndex) => otherIndex !== index);

        setStatus(`Saved ${item.displayName}. Applying the same banana and blemish settings to the remaining images…`, "info");
        await startBatchAnalysis(otherIndices);
        item.status = "Saved";
        renderBatchResults();
        updateBatchSummary();
        refreshBatchWarning();
        setStatus(`Saved ${item.displayName} and updated the shared settings for all batch images.`, "success");
        return;
    }

    setStatus(`Saved settings, calibration, ripeness and confidence for ${item.displayName}.`, "success");
}

function deleteBatchItem(index) {
    const item = batchResults[index];
    if (!item) return;

    batchRunToken += 1;
    batchResults.splice(index, 1);
    batchFiles.splice(index, 1);

    if (activeBatchIndex === index) {
        activeBatchIndex = null;
        originalImageBlob = null;
        activeImageBlob = null;
        currentFileName = "banana.jpg";
        resetCropEditorState();
        resetCalibrationEditorState();
        resetAnalysisUI();
    } else if (activeBatchIndex != null && activeBatchIndex > index) {
        activeBatchIndex -= 1;
    }

    batchMode = batchResults.length > 0;
    renderBatchResults();
    updateBatchSummary();
    updateUploadMeta();
    refreshBatchWarning();
    setStatus(`Removed ${item.displayName} from Batch Assessment.`, "success");
}

async function openBatchDetail(index, { scroll = true } = {}) {
    const item = batchResults[index];
    if (!item?.file) return;
    const wasUnanalysed = !item.data;

    activeBatchIndex = index;
    batchDetailInitialising = true;
    analysisEpoch += 1;
    pendingAnalysis = false;
    clearTimeout(analysisTimer);

    originalImageBlob = item.file;
    activeImageBlob = item.workingBlob || item.file;
    capturedBlob = null;
    currentFileName = item.displayName;

    applyAnalysisSettings(item.settings || {});
    applyRipenessMethod(item.ripenessMethod || "baseline");
    await initialiseCropEditor(activeImageBlob);
    await initialiseCalibrationEditor(activeImageBlob);
    await restoreCalibrationState(item.calibration);

    renderBatchResults();
    setStatus(`Loading detailed analysis for ${item.displayName}…`, "loading");
    const startedAt = performance.now();

    try {
        const data = await requestAnalysisForBlob(activeImageBlob, item.file?.name || "image.jpg", {
            includeRipeness: true,
            compactResponse: false,
            settingsOverride: snapshotAnalysisSettings(),
            calibrationOverride: snapshotCalibrationState(),
            ripenessMethodOverride: selectedRipenessMethod(),
        });
        const elapsedSeconds = (performance.now() - startedAt) / 1000;
        data.filename = item.displayName;
        renderResults(data, elapsedSeconds);
        if (wasUnanalysed) {
            item.ok = true;
            item.data = data;
            item.settings = snapshotAnalysisSettings();
            item.ripenessMethod = selectedRipenessMethod();
            item.calibration = snapshotCalibrationState();
            item.workingBlob = activeImageBlob || item.file;
            item.status = "Complete";
            item.draftData = null;
        } else {
            item.draftData = data;
            item.status = item.status === "Saved" ? "Saved" : "Complete";
        }
        item.draftDirty = false;
        updateBatchSummary();
        if (scroll) {
            document.getElementById("results-section")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        setStatus(`Detailed result loaded for ${item.displayName}. Change settings if needed, then Save.`, "success");
    } catch (error) {
        if (wasUnanalysed) {
            item.ok = false;
            item.error = error.message;
            item.status = "Failed";
        }
        setStatus(error.message, "error");
    } finally {
        batchDetailInitialising = false;
        renderBatchResults();
        updateBatchSummary();
        refreshBatchWarning();
    }
}

if (batchSharedSettingsToggle) {
    batchSharedSettingsToggle.addEventListener("change", () => {
        if (!batchResults.length) return;
        if (batchSharedSettingsToggle.checked) {
            markBatchSettingsDirty();
            setStatus("Shared settings enabled. Save the opened image to apply its current banana and blemish settings to every batch image automatically.", "info");
        } else {
            setStatus("Individual image settings enabled. Each row keeps its saved settings.", "info");
        }
    });
}

async function downloadPdfReport(results, button, filename) {
    const successful = results.filter((result) => result && !result.error);
    if (!successful.length) throw new Error("No successful assessment results are available for PDF export.");

    button.disabled = true;
    try {
        const response = await fetch("/api/report", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ results: successful.slice(-200) }),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || "Report export failed.");
        }
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
    } finally {
        button.disabled = false;
    }
}

if (batchExportPdfButton) {
    batchExportPdfButton.addEventListener("click", async () => {
        try {
            const results = batchResults
                .filter((item) => item.ok && item.data)
                .map((item) => ({ ...item.data, filename: item.displayName }));
            await downloadPdfReport(results, batchExportPdfButton, "banana_batch_assessment.pdf");
        } catch (error) {
            setStatus(error.message, "error");
        }
    });
}


function mergeRipenessFromPrevious(nextData, previousData) {
    if (!nextData || !nextData.metrics) return nextData;
    if (!nextData.ripeness?.skipped) return nextData;

    const previousMetrics = previousData?.metrics;
    if (!previousMetrics?.ripeness) return nextData;

    return {
        ...nextData,
        metrics: {
            ...nextData.metrics,
            ripeness: previousMetrics.ripeness,
            ripeness_confidence: previousMetrics.ripeness_confidence,
        },
    };
}

function renderResults(data, elapsedSeconds) {
    const previousResult = activeBatchIndex != null
        ? batchResults[activeBatchIndex]?.data
        : latestReportResult;
    data = mergeRipenessFromPrevious(data, previousResult);
    latestReportResult = data;
    if (activeBatchIndex != null && batchResults[activeBatchIndex]) {
        const item = batchResults[activeBatchIndex];
        item.draftData = data;
        if (!batchDetailInitialising) {
            item.draftDirty = true;
            item.status = "Unsaved";
            markBatchSettingsDirty();
            renderBatchResults();
        }
    }
    renderMiniPipeline("top-preprocessing", data.processing.preprocessing || []);
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
        : qualityText === "class i"
            ? "High quality surface"
            : qualityText.includes("class ii")
                ? "Acceptable surface quality"
                : qualityText.includes("reject")
                    ? "Below quality threshold"
                    : "Quality grade";

    const sizeCard = document.getElementById("metric-size-card");
    if (data.calibration?.applied && data.calibration?.length_cm != null) {
        const length = Number(data.calibration.length_cm);
        const width = Number(data.calibration.width_cm);
        sizeCard.classList.remove("hidden");
        document.getElementById("metric-size").textContent = `${length.toFixed(1)} × ${width.toFixed(1)} cm`;
        document.getElementById("metric-size-note").textContent = `Scale ${Number(data.calibration.pixels_per_cm).toFixed(1)} px/cm`;
    } else {
        sizeCard.classList.remove("hidden");
        document.getElementById("metric-size").textContent = "N/A";
        document.getElementById("metric-size-note").textContent = "2-point calibration required";
    }

    document.getElementById("processing-time").textContent = `Updated in ${elapsedSeconds.toFixed(2)} s`;

    renderImageList("final-images", [
        { caption: "Final output", src: data.images.overlay },
    ]);

    renderMiniPipeline("top-banana-processing", data.processing.banana);
    renderMiniPipeline("top-blemish-processing", (data.processing.blemish || []).slice(0, 3));
    renderRipenessProcessing(data);
    showMobilePreview(data, elapsedSeconds);

    document.getElementById("processing-placeholder").classList.add("hidden");
    document.getElementById("results-section").classList.remove("hidden");
}

function resetAnalysisUI() {
    hideMobilePreview({ clearContent: true });
    document.getElementById("metric-size-card")?.classList.remove("hidden");
    const metricSize = document.getElementById("metric-size");
    const metricSizeNote = document.getElementById("metric-size-note");
    if (metricSize) metricSize.textContent = "N/A";
    if (metricSizeNote) metricSizeNote.textContent = "2-point calibration required";
    document.getElementById("results-section").classList.add("hidden");
    document.getElementById("processing-placeholder").classList.remove("hidden");

    document.getElementById("top-banana-processing").innerHTML = `
        <div class="empty-process"><p>Input</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><p>K-means candidate</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><p>Final mask</p></div>`;

    document.getElementById("top-blemish-processing").innerHTML = `
        <div class="empty-process"><p>Banana mask</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><p>K-means candidate</p></div>
        <div class="process-arrow"></div>
        <div class="empty-process"><p>Final mask</p></div>`;

    document.getElementById("top-ripeness-processing").innerHTML = `
        <div class="empty-process"><p>Classifier input</p></div>
        <div class="ripeness-info-stack">
            <div class="empty-process"><p>Prediction</p></div>
            <div class="empty-process"><p>Class probabilities</p></div>
        </div>`;

    document.getElementById("ripeness-warning").classList.add("hidden");
}

resetAnalysisUI();

let latestReportResult = null;
let videoRunning = false;
let videoGeneration = 0;
let videoObjectUrl = null;
let liveVideoStream = null;
let selectedVideoFile = null;
let lastVideoFrame = null;
let videoFramesAnalysed = 0;
let frozenVideoFrameUrl = null;
let videoInputMode = null;

const analysisVideo = document.getElementById("analysis-video");
const videoFrozenFrame = document.getElementById("video-frozen-frame");
const videoFileInput = document.getElementById("video-file");
const videoStartButton = document.getElementById("video-start");
const videoCameraButton = document.getElementById("video-camera");
const videoStopButton = document.getElementById("video-stop");
const videoAddFrameButton = document.getElementById("video-add-frame");
const videoStatus = document.getElementById("video-status");

function clearFrozenVideoFrame() {
    if (frozenVideoFrameUrl) URL.revokeObjectURL(frozenVideoFrameUrl);
    frozenVideoFrameUrl = null;
    if (videoFrozenFrame) {
        videoFrozenFrame.src = "";
        videoFrozenFrame.classList.add("hidden");
    }
}

function showFrozenVideoFrame(blob) {
    clearFrozenVideoFrame();
    if (!videoFrozenFrame || !blob) return;
    frozenVideoFrameUrl = URL.createObjectURL(blob);
    videoFrozenFrame.src = frozenVideoFrameUrl;
    videoFrozenFrame.classList.remove("hidden");
    analysisVideo?.classList.add("hidden");
}

function showAnalysisVideo() {
    clearFrozenVideoFrame();
    analysisVideo?.classList.remove("hidden");
}

function resetVideoFrameSelection() {
    lastVideoFrame = null;
    videoFramesAnalysed = 0;
    clearFrozenVideoFrame();
}

function setVideoButtons(running) {
    if (videoStartButton) videoStartButton.disabled = running;

    const liveSupported = liveCameraSupported();
    if (videoCameraButton) {
        videoCameraButton.disabled = running || !liveSupported;
        videoCameraButton.title = liveSupported
            ? "Analyse frames continuously from the live camera."
            : "Live camera analysis requires a secure HTTPS/localhost connection. Upload a video on this device instead.";
        videoCameraButton.setAttribute("aria-disabled", String(videoCameraButton.disabled));
    }

    if (videoStopButton) videoStopButton.disabled = !running;
    // A frame must first be analysed. Stop the stream before preserving it so
    // the user knows exactly which still frame is being added to the batch.
    if (videoAddFrameButton) {
        videoAddFrameButton.disabled = running || !lastVideoFrame?.blob;
    }
}

function stopVideoAnalysis({ silent = false } = {}) {
    const wasRunning = videoRunning;
    const wasLive = Boolean(liveVideoStream || analysisVideo?.srcObject);
    videoRunning = false;
    videoGeneration += 1;

    if (analysisVideo) analysisVideo.pause();
    if (liveVideoStream) {
        liveVideoStream.getTracks().forEach((track) => track.stop());
    }
    liveVideoStream = null;
    if (analysisVideo?.srcObject) analysisVideo.srcObject = null;

    // Uploaded videos can remain paused on the selected timestamp so the user
    // may scrub with the normal video controls before adding a frame. A live
    // stream has no seekable timeline, so preserve the latest analysed frame as
    // a frozen still image after stopping.
    if (wasLive && lastVideoFrame?.blob) {
        showFrozenVideoFrame(lastVideoFrame.blob);
    }

    setVideoButtons(false);

    if (!silent && videoStatus && wasRunning) {
        videoStatus.textContent = lastVideoFrame?.blob
            ? `Stopped after ${videoFramesAnalysed} analysed frame${videoFramesAnalysed === 1 ? "" : "s"}. The current frame is ready to add to Batch Assessment for crop, calibration and manual tuning.`
            : "Video analysis stopped before a frame was completed.";
    }
}

function waitForVideoEvent(element, eventName) {
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
            cleanup();
            reject(new Error(`Timed out while waiting for video ${eventName}.`));
        }, 10000);
        const onEvent = () => { cleanup(); resolve(); };
        const onError = () => { cleanup(); reject(new Error("The selected video could not be read.")); };
        function cleanup() {
            clearTimeout(timeout);
            element.removeEventListener(eventName, onEvent);
            element.removeEventListener("error", onError);
        }
        element.addEventListener(eventName, onEvent, { once: true });
        element.addEventListener("error", onError, { once: true });
    });
}

async function captureAnalysisVideoFrame() {
    if (!analysisVideo || analysisVideo.videoWidth <= 0 || analysisVideo.videoHeight <= 0) {
        throw new Error("Video frame is not ready yet.");
    }
    const frameCanvas = document.createElement("canvas");
    const scale = Math.min(1, 960 / Math.max(analysisVideo.videoWidth, analysisVideo.videoHeight));
    frameCanvas.width = Math.max(1, Math.round(analysisVideo.videoWidth * scale));
    frameCanvas.height = Math.max(1, Math.round(analysisVideo.videoHeight * scale));
    frameCanvas.getContext("2d").drawImage(
        analysisVideo,
        0,
        0,
        frameCanvas.width,
        frameCanvas.height
    );
    const blob = await new Promise((resolve) => frameCanvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) throw new Error("Could not extract the current video frame.");
    return blob;
}

async function analyseVideoFrame(blob, frameName, displayName, token, metadata = {}) {
    const startedAt = performance.now();
    const data = await requestAnalysisForBlob(blob, frameName, {
        includeRipeness: true,
        compactResponse: false,
        calibrationOverride: { enabled: false, referenceSizeCm: 10, points: [] },
    });
    if (token !== videoGeneration || !videoRunning) return null;

    data.filename = displayName;
    const elapsedSeconds = (performance.now() - startedAt) / 1000;
    renderResults(data, elapsedSeconds);
    videoFramesAnalysed += 1;
    lastVideoFrame = {
        blob,
        frameName,
        displayName,
        data,
        source: metadata.source || videoInputMode || "video",
        timeSeconds: Number.isFinite(Number(metadata.timeSeconds)) ? Number(metadata.timeSeconds) : null,
        frameNumber: Number(metadata.frameNumber) || videoFramesAnalysed,
    };
    return elapsedSeconds;
}

if (videoFileInput) {
    videoFileInput.addEventListener("change", async () => {
        stopVideoAnalysis({ silent: true });
        if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
        videoObjectUrl = null;
        selectedVideoFile = videoFileInput.files?.[0] || null;
        videoInputMode = selectedVideoFile ? "uploaded" : null;
        resetVideoFrameSelection();
        setVideoButtons(false);

        if (!selectedVideoFile) {
            analysisVideo?.classList.add("hidden");
            if (videoStatus) {
                videoStatus.textContent = "Select a video or start live analysis. Video frames are not stored automatically; add a specific frame to Batch Assessment if you want to keep it.";
            }
            return;
        }

        videoObjectUrl = URL.createObjectURL(selectedVideoFile);
        analysisVideo.srcObject = null;
        analysisVideo.src = videoObjectUrl;
        showAnalysisVideo();
        if (videoStatus) {
            videoStatus.textContent = `${selectedVideoFile.name} selected. Choose Analyse uploaded video. Stop on a useful frame, then add it to Batch Assessment.`;
        }
    });
}

async function runUploadedVideoAnalysis() {
    if (!selectedVideoFile || !videoObjectUrl) {
        setStatus("Select a video file first.", "error");
        return;
    }

    stopVideoAnalysis({ silent: true });
    stopCameraStream();
    activeBatchIndex = null;
    activeImageBlob = null;
    originalImageBlob = null;
    resetCropEditorState();
    resetCalibrationEditorState();
    renderBatchResults();
    analysisEpoch += 1;
    resetVideoFrameSelection();
    videoInputMode = "uploaded";

    const token = ++videoGeneration;
    videoRunning = true;
    setVideoButtons(true);

    try {
        analysisVideo.srcObject = null;
        analysisVideo.src = videoObjectUrl;
        showAnalysisVideo();
        analysisVideo.pause();
        if (analysisVideo.readyState < 1) await waitForVideoEvent(analysisVideo, "loadedmetadata");

        const duration = Number(analysisVideo.duration);
        if (!Number.isFinite(duration) || duration <= 0) {
            throw new Error("Could not determine the selected video duration.");
        }

        // Sample roughly once per second. For long videos, cap the automatic
        // analysis at about 60 frames so the interface remains responsive.
        const sampleStep = Math.max(1, duration / 60);
        const sampleTimes = [];
        for (let time = 0; time <= duration && sampleTimes.length < 60; time += sampleStep) {
            sampleTimes.push(Math.min(time, Math.max(0, duration - 0.01)));
        }

        for (let index = 0; index < sampleTimes.length; index += 1) {
            if (!videoRunning || token !== videoGeneration) break;
            const time = sampleTimes[index];
            if (Math.abs(analysisVideo.currentTime - time) > 0.02) {
                analysisVideo.currentTime = time;
                await waitForVideoEvent(analysisVideo, "seeked");
            }

            const blob = await captureAnalysisVideoFrame();
            const displayName = `${selectedVideoFile.name} @ ${time.toFixed(1)} s`;
            if (videoStatus) {
                videoStatus.textContent = `Analysing frame ${index + 1} of ${sampleTimes.length} · ${time.toFixed(1)} s`;
            }
            await analyseVideoFrame(
                blob,
                `video_frame_${String(index + 1).padStart(3, "0")}.jpg`,
                displayName,
                token,
                { source: "uploaded", timeSeconds: time, frameNumber: index + 1 }
            );
        }

        if (token === videoGeneration && videoStatus) {
            videoStatus.textContent = `Completed ${videoFramesAnalysed} sampled frame${videoFramesAnalysed === 1 ? "" : "s"}. Use the video controls to choose a frame if needed, then Add current frame to Batch Assessment.`;
        }
    } catch (error) {
        if (token === videoGeneration) {
            if (videoStatus) videoStatus.textContent = error.message;
            setStatus(error.message, "error");
        }
    } finally {
        if (token === videoGeneration) {
            videoRunning = false;
            setVideoButtons(false);
        }
    }
}

async function runLiveVideoAnalysis() {
    if (!liveCameraSupported()) {
        const message = "Live camera analysis is unavailable on this connection. On mobile/local HTTP, upload a recorded video instead.";
        if (videoStatus) videoStatus.textContent = message;
        setStatus(message, "info");
        setVideoButtons(false);
        return;
    }

    stopVideoAnalysis({ silent: true });
    stopCameraStream();
    activeBatchIndex = null;
    activeImageBlob = null;
    originalImageBlob = null;
    resetCropEditorState();
    resetCalibrationEditorState();
    renderBatchResults();
    analysisEpoch += 1;
    resetVideoFrameSelection();
    videoInputMode = "live";

    const token = ++videoGeneration;
    try {
        liveVideoStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: "environment" } },
            audio: false,
        });
        if (token !== videoGeneration) {
            liveVideoStream.getTracks().forEach((track) => track.stop());
            return;
        }

        analysisVideo.src = "";
        analysisVideo.srcObject = liveVideoStream;
        showAnalysisVideo();
        await analysisVideo.play();
        videoRunning = true;
        setVideoButtons(true);

        let count = 0;
        const sessionStart = performance.now();
        while (videoRunning && token === videoGeneration) {
            const cycleStart = performance.now();
            if (analysisVideo.readyState < 2) {
                await new Promise((resolve) => setTimeout(resolve, 100));
                continue;
            }

            const blob = await captureAnalysisVideoFrame();
            count += 1;
            const elapsedLive = (performance.now() - sessionStart) / 1000;
            if (videoStatus) {
                videoStatus.textContent = `Live analysis · frame ${count} · ${elapsedLive.toFixed(1)} s`;
            }
            await analyseVideoFrame(
                blob,
                `live_frame_${String(count).padStart(3, "0")}.jpg`,
                `Live camera frame ${count} @ ${elapsedLive.toFixed(1)} s`,
                token,
                { source: "live", timeSeconds: elapsedLive, frameNumber: count }
            );

            // Analyse at most about one frame per second and never queue a new
            // request before the previous K-means/CNN analysis finishes.
            const remaining = Math.max(0, 1000 - (performance.now() - cycleStart));
            if (remaining) await new Promise((resolve) => setTimeout(resolve, remaining));
        }
    } catch (error) {
        if (token === videoGeneration) {
            if (videoStatus) videoStatus.textContent = error.message;
            setStatus(error.message, "error");
        }
    } finally {
        if (token === videoGeneration) {
            videoRunning = false;
            if (liveVideoStream) liveVideoStream.getTracks().forEach((track) => track.stop());
            liveVideoStream = null;
            analysisVideo.srcObject = null;
            if (lastVideoFrame?.blob) showFrozenVideoFrame(lastVideoFrame.blob);
            setVideoButtons(false);
        }
    }
}

function safeVideoFrameFileName(name) {
    return String(name || "video_frame.jpg")
        .replace(/[^a-zA-Z0-9._-]+/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "") || "video_frame.jpg";
}

async function addCurrentVideoFrameToBatch() {
    if (videoRunning) {
        setStatus("Stop video analysis first so the selected frame stays fixed.", "info");
        return;
    }
    if (batchProcessing) {
        setStatus("Wait for the current batch analysis to finish before adding another video frame.", "info");
        return;
    }
    if (!lastVideoFrame?.blob) {
        setStatus("Analyse at least one video frame before adding it to Batch Assessment.", "error");
        return;
    }

    try {
        if (videoAddFrameButton) videoAddFrameButton.disabled = true;
        let blob = lastVideoFrame.blob;
        let displayName = lastVideoFrame.displayName || "Video frame";
        let fileName = lastVideoFrame.frameName || "video_frame.jpg";

        // For an uploaded video, the user may scrub to another timestamp after
        // analysis stops/completes. Capture that exact visible timestamp when
        // Add current frame is pressed. The batch will analyse it normally.
        if (
            lastVideoFrame.source === "uploaded"
            && selectedVideoFile
            && videoObjectUrl
            && analysisVideo
            && !analysisVideo.srcObject
            && analysisVideo.readyState >= 2
            && analysisVideo.videoWidth > 0
        ) {
            blob = await captureAnalysisVideoFrame();
            const time = Number(analysisVideo.currentTime) || 0;
            const baseName = selectedVideoFile.name.replace(/\.[^.]+$/, "");
            fileName = `${baseName}_frame_${time.toFixed(1).replace(".", "_")}s.jpg`;
            displayName = `${selectedVideoFile.name} @ ${time.toFixed(1)} s`;
        }

        const file = new File([blob], safeVideoFrameFileName(fileName), {
            type: blob.type || "image/jpeg",
            lastModified: Date.now(),
        });
        const [index] = addFilesToBatch([file]);
        const item = batchResults[index];
        if (item) {
            item.displayName = displayName;
            item.settings = snapshotAnalysisSettings();
            item.ripenessMethod = selectedRipenessMethod();
            item.calibration = {
                enabled: false,
                referenceSizeCm: Number(referenceSizeCm?.value) || 10,
                points: [],
            };
            item.workingBlob = file;
            item.status = "Queued";
        }

        updateUploadMeta(`${batchResults.length} image/frame${batchResults.length === 1 ? "" : "s"} kept in this session`);
        renderBatchResults();
        setStatus(`Adding ${displayName} to Batch Assessment and analysing it…`, "loading");
        await startBatchAnalysis([index]);
        renderBatchResults();
        updateBatchSummary();
        setStatus(`${displayName} was added to Batch Assessment. Use the eye button to crop, calibrate or fine-tune its settings before saving.`, "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        setVideoButtons(videoRunning);
    }
}

// Apply browser/connection availability immediately. This keeps Start live
// analysis visibly disabled on mobile local-HTTP connections while leaving
// uploaded-video analysis enabled.
setVideoButtons(false);

if (videoStartButton) videoStartButton.addEventListener("click", () => void runUploadedVideoAnalysis());
if (videoCameraButton) videoCameraButton.addEventListener("click", () => void runLiveVideoAnalysis());
if (videoStopButton) videoStopButton.addEventListener("click", () => stopVideoAnalysis());
if (videoAddFrameButton) videoAddFrameButton.addEventListener("click", () => void addCurrentVideoFrameToBatch());

window.addEventListener("beforeunload", () => {
    stopVideoAnalysis({ silent: true });
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    clearFrozenVideoFrame();
});
