let cameraStream = null;
let capturedBlob = null;
let currentFileName = "camera_capture.jpg";
let analysisStartedAt = null;

// Optional client-side crop state.
// The Flask backend receives the cropped blob when a crop has been applied.
let originalImageBlob = null;
let activeImageBlob = null;
let cropSourceImage = null;
let cropSelection = null;
let cropDragging = false;
let cropStartPoint = null;

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

function updateRangeLabel(id) {
    const input = document.getElementById(id);
    const target = document.getElementById(`${id}_value`);
    if (input && target) target.textContent = input.value;
}

for (const id of settingIds) {
    const element = document.getElementById(id);
    if (!element) continue;
    element.addEventListener("input", () => updateRangeLabel(id));
    updateRangeLabel(id);
}

document.getElementById("reset-settings").addEventListener("click", () => {
    for (const id of settingIds) {
        const input = document.getElementById(id);
        if (!input) continue;
        input.value = input.dataset.default;
        updateRangeLabel(id);
    }
});

const uploadPanel = document.getElementById("upload-panel");
const cameraPanel = document.getElementById("camera-panel");

for (const radio of document.querySelectorAll('input[name="source"]')) {
    radio.addEventListener("change", () => {
        const useCamera = radio.value === "camera" && radio.checked;
        uploadPanel.classList.toggle("hidden", useCamera);
        cameraPanel.classList.toggle("hidden", !useCamera);
    });
}

const imageFile = document.getElementById("image-file");
const uploadPreview = document.getElementById("upload-preview");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const uploadMeta = document.getElementById("upload-meta");
const uploadName = document.getElementById("upload-name");
const uploadSize = document.getElementById("upload-size");

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

    currentFileName = file.name;
    originalImageBlob = file;
    activeImageBlob = file;

    uploadPreview.src = URL.createObjectURL(file);
    uploadPreview.classList.remove("hidden");
    uploadPlaceholder.classList.add("hidden");
    uploadMeta.classList.remove("hidden");
    uploadName.textContent = file.name;
    uploadSize.textContent = prettyBytes(file.size);

    await initialiseCropEditor(file);
});

function clearUpload() {
    imageFile.value = "";
    uploadPreview.src = "";
    uploadPreview.classList.add("hidden");
    uploadPlaceholder.classList.remove("hidden");
    uploadMeta.classList.add("hidden");
    currentFileName = "banana.jpg";
    originalImageBlob = null;
    activeImageBlob = null;
    resetCropEditorState();
}

document.getElementById("clear-upload").addEventListener("click", clearUpload);

const video = document.getElementById("camera-video");
const canvas = document.getElementById("camera-canvas");
const cameraPreview = document.getElementById("camera-preview");
const startCamera = document.getElementById("start-camera");
const captureCamera = document.getElementById("capture-camera");
const stopCamera = document.getElementById("stop-camera");

startCamera.addEventListener("click", async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
            },
            audio: false,
        });

        video.srcObject = cameraStream;
        captureCamera.disabled = false;
        stopCamera.disabled = false;
        setStatus("Camera started. Capture one frame when ready.", "info");
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

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        capturedBlob = blob;
        originalImageBlob = blob;
        activeImageBlob = blob;
        currentFileName = "camera_capture.jpg";
        cameraPreview.src = URL.createObjectURL(blob);
        cameraPreview.classList.remove("hidden");
        await initialiseCropEditor(blob);
        setStatus("Frame captured and ready for analysis.", "success");
    }, "image/jpeg", 0.95);
});

stopCamera.addEventListener("click", stopCameraStream);

function stopCameraStream() {
    if (cameraStream) {
        for (const track of cameraStream.getTracks()) {
            track.stop();
        }
        cameraStream = null;
    }

    video.srcObject = null;
    captureCamera.disabled = true;
    stopCamera.disabled = true;
}

window.addEventListener("beforeunload", stopCameraStream);


// ---------------------------------------------------------------------
// Optional crop editor
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

async function blobToImage(blob) {
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

        // Keep the crop editor responsive without allocating an enormous canvas.
        const maxWidth = 760;
        const maxHeight = 520;
        const scale = Math.min(
            1,
            maxWidth / cropSourceImage.naturalWidth,
            maxHeight / cropSourceImage.naturalHeight
        );

        cropCanvas.width = Math.max(
            1,
            Math.round(cropSourceImage.naturalWidth * scale)
        );
        cropCanvas.height = Math.max(
            1,
            Math.round(cropSourceImage.naturalHeight * scale)
        );

        // Start with the whole image selected.
        cropSelection = {
            x: 0,
            y: 0,
            width: cropCanvas.width,
            height: cropCanvas.height,
        };

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
    cropContext.drawImage(
        cropSourceImage,
        0,
        0,
        cropCanvas.width,
        cropCanvas.height
    );

    if (!cropSelection) return;

    // Darken everything outside the selected rectangle.
    cropContext.save();
    cropContext.fillStyle = "rgba(10, 24, 18, 0.42)";
    cropContext.fillRect(0, 0, cropCanvas.width, cropCanvas.height);

    cropContext.clearRect(
        cropSelection.x,
        cropSelection.y,
        cropSelection.width,
        cropSelection.height
    );

    // Redraw the selected image region so it remains fully visible.
    const sx = cropSelection.x / cropCanvas.width * cropSourceImage.naturalWidth;
    const sy = cropSelection.y / cropCanvas.height * cropSourceImage.naturalHeight;
    const sw = cropSelection.width / cropCanvas.width * cropSourceImage.naturalWidth;
    const sh = cropSelection.height / cropCanvas.height * cropSourceImage.naturalHeight;

    cropContext.drawImage(
        cropSourceImage,
        sx, sy, sw, sh,
        cropSelection.x,
        cropSelection.y,
        cropSelection.width,
        cropSelection.height
    );

    cropContext.strokeStyle = "#f1c84b";
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
        x: Math.max(
            0,
            Math.min(
                cropCanvas.width,
                (event.clientX - rect.left) * cropCanvas.width / rect.width
            )
        ),
        y: Math.max(
            0,
            Math.min(
                cropCanvas.height,
                (event.clientY - rect.top) * cropCanvas.height / rect.height
            )
        ),
    };
}

cropCanvas.addEventListener("pointerdown", (event) => {
    if (!cropSourceImage) return;

    cropDragging = true;
    cropStartPoint = canvasPointFromEvent(event);
    cropSelection = {
        x: cropStartPoint.x,
        y: cropStartPoint.y,
        width: 0,
        height: 0,
    };

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

    if (
        !cropSelection ||
        cropSelection.width < 12 ||
        cropSelection.height < 12
    ) {
        cropSelection = {
            x: 0,
            y: 0,
            width: cropCanvas.width,
            height: cropCanvas.height,
        };
    }

    if (event?.pointerId != null) {
        try {
            cropCanvas.releasePointerCapture(event.pointerId);
        } catch (_) {}
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
    const sw = Math.min(
        cropSourceImage.naturalWidth - sx,
        Math.max(1, Math.round(cropSelection.width * scaleX))
    );
    const sh = Math.min(
        cropSourceImage.naturalHeight - sy,
        Math.max(1, Math.round(cropSelection.height * scaleY))
    );

    // Ignore a crop that is effectively the whole image.
    const fullArea = cropSourceImage.naturalWidth * cropSourceImage.naturalHeight;
    const cropArea = sw * sh;

    if (cropArea / fullArea > 0.995) {
        activeImageBlob = originalImageBlob;
        cropStatus.textContent = "Full image";
        cropStatus.classList.remove("applied");
        setStatus("Full image will be analysed.", "info");
        return;
    }

    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = sw;
    outputCanvas.height = sh;

    const outputContext = outputCanvas.getContext("2d");
    outputContext.drawImage(
        cropSourceImage,
        sx, sy, sw, sh,
        0, 0, sw, sh
    );

    const croppedBlob = await new Promise((resolve) => {
        outputCanvas.toBlob(resolve, "image/jpeg", 0.96);
    });

    if (!croppedBlob) {
        setStatus("Could not create the cropped image.", "error");
        return;
    }

    activeImageBlob = croppedBlob;
    cropStatus.textContent = "Crop applied";
    cropStatus.classList.add("applied");

    const croppedUrl = URL.createObjectURL(croppedBlob);

    if (selectedSource() === "upload") {
        uploadPreview.src = croppedUrl;
    } else {
        cameraPreview.src = croppedUrl;
    }

    setStatus(
        "Crop applied. K-means segmentation will still run on the cropped image.",
        "success"
    );
});

resetCropButton.addEventListener("click", async () => {
    if (!originalImageBlob) return;

    activeImageBlob = originalImageBlob;
    cropStatus.textContent = "Not cropped";
    cropStatus.classList.remove("applied");

    const originalUrl = URL.createObjectURL(originalImageBlob);

    if (selectedSource() === "upload") {
        uploadPreview.src = originalUrl;
    } else {
        cameraPreview.src = originalUrl;
    }

    await initialiseCropEditor(originalImageBlob);
    setStatus("Crop reset. The original image will be analysed.", "info");
});

function selectedSource() {
    return document.querySelector('input[name="source"]:checked').value;
}

function setStatus(message, type = "info") {
    const status = document.getElementById("status");
    status.textContent = message;
    status.className = `status ${type}`;
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
    if (event.key === "Escape" && !imageModal.classList.contains("hidden")) {
        closeImageModal();
    }
});

function makeImageCard(src, caption) {
    if (!src) return null;

    const figure = document.createElement("figure");
    const img = document.createElement("img");
    const cap = document.createElement("figcaption");

    img.src = src;
    img.alt = caption;
    img.loading = "lazy";
    img.title = "Click to enlarge";
    img.classList.add("expandable-image");
    img.addEventListener("click", () => openImageModal(src, caption));

    cap.textContent = caption;
    figure.append(img, cap);
    return figure;
}

function renderImageList(containerId, items) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = "";

    for (const item of items || []) {
        if (!item || !item.src) continue;
        const card = makeImageCard(item.src, item.caption);
        if (card) container.appendChild(card);
    }
}

function renderMiniPipeline(containerId, items) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = "";

    items.forEach((item, index) => {
        const card = makeImageCard(item.src, item.caption);
        if (card) {
            card.classList.add("mini-card");
            container.appendChild(card);
        }

        if (index < items.length - 1) {
            const arrow = document.createElement("div");
            arrow.className = "process-arrow";
            arrow.textContent = "→";
            container.appendChild(arrow);
        }
    });
}

function renderTable(containerId, rows) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";

    if (!rows || rows.length === 0) {
        container.textContent = "No rows available.";
        return;
    }

    const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");

    for (const column of columns) {
        const th = document.createElement("th");
        th.textContent = column;
        headRow.appendChild(th);
    }

    thead.appendChild(headRow);

    const tbody = document.createElement("tbody");

    for (const row of rows) {
        const tr = document.createElement("tr");

        for (const column of columns) {
            const td = document.createElement("td");
            const value = row[column];
            td.textContent = value == null ? "" : String(value);
            tr.appendChild(td);
        }

        tbody.appendChild(tr);
    }

    table.append(thead, tbody);
    container.appendChild(table);
}

function renderProbabilities(probabilities) {
    const container = document.getElementById("probability-list");
    container.innerHTML = "";

    const entries = Object.entries(probabilities || {}).sort(
        (a, b) => Number(b[1]) - Number(a[1])
    );

    for (const [name, raw] of entries) {
        const probability = Math.max(0, Math.min(1, Number(raw)));

        const row = document.createElement("div");
        row.className = "probability-row";
        row.innerHTML = `
            <span>${name}</span>
            <div class="progress">
                <div class="progress-fill" style="width:${probability * 100}%"></div>
            </div>
            <strong>${(probability * 100).toFixed(1)}%</strong>
        `;

        container.appendChild(row);
    }
}

document.getElementById("show-details-toggle").addEventListener("change", (event) => {
    document.getElementById("top-processing-details")
        .classList.toggle("hidden", !event.target.checked);
});

async function analyseImage() {
    const source = selectedSource();

    let blob = null;
    let filename = "banana.jpg";

    if (source === "upload") {
        const file = imageFile.files[0];

        if (!file || !activeImageBlob) {
            setStatus("Choose an image first.", "error");
            return;
        }

        blob = activeImageBlob;
        filename = file.name;
        currentFileName = file.name;
    } else {
        if (!capturedBlob || !activeImageBlob) {
            setStatus("Capture a camera frame first.", "error");
            return;
        }

        blob = activeImageBlob;
        filename = "camera_capture.jpg";
        currentFileName = filename;
    }

    const form = new FormData();
    form.append("image", blob, filename);

    for (const id of settingIds) {
        form.append(id, document.getElementById(id).value);
    }

    const button = document.getElementById("analyse-button");
    button.disabled = true;
    analysisStartedAt = performance.now();
    setStatus("Analysing banana…", "info");

    try {
        const response = await fetch("/api/analyse", {
            method: "POST",
            body: form,
        });

        const data = await response.json();

        if (!response.ok || !data.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        const elapsedSeconds = (performance.now() - analysisStartedAt) / 1000;
        renderResults(data, elapsedSeconds);
        setStatus("Analysis complete.", "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        button.disabled = false;
    }
}

document.getElementById("analyse-button").addEventListener("click", analyseImage);

function renderResults(data, elapsedSeconds) {
    const metrics = data.metrics;

    const confidenceText =
        metrics.ripeness_confidence == null
            ? "—"
            : `${(metrics.ripeness_confidence * 100).toFixed(1)}%`;

    const blemishText = `${metrics.blemish_percentage.toFixed(2)}%`;

    document.getElementById("metric-ripeness").textContent = metrics.ripeness;
    document.getElementById("metric-confidence").textContent = confidenceText;
    document.getElementById("metric-blemish").textContent = blemishText;
    document.getElementById("metric-quality").textContent = metrics.surface_quality;

    document.getElementById("summary-image-name").textContent = currentFileName;
    document.getElementById("summary-ripeness").textContent = metrics.ripeness;
    document.getElementById("summary-confidence").textContent = confidenceText;
    document.getElementById("summary-blemish").textContent = blemishText;
    document.getElementById("summary-quality").textContent = metrics.surface_quality;

    document.getElementById("processing-time").textContent =
        `Processed in ${elapsedSeconds.toFixed(2)} s`;

    renderImageList("final-images", [
        { caption: "Input image", src: data.images.original },
        { caption: "Final banana mask", src: data.images.banana_mask },
        { caption: "Final blemish mask", src: data.images.blemish_mask },
        { caption: "Detection overlay", src: data.images.overlay },
    ]);

    renderMiniPipeline("top-banana-processing", [
        { caption: "1. Input", src: data.processing.banana?.[0]?.src },
        { caption: "2. K-means candidate", src: data.processing.banana?.[5]?.src },
        { caption: "3. Final mask", src: data.images.banana_mask },
    ]);

    renderMiniPipeline("top-blemish-processing", [
        { caption: "1. Banana mask", src: data.processing.blemish?.[0]?.src },
        { caption: "2. K-means candidate", src: data.processing.blemish?.[4]?.src },
        { caption: "3. Combined candidate", src: data.processing.blemish?.[6]?.src },
        { caption: "4. Final mask", src: data.images.blemish_mask },
        { caption: "5. Overlay", src: data.images.overlay },
    ]);

    document.getElementById("processing-placeholder").classList.add("hidden");

    renderImageList("ripeness-images", [
        {
            caption: "Background-removed classifier input",
            src: data.images.ripeness_segmented,
        },
        {
            caption: "Shared banana mask",
            src: data.images.ripeness_mask,
        },
    ]);

    renderProbabilities(data.ripeness.probabilities);

    const warning = document.getElementById("ripeness-warning");

    if (!data.ripeness.available) {
        warning.textContent =
            data.ripeness.error || "Ripeness classification is unavailable.";
        warning.classList.remove("hidden");
    } else {
        warning.classList.add("hidden");
    }

    renderImageList("banana-processing", data.processing.banana);
    renderImageList("blemish-processing", data.processing.blemish);

    renderTable("cluster-table", data.cluster_summary);
    renderTable("blemish-cluster-table", data.blemish_cluster_summary);

    document.getElementById("blemish-cluster-ids").textContent =
        JSON.stringify(
            {
                healthy_cluster_ids: data.healthy_cluster_ids,
                blemish_cluster_ids: data.blemish_cluster_ids,
            },
            null,
            2
        );

    document.getElementById("results-section").classList.remove("hidden");
    document.getElementById("results-section").scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}
