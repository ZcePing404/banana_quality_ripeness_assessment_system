import cv2
import numpy as np


K = 4
KMEANS_ATTEMPTS = 10
BACKGROUND_VALUE = 255


def preprocess_image(image_bgr):
    """
    Apply light smoothing before segmentation.

    The Gaussian filter reduces small colour variations/noise
    that may cause unnecessary K-means clusters.
    """
    blurred = cv2.GaussianBlur(
        image_bgr,
        (5, 5),
        0
    )

    return blurred

def run_kmeans(image_bgr, k=4):
    """
    Cluster image pixels according to colour using K-means.

    LAB colour space is used because it separates brightness
    from colour information better than raw BGR.
    """

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    height, width = lab.shape[:2]

    pixels = lab.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS
    )

    label_image = labels.reshape((height, width))

    return label_image, centers

def choose_banana_cluster(image_bgr, label_image, k):
    """
    Select the K-means cluster that most likely belongs
    to the banana.

    The score considers:

    1. Banana-like hue (green/yellow/brown range)
    2. Saturation
    3. Presence near the image centre
    4. Penalty for clusters dominating the image border

    This helps avoid selecting plain backgrounds.
    """

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    height, width = image_bgr.shape[:2]

    # Central region of image.
    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    y1 = int(height * 0.20)
    y2 = int(height * 0.80)

    # Border region.
    border_width = max(3, int(min(height, width) * 0.04))

    border_mask = np.zeros((height, width), dtype=np.uint8)

    border_mask[:border_width, :] = 1
    border_mask[-border_width:, :] = 1
    border_mask[:, :border_width] = 1
    border_mask[:, -border_width:] = 1

    best_cluster = None
    best_score = -999999

    for cluster_id in range(k):

        cluster_mask = label_image == cluster_id

        pixel_count = np.count_nonzero(cluster_mask)

        if pixel_count == 0:
            continue

        area_ratio = pixel_count / (height * width)

        hsv_pixels = hsv[cluster_mask]

        mean_h = float(np.mean(hsv_pixels[:, 0]))
        mean_s = float(np.mean(hsv_pixels[:, 1]))
        mean_v = float(np.mean(hsv_pixels[:, 2]))

        # ----------------------------------------------------
        # Banana colour score
        # ----------------------------------------------------

        colour_score = 0

        # Brown / yellow / green ranges commonly appearing
        # on bananas at different ripeness stages.
        if 5 <= mean_h <= 85:
            colour_score += 4

        if mean_s >= 40:
            colour_score += 2

        if mean_v >= 40:
            colour_score += 1

        # ----------------------------------------------------
        # Centrality score
        # ----------------------------------------------------

        centre_pixels = np.count_nonzero(
            cluster_mask[y1:y2, x1:x2]
        )

        centre_ratio = centre_pixels / max(pixel_count, 1)

        # ----------------------------------------------------
        # Border penalty
        # ----------------------------------------------------

        border_pixels = np.count_nonzero(
            cluster_mask & (border_mask == 1)
        )

        border_ratio = border_pixels / max(pixel_count, 1)

        # ----------------------------------------------------
        # Area constraint
        # ----------------------------------------------------

        area_score = 0

        if 0.03 <= area_ratio <= 0.85:
            area_score = 2
        else:
            area_score = -3

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        score = (
            colour_score
            + (centre_ratio * 5)
            - (border_ratio * 5)
            + area_score
        )

        if score > best_score:
            best_score = score
            best_cluster = cluster_id

    return best_cluster

def create_banana_mask(label_image, banana_cluster):
    """
    Convert the selected cluster into a binary mask.
    """

    mask = np.zeros(
        label_image.shape,
        dtype=np.uint8
    )

    mask[label_image == banana_cluster] = 255

    return mask

def clean_mask(mask):
    """
    Clean the K-means mask using morphological operations.

    Opening:
        removes isolated noise.

    Closing:
        reconnects small gaps within the banana region.
    """

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=2
    )

    return mask

def keep_largest_object(mask):
    """
    Keep the largest connected contour.

    This is useful because your classification experiment
    currently assumes one primary banana in each image.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    final_mask = np.zeros_like(mask)

    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return final_mask

def remove_background(image_bgr, mask):
    """
    Keep banana pixels and replace everything else
    with a fixed white background.
    """

    output = np.full_like(
        image_bgr,
        BACKGROUND_VALUE
    )

    output[mask == 255] = image_bgr[mask == 255]

    return output

def refine_mask_with_grabcut(image_bgr, initial_mask):
    """
    Refine the K-means mask using GrabCut.

    K-means identifies the approximate banana region.
    GrabCut then expands/refines the region based on image
    boundaries instead of relying only on colour.
    """

    h, w = initial_mask.shape

    grabcut_mask = np.full(
        (h, w),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Slightly expand the initial banana region.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )

    expanded = cv2.dilate(
        initial_mask,
        kernel,
        iterations=2
    )

    # Expanded region = probably foreground
    grabcut_mask[expanded > 0] = cv2.GC_PR_FGD

    # Original K-means region = definite foreground
    grabcut_mask[initial_mask > 0] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image_bgr,
        grabcut_mask,
        None,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    final_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) |
        (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return final_mask

def segment_banana(image_bgr):

    # 1. Light preprocessing
    prepared = preprocess_image(image_bgr)

    # 2. K-means gives an approximate banana region
    label_image, _ = run_kmeans(
        prepared,
        k=K
    )

    banana_cluster = choose_banana_cluster(
        prepared,
        label_image,
        K
    )

    initial_mask = create_banana_mask(
        label_image,
        banana_cluster
    )

    # 3. Clean initial K-means mask
    initial_mask = clean_mask(initial_mask)

    initial_mask = keep_largest_object(
        initial_mask
    )

    # 4. IMPORTANT:
    # Refine using GrabCut so differently coloured
    # peel regions can be recovered.
    refined_mask = refine_mask_with_grabcut(
        image_bgr,
        initial_mask
    )

    # 5. Light cleanup only
    refined_mask = clean_final_mask(
        refined_mask
    )

    # 6. Remove background
    segmented = remove_background(
        image_bgr,
        refined_mask
    )

    return segmented, refined_mask

def clean_final_mask(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Fill small gaps without aggressively removing peel.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return mask

    # 5. Light cleanup only
    refined_mask = clean_final_mask(
        refined_mask
    )

    # 6. Remove background
    segmented = remove_background(
        image_bgr,
        refined_mask
    )

    return segmented, refined_mask
