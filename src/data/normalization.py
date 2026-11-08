import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

FEATURE_DIM = 150

def normalize_sequence(features, eps=1e-6):

    features = np.asarray(features, dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected (T, {FEATURE_DIM}), got {features.shape}"
        )

    # Convert (T, 150) into (T, 75, 2).
    coords = features.reshape(-1, 75, 2)

    # 1. Extract shoulder
    left = coords[:, LEFT_SHOULDER, :]
    right = coords[:, RIGHT_SHOULDER, :]

    valid = (
        np.isfinite(left).all(axis=1) &
        np.isfinite(right).all(axis=1)
    )

    if not valid.any():
        raise ValueError("No valid shoulder observations.")

    # 2. Calculate recording-level center
    center = ((left[valid] + right[valid]) / 2).mean(axis=0)

    # 3. Calculate recording-level scale
    widths = np.linalg.norm(
        left[valid] - right[valid],
        axis=1
    )

    widths = widths[
        np.isfinite(widths) &
        (widths > eps)
    ]

    if len(widths) == 0:
        raise ValueError("No valid shoulder widths.")

    scale = widths.mean()

    if not np.isfinite(scale) or scale < eps:
        raise ValueError(f"Invalid shoulder scale: {scale}")

    # 4. Normalize valid coordinates
    valid_coords = np.isfinite(coords)

    # 5. Preserve missing coordinates as zero
    normalized = np.zeros_like(coords)

    normalized[valid_coords] = np.broadcast_to(
        (coords - center) / scale,
        coords.shape
    )[valid_coords]

    return normalized.reshape(-1, FEATURE_DIM).astype(np.float32)
