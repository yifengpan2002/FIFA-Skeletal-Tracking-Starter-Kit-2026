import numpy as np
def nanmedian_filter_1d(x, k=5):
    """
    NEW FUNCTION ADDED BY US

    What it does:
    - Applies a sliding-window median filter to a 1D signal.
    - Ignores NaN values using np.nanmedian().

    Why we added it:
    - The baseline pipeline can contain noisy frame-by-frame predictions.
    - A median filter is very good at removing sudden spikes/outliers.

    Why it helps:
    - Reduces jitter in trajectories.
    - More robust than average smoothing when there are bad frames.
    """
    y = x.copy()
    n = len(x)
    half = k // 2
    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        y[i] = np.nanmedian(x[l:r])
    return y


def smooth_xy_sequence(xy, k=5):
    """
    NEW FUNCTION ADDED BY US

    What it does:
    - Applies temporal median smoothing to a 2D sequence of shape (T, 2).
    - Smooths x and y independently through time.

    Why we added it:
    - The baseline uses raw 2D keypoints for translation refinement.
    - Raw 2D detections are often noisy and unstable frame-to-frame.

    Why it helps:
    - Makes 2D hip locations more stable before optimization.
    - Leads to better 3D translation estimation.
    """
    out = xy.copy()
    for d in range(2):
        out[:, d] = nanmedian_filter_1d(out[:, d], k=k)
    return out


def smooth_xyz_sequence(xyz, k=5):
    """
    NEW FUNCTION ADDED BY US

    What it does:
    - Applies temporal median smoothing to a 3D sequence of shape (T, 3).
    - Smooths x, y, z independently through time.

    Why we added it:
    - 3D predictions from the baseline can jitter across frames.
    - Smoothing 3D points before/after optimization improves stability.

    Why it helps:
    - Reduces unrealistic motion noise.
    - Produces smoother trajectories and lower error.
    """
    out = xyz.copy()
    for d in range(3):
        out[:, d] = nanmedian_filter_1d(out[:, d], k=k)
    return out


def remove_3d_spikes(traj, factor=3.0):
    """
    NEW FUNCTION ADDED BY US

    What it does:
    - Detects sudden large jumps ("spikes") in a 3D trajectory.
    - If a point is an obvious outlier compared with its neighbors,
      it replaces it by interpolation between previous and next valid frames.

    Input:
    - traj: (T, 3)

    Why we added it:
    - The baseline may produce occasional large translation jumps in one frame.
    - A single bad frame can hurt the whole sequence quality.

    Why it helps:
    - Removes obvious outlier frames.
    - Makes the final body trajectory much more stable.
    """
    out = traj.copy()
    valid = ~np.isnan(out).any(axis=-1)
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return out

    steps = np.linalg.norm(np.diff(out[idx], axis=0), axis=-1)
    if len(steps) == 0:
        return out

    thr = np.nanmedian(steps) * factor
    if not np.isfinite(thr) or thr <= 0:
        return out

    for j in range(1, len(idx) - 1):
        i = idx[j]
        prev_i = idx[j - 1]
        next_i = idx[j + 1]
        step_prev = np.linalg.norm(out[i] - out[prev_i])
        step_next = np.linalg.norm(out[next_i] - out[i])

        # NEW LOGIC ADDED BY US:
        # If both sides around the current frame are abnormally large,
        # this frame is likely a spike/outlier.
        if step_prev > thr and step_next > thr:
            out[i] = 0.5 * (out[prev_i] + out[next_i])

    return out