"""
This script provides a naive baseline for FIFA Skeletal Tracking Challenge.

Author: Tianjian Jiang
Date: Nov 10, 2025
"""
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.optim as optim
from tqdm import tqdm
from lib.camera_tracker import CameraTracker, CameraTrackerOptions
from lib.postprocess import smoothen
# from smoothing_help import (
#     nanmedian_filter_1d,
#     smooth_xy_sequence,
#     smooth_xyz_sequence,
#     remove_3d_spikes,
# )
import subprocess
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
OPENPOSE_TO_OURS = [0, 2, 5, 3, 6, 4, 7, 9, 12, 10, 13, 11, 14, 22, 19]


def intersection_over_plane(o, d):
    """
    args:
        o: (3,) origin of the ray
        d: (3,) direction of the ray

    returns:
        intersection: (3,) intersection point
    """
    # solve the x and y where z = 0
    t = -o[2] / d[2]
    return o + t * d


def ray_from_xy(xy, K, R, t, k1=0.0, k2=0.0):
    """
    Compute the ray from the camera center through the image point (x, y),
    correcting for radial distortion using coefficients k1 and k2.

    Args:
        xy: (2,) array_like containing pixel coordinates [x, y] in the image.
        K: (3, 3) ndarray representing the camera intrinsic matrix.
        R: (3, 3) ndarray representing the camera rotation matrix.
        t: (3,) ndarray representing the camera translation vector.
        k1: float, the first radial distortion coefficient (default 0).
        k2: float, the second radial distortion coefficient (default 0).

    Returns:
        origin: (3,) ndarray representing the camera center in world coordinates.
        direction: (3,) unit ndarray representing the direction of the ray in world coordinates.
    """
    # Convert the pixel coordinate to homogeneous coordinates.
    p = np.array([xy[0], xy[1], 1.0])

    # Compute the normalized coordinate (distorted) in the camera coordinate system.
    p_norm = np.linalg.inv(K) @ p  # p_norm = [x_d, y_d, 1]
    x_d, y_d = p_norm[0], p_norm[1]

    # Compute the radial distance (squared) in the normalized plane.
    r2 = x_d**2 + y_d**2
    # Compute the distortion factor.
    factor = 1 + k1 * r2 + k2 * (r2**2)

    # Correct the distorted normalized coordinates.
    x_undist = x_d / factor
    y_undist = y_d / factor

    # Construct the undistorted direction in camera coordinates (z = 1).
    d_cam = np.array([x_undist, y_undist, 1.0])

    # Transform the direction to world coordinates.
    direction = R.T @ d_cam
    direction = direction / np.linalg.norm(direction)

    # The camera center in world coordinates is given by -R^T t.
    origin = -R.T @ t
    return origin, direction


def project_points_th(obj_pts, R, C, K, k):
    """Projects 3D points onto 2D image plane using camera intrinsics and distortion.

    args:
        obj_pts: (N, 3) - 3D points in world space
        R: (3, 3) - Rotation matrix
        C: (3,) - Camera center
        K: (3, 3) - Camera intrinsic matrix
        k: (5,) - Distortion coefficients

    returns:
        img_pts: (N, 2) - Projected 2D points
    """

    # Transform world points to camera coordinates
    pts_c = (R @ ((obj_pts - C).unsqueeze(-1))).squeeze(-1)

    # Normalize to get image plane coordinates
    img_pts = pts_c[:, :2] / pts_c[:, 2:]

    # Compute radial distortion
    r2 = (img_pts**2).sum(dim=-1, keepdim=True)
    r2 = torch.clamp(r2, 0, 0.5 / min(max(torch.abs(k).max().item(), 1.0), 1.0))
    p = torch.arange(1, k.shape[-1] + 1, device=k.device)
    img_pts = img_pts * (torch.ones_like(r2) + (k * r2.pow(p)).sum(-1, keepdim=True))

    # Apply intrinsics K
    img_pts_h = torch.cat([img_pts, torch.ones_like(img_pts[:, :1])], dim=-1)  # Homogeneous coords
    img_pts = (K @ img_pts_h.unsqueeze(-1)).squeeze(-1)[:, :2]  # Convert back to 2D

    return img_pts


def minimize_reprojection_error(pts_3d, pts_2d, R, C, K, k, iterations=10):
    """
    Optimize 3D points to minimize reprojection error.

    args:
        pts_3d: (N, 3)  - Initial 3D points (learnable)
        pts_2d: (N, 2)  - Corresponding 2D points
        R: (N, 3, 3)    - Rotation matrix (fixed)
        C: (N, 3)       - Camera center (fixed)
        K: (N, 3, 3)    - Camera intrinsic matrix (fixed)
        k: (N, 2,)      - Distortion coefficients (fixed)
        iterations: int - Number of optimization steps

    returns:
        t: (N, 3) - Optimized translation
    """
    # Convert 3D points to learnable parameters
    # pts_3d = torch.nn.Parameter(pts_3d.clone().detach().requires_grad_(True))
    t = torch.nn.Parameter(torch.zeros_like(pts_3d).clone().detach().requires_grad_(True))
    offset = torch.tensor([3, 3, 0.2], dtype=pts_3d.dtype, device=pts_3d.device)
    lower_bounds = t - offset
    upper_bounds = t + offset

    # check if there are any NaN values
    assert not torch.isnan(pts_3d).any()
    assert not torch.isnan(pts_2d).any()

    def closure():
        optimizer.zero_grad()
        projected_pts = project_points_th(pts_3d + t, R, C, K, k)
        loss = torch.nn.functional.mse_loss(projected_pts, pts_2d)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([t], line_search_fn="strong_wolfe")
    for _ in range(iterations):
        optimizer.step(closure)
        with torch.no_grad():
            t.copy_(torch.clamp(t, lower_bounds, upper_bounds))

    return t.detach()


def fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes):
    """wrapper function to fine-tune the translation of the 3D predictions to minimize reprojection error"""
    NUM_PERSONS = predictions.shape[0]
    mid_hip_3d = predictions[..., [7, 8], :].mean(axis=-2, keepdims=False)
    mid_hip_2d = skels_2d[..., [7, 8], :].mean(axis=-2, keepdims=False).transpose(1, 0, 2)

    R = np.array([k[0] for k in Rt])
    t = np.array([k[1] for k in Rt])
    C = (-t[:, None] @ R).squeeze(1)

    camera_params = {
        "K": cameras["K"][None].repeat(NUM_PERSONS, axis=0),
        "R": R[None].repeat(NUM_PERSONS, axis=0),
        "C": C[None].repeat(NUM_PERSONS, axis=0),
        "k": cameras["k"][None, ..., :2].repeat(NUM_PERSONS, axis=0),
    }
    valid = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)
    traj_3d = minimize_reprojection_error(
    pts_3d=torch.tensor(mid_hip_3d[valid], dtype=torch.float32).to(DEVICE),
    pts_2d=torch.tensor(mid_hip_2d[valid], dtype=torch.float32).to(DEVICE),
    R=torch.tensor(camera_params["R"][valid], dtype=torch.float32).to(DEVICE),
    C=torch.tensor(camera_params["C"][valid], dtype=torch.float32).to(DEVICE),
    K=torch.tensor(camera_params["K"][valid], dtype=torch.float32).to(DEVICE),
    k=torch.tensor(camera_params["k"][valid], dtype=torch.float32).to(DEVICE),
)
    return traj_3d, valid

# def fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes):
#     """More stable translation refinement with smoothing and stricter valid-frame masking."""
#     NUM_PERSONS = predictions.shape[0]
#     NUM_FRAMES = predictions.shape[1]

#     # body joints in current baseline ordering:
#     # 7 = left hip, 8 = right hip
#     left_hip_3d = predictions[..., 7, :]
#     right_hip_3d = predictions[..., 8, :]
#     mid_hip_3d = 0.5 * (left_hip_3d + right_hip_3d)

#     left_hip_2d = skels_2d[..., 7, :].transpose(1, 0, 2)   # (P, T, 2)
#     right_hip_2d = skels_2d[..., 8, :].transpose(1, 0, 2)  # (P, T, 2)
#     mid_hip_2d = 0.5 * (left_hip_2d + right_hip_2d)

#     # smooth per person through time before optimization
#     for person in range(NUM_PERSONS):
#         mid_hip_3d[person] = smooth_xyz_sequence(mid_hip_3d[person], k=5)
#         mid_hip_2d[person] = smooth_xy_sequence(mid_hip_2d[person], k=5)

#     R = np.array([k[0] for k in Rt])
#     t = np.array([k[1] for k in Rt])
#     C = (-t[:, None] @ R).squeeze(1)

#     camera_params = {
#         "K": cameras["K"][None].repeat(NUM_PERSONS, axis=0),
#         "R": R[None].repeat(NUM_PERSONS, axis=0),
#         "C": C[None].repeat(NUM_PERSONS, axis=0),
#         "k": cameras["k"][None, ..., :2].repeat(NUM_PERSONS, axis=0),
#     }

#     # old baseline only used boxes
#     valid_boxes = ~np.isnan(boxes).any(axis=-1).transpose(1, 0)            # (P, T)
#     valid_2d = ~np.isnan(mid_hip_2d).any(axis=-1)                          # (P, T)
#     valid_3d = ~np.isnan(mid_hip_3d).any(axis=-1)                          # (P, T)
#     valid = valid_boxes & valid_2d & valid_3d

#     traj_3d = torch.zeros((valid.sum(), 3), dtype=torch.float32, device=DEVICE)
#     if valid.sum() == 0:
#         return traj_3d, valid

#     traj_3d = minimize_reprojection_error(
#         pts_3d=torch.tensor(mid_hip_3d[valid], dtype=torch.float32).to(DEVICE),
#         pts_2d=torch.tensor(mid_hip_2d[valid], dtype=torch.float32).to(DEVICE),
#         R=torch.tensor(camera_params["R"][valid], dtype=torch.float32).to(DEVICE),
#         C=torch.tensor(camera_params["C"][valid], dtype=torch.float32).to(DEVICE),
#         K=torch.tensor(camera_params["K"][valid], dtype=torch.float32).to(DEVICE),
#         k=torch.tensor(camera_params["k"][valid], dtype=torch.float32).to(DEVICE),
#     )
#     return traj_3d, valid


def process_sequence(
    boxes: np.ndarray,
    cameras: dict,
    skels_3d: np.ndarray,
    skels_2d: np.ndarray,
    video_path: Path | str,
    tracker_options: CameraTrackerOptions,
) -> np.ndarray:
    """a naive baseline that uses the bounding boxes to estimate the camera pose
    1. estimate the camera pose using the bounding boxes
    2. periodically refine the camera pose using lane lines
    3. project the 3D skeletons to the 2D image plane and optimize the translation to minimize reprojection error
    """
    
    #make each path
    project_root = Path.cwd().resolve()

    crop_dir = (project_root / "outputs" / "player_crops" / video_path.stem).resolve()
    crop_dir.mkdir(parents=True, exist_ok=True)

    json_dir = (project_root / "outputs" / "openpose_json" / video_path.stem).resolve()
    json_dir.mkdir(parents=True, exist_ok=True)

    rerender_dir = (project_root / "outputs" / "rerender_dir" / video_path.stem).resolve()
    rerender_dir.mkdir(parents=True, exist_ok=True)

    #for ivslab path
    #openpose_root = Path(r"C:\Users\IVSLab\Documents\ypan179-sport research\my_own_repo\openpose")
    openpose_root = Path(r"C:\Users\Yifeng Pan\Downloads\openpose")

    NUM_FRAMES, NUM_PERSONS, _ = boxes.shape
    predictions = np.zeros((NUM_PERSONS, NUM_FRAMES, 15, 3))
    predictions.fill(np.nan)
    pitch_points = np.loadtxt("data/pitch_points.txt")

    video = cv2.VideoCapture(video_path)
    camera_tracker = CameraTracker(
        pitch_points=pitch_points,
        fps=50.0,
        options=tracker_options,
    )
    camera_tracker.initialize(
        frame_idx=0,
        K=cameras["K"][0],
        k=cameras["k"][0],
        R=cameras["R"][0],
        t=cameras["t"][0],
    )

    Rt = []
    for frame_idx in (pbar := tqdm(range(NUM_FRAMES), desc=f"{video_path.stem}")):
        success, img = video.read()
        if not success:
            print(f"Failed to read frame {frame_idx} from {video_path}")
            break

        state = camera_tracker.track(
            frame_idx=frame_idx,
            frame=img,
            K=cameras["K"][frame_idx],
            dist_coeffs=cameras["k"][frame_idx],
        )
        yaw, pitch, roll = state.get_ypr()
        pbar.set_postfix_str(f"yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
        Rt.append((state.R.copy(), state.t.copy()))
        H, W = img.shape[:2]

        for person in range(NUM_PERSONS):
            # decide which foot is in contact with the ground by checking which has lower y
            box = boxes[frame_idx, person]
            
            if np.isnan(box).any():
                continue
             # 2. 取出 box，并转成 int
            x1, y1, x2, y2 = box.astype(int)


            # 4. 检查 box 是否合法
            if x2 <= x1 or y2 <= y1:
                print(f"Invalid box at frame {frame_idx}, person {person}: {box}")
                continue

            # 5. crop 当前球员
            crop = img[y1:y2, x1:x2]

            # 6. 防止空 crop
            if crop.size == 0:
                print(f"Empty crop at frame {frame_idx}, person {person}: {box}")
                continue

            # 7. 保存 crop
            crop_name = f"f{frame_idx:05d}_p{person:03d}.jpg"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            #print out the example
            # print(f"Saved crop: {crop_path} | frame={frame_idx}, person={person}, box={box}")
            skel_2d = skels_2d[frame_idx, person]

            IDX = np.argmax(skel_2d[:, 1])
            x, y = skel_2d[IDX]
            K = cameras["K"][frame_idx]
            k = cameras["k"][frame_idx]
            R, t = Rt[-1]
            o, d = ray_from_xy((x, y), K, R, t, k[0], k[1])
            intersection = intersection_over_plane(o, d)

            # convert from camera space to world space
            skel_3d = skels_3d[frame_idx, person]
            skel_3d = skel_3d @ R
            skel_3d = skel_3d - skel_3d[IDX] + intersection
            predictions[person, frame_idx] = skel_3d

    run_openpose_on_crops(
        openpose_root=openpose_root,
        crop_dir=crop_dir,
        json_dir=json_dir,
        render_pose=1,
        write_images_dir=rerender_dir,
    )
    # fine-tune the translation to minimize reprojection error
        # MODIFIED BASELINE POST-PROCESSING

    # Step 1:
    # Keep the baseline translation refinement,
    # but now it uses our improved fine_tune_translation().
    traj_3d, valid = fine_tune_translation(predictions, skels_2d, cameras, Rt, boxes)
    predictions[valid] = predictions[valid] + traj_3d.cpu().numpy()[:, None, :]

    # Step 2:
    # Keep the original baseline smoother.
    # This is NOT new; it already existed in the baseline.
    for person in range(NUM_PERSONS):
        # comment out the baseline panda smoothing.
        predictions[person] = smoothen(predictions[person])

        # # NEW PART ADDED BY US:
        # # Smooth each joint trajectory across time with our robust median smoother.
        # # This reduces per-joint jitter after the baseline smoothing step.
        # for joint in range(predictions.shape[2]):
        #     predictions[person, :, joint, :] = smooth_xyz_sequence(
        #         predictions[person, :, joint, :], k=5
        #     )

        # # NEW PART ADDED BY US:
        # # Compute pelvis center from left and right hips.
        # # We use the pelvis because it is a stable body center.
        # pelvis = 0.5 * (
        #     predictions[person, :, 7, :] + predictions[person, :, 8, :]
        # )

        # # NEW PART ADDED BY US:
        # # Remove sudden unrealistic jumps in the pelvis trajectory.
        # pelvis = remove_3d_spikes(pelvis, factor=3.0)

        # # NEW PART ADDED BY US:
        # # Compute how much the corrected pelvis differs from the current pelvis.
        # current_pelvis = 0.5 * (
        #     predictions[person, :, 7, :] + predictions[person, :, 8, :]
        # )
        # delta = pelvis - current_pelvis

        # # NEW PART ADDED BY US:
        # # Apply the same pelvis correction to the whole skeleton.
        # # This shifts the whole body consistently instead of correcting joints independently.
        # predictions[person] = predictions[person] + delta[:, None, :]

    # update the camera parameters
    cameras["R"] = np.array([k[0] for k in Rt], dtype=np.float32)
    cameras["t"] = np.array([k[1] for k in Rt], dtype=np.float32)
    return predictions.astype(np.float32)


def load_sequences(sequences_file: Path | str) -> list[str]:
    with open(sequences_file) as f:
        sequences = f.read().splitlines()
    sequences = filter(lambda x: not x.startswith("#"), sequences)
    sequences = [s.strip() for s in sequences]
    return sequences

#function add by me:
def run_openpose_on_crops(
    openpose_root: str | Path,
    crop_dir: str | Path,
    json_dir: str | Path,
    render_pose: int = 0,
    write_images_dir: str | Path | None = None,
):
    """
    Run OpenPose on a folder of cropped player images.

    Args:
        openpose_root: root folder of OpenPose, e.g. C:/.../openpose
        crop_dir: folder containing cropped player images
        json_dir: output folder for OpenPose JSON results
        render_pose: 0 = no rendered images, 1 = save rendered images if write_images_dir is given
        write_images_dir: optional folder to save rendered images

    Returns:
        None
    """
    openpose_root = Path(openpose_root)
    crop_dir = Path(crop_dir)
    json_dir = Path(json_dir)

    json_dir.mkdir(parents=True, exist_ok=True)

    exe_path = openpose_root / "bin" / "OpenPoseDemo.exe"

    cmd = [
        str(exe_path),
        "--image_dir", str(crop_dir),
        "--display", "0",
        "--render_pose", str(render_pose),
        "--write_json", str(json_dir),
    ]

    if write_images_dir is not None:
        write_images_dir = Path(write_images_dir)
        write_images_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--write_images", str(write_images_dir)])

    print("Running OpenPose command:")
    print(" ".join(cmd))

    # Very important: cwd must be OpenPose root so that models/ can be found
    subprocess.run(
        cmd,
        cwd=str(openpose_root),
        check=True
    )


def main(
    sequences: list[str],
    output: Path | str,
    max_refine_interval: int,
    export_camera: bool,
    visualize: bool,
):
    debug_stages = ["projection", "flow", "mask"] if visualize else []
    if export_camera:
        camera_dir = Path("outputs/calibration/")
        camera_dir.mkdir(parents=True, exist_ok=True)
    else:
        camera_dir = None

    root = Path("data/")
    solutions = {}

    for sequence in sequences:
        camera = dict(np.load(root / "cameras" / f"{sequence}.npz"))
        skel2d = np.load(root / "skel_2d" / f"{sequence}.npy")
        skel3d = np.load(root / "skel_3d" / f"{sequence}.npy")
        boxes = np.load(root / "boxes" / f"{sequence}.npy")
        video_path = root / "videos" / f"{sequence}.mp4"

        NUM_FRAMES = boxes.shape[0]
        solutions[sequence] = process_sequence(
            cameras=camera,
            boxes=boxes,
            skels_2d=skel2d[:, :, OPENPOSE_TO_OURS],
            skels_3d=skel3d[:, :, OPENPOSE_TO_OURS],
            video_path=video_path,
            tracker_options=CameraTrackerOptions(
                refine_interval=np.clip(NUM_FRAMES // 500, a_min=1, a_max=max_refine_interval),
                debug_stages=tuple(debug_stages),
            ),
        )

        if export_camera:
            camera_path = camera_dir / f"{sequence}.npz"
            np.savez(camera_path, **camera)
    
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **solutions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences", "-s", type=str, default="data/sequences_full.txt", help="Path to the sequences file"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default="output/submission_full.npz", help="Path to the output npz file"
    )
    parser.add_argument("--refine_interval", "-r", type=int, default=1, help="Interval to refine the camera pose")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the tracking results")
    parser.add_argument("--export_camera", "-c", action="store_true", help="Export the camera parameters")
    args = parser.parse_args()

    sequences = load_sequences(args.sequences)
    main(sequences, args.output, args.refine_interval, args.export_camera, args.visualize)
