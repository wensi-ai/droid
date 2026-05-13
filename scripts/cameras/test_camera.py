#!/usr/bin/env python3
"""
Test script to stream all ZED camera feeds using the droid ZedCamera wrapper.

This script:
1. Detects all connected ZED cameras
2. Opens and configures each camera via the droid ZedCamera class
3. Streams live feeds from all cameras in a grid layout
4. Displays camera information (serial number, FPS, frame count)

Usage:
    python scripts/cameras/test_camera.py [--exposure EXPOSURE] [--pc]

Controls:
    'q' or ESC - Quit
    Ctrl+C - Quit Viser point-cloud mode
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from droid.camera_utils.camera_readers.zed_camera import gather_zed_cameras


DEFAULT_PC_RESOLUTION = (640, 360)


def zed_pointcloud_to_viser_arrays(
    pointcloud,
    decimation=2,
    max_points=100_000,
    min_distance=0.1,
    max_distance=5.0,
):
    """Convert a ZED XYZRGBA image into filtered Viser point/color arrays."""
    if pointcloud is None or pointcloud.ndim != 3 or pointcloud.shape[-1] < 4:
        return None, None

    decimation = max(1, decimation)
    pointcloud = pointcloud[::decimation, ::decimation]

    points = pointcloud[..., :3].reshape(-1, 3).astype(np.float32, copy=False)
    rgba_size = (*pointcloud.shape[:2], 4)
    rgba_float = np.ascontiguousarray(pointcloud[..., 3])
    colors = rgba_float.view(np.uint8).reshape(rgba_size)[..., :3].reshape(-1, 3)

    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]
    if len(finite_points) > 0:
        finite_distances = np.linalg.norm(finite_points, axis=1)
        median_distance = np.nanmedian(finite_distances[np.isfinite(finite_distances)])
        if np.isfinite(median_distance) and median_distance > max_distance * 10:
            points = points / 1000.0

    distances = np.linalg.norm(points, axis=1)
    valid_mask = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(distances)
        & (distances >= min_distance)
        & (distances <= max_distance)
    )
    points = points[valid_mask]
    colors = colors[valid_mask]

    if len(points) > max_points:
        stride = int(np.ceil(len(points) / max_points))
        points = points[::stride]
        colors = colors[::stride]

    return points, colors


class ZEDCameraStreamer:
    """Manages multiple ZED cameras for live streaming via the droid ZedCamera wrapper."""

    def __init__(
        self,
        exposure=None,
        pointcloud=False,
        viser_port=8080,
        pc_resolution=DEFAULT_PC_RESOLUTION,
        pc_decimation=2,
        pc_max_points=100_000,
        pc_min_distance=0.1,
        pc_max_distance=5.0,
        pc_point_size=0.005,
    ):
        self.zed_cameras = []
        self.exposure = exposure
        self.pointcloud = pointcloud
        self.viser_port = viser_port
        self.pc_resolution = pc_resolution
        self.pc_decimation = pc_decimation
        self.pc_max_points = pc_max_points
        self.pc_min_distance = pc_min_distance
        self.pc_max_distance = pc_max_distance
        self.pc_point_size = pc_point_size
        self.frame_counts = {}
        self.start_time = time.time()

    def detect_and_initialize(self):
        """Detect and initialize ZED cameras using the droid ZedCamera wrapper."""
        print("\n" + "=" * 60)
        print("Detecting ZED cameras...")
        print("=" * 60)

        detected_cameras = gather_zed_cameras()
        if not detected_cameras:
            print("ERROR: No ZED cameras detected!")
            return False

        print(f"\nFound {len(detected_cameras)} ZED camera(s):")
        for cam in detected_cameras:
            print(f"  Serial: {cam.serial_number}, Hand camera: {cam.is_hand_camera}")

        if self.pointcloud:
            external_cameras = [cam for cam in detected_cameras if not cam.is_hand_camera]
            self.zed_cameras = external_cameras or detected_cameras
            selected_serials = ", ".join(cam.serial_number for cam in self.zed_cameras)
            print(f"\nPoint-cloud mode enabled; streaming external camera(s): {selected_serials}")
        else:
            self.zed_cameras = detected_cameras

        # Configure reading parameters and open in trajectory mode
        for cam in self.zed_cameras:
            cam.set_reading_parameters(
                image=not self.pointcloud,
                pointcloud=self.pointcloud,
                concatenate_images=False,
                resolution=self.pc_resolution if self.pointcloud else (0, 0),
            )
            cam.set_trajectory_mode()
            self.frame_counts[cam.serial_number] = 0

        # Apply exposure settings
        for cam in self.zed_cameras:
            if self.exposure is None:
                cam.set_exposure(auto=True)
                print(f"  Camera {cam.serial_number}: auto exposure")
            else:
                cam.set_exposure(exposure_value=self.exposure, auto=False)
                print(f"  Camera {cam.serial_number}: manual exposure={self.exposure}")

        print(f"\nSuccessfully initialized {len(self.zed_cameras)} camera(s)")
        return True

    def create_grid_layout(self, frames):
        """Arrange multiple camera frames in a grid layout."""
        num_cameras = len(frames)
        if num_cameras == 0:
            return None

        if num_cameras == 1:
            return frames[0]

        # Calculate grid dimensions (prefer wider than tall)
        cols = int(np.ceil(np.sqrt(num_cameras * 1.5)))
        rows = int(np.ceil(num_cameras / cols))

        # Get frame dimensions
        h, w = frames[0].shape[:2]

        # Create blank canvas
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # Place frames in grid
        for idx, frame in enumerate(frames):
            row = idx // cols
            col = idx % cols
            grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = frame

        return grid

    def add_camera_info(self, frame, serial, frame_count):
        """Add camera information overlay to frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        info_height = 125 if self.exposure is not None else 100
        cv2.rectangle(overlay, (10, 10), (w - 10, info_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        elapsed_time = time.time() - self.start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Serial: {serial}", (20, 35), font, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 60), font, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 85), font, 0.6, (0, 255, 0), 2)
        if self.exposure is not None:
            cv2.putText(frame, f"Exposure: {self.exposure}", (20, 110), font, 0.6, (0, 255, 255), 2)

        return frame

    def save_frames(self, frames):
        """Save current frames to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("camera_frames") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        for frame, cam in zip(frames, self.zed_cameras):
            filename = output_dir / f"camera_{cam.serial_number}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"  Saved: {filename}")

        print(f"\nSaved {len(frames)} frame(s) to {output_dir}")

    def stream(self):
        """Main streaming loop."""
        if not self.zed_cameras:
            print("ERROR: No cameras available for streaming!")
            return

        if self.pointcloud:
            self.stream_pointclouds()
            return

        print("\n" + "=" * 60)
        print("Starting camera stream...")
        print("=" * 60)
        print("\nControls:")
        print("  'q' or ESC - Quit\n")

        window_name = f"ZED Camera Stream ({len(self.zed_cameras)} cameras)"
        window_created = False
        self.start_time = time.time()
        last_display_frame = None

        try:
            while True:
                frames = []

                for cam in self.zed_cameras:
                    serial = cam.serial_number
                    read_result = cam.read_camera()
                    data_dict = None if read_result is None else read_result[0]

                    if data_dict is None or "image" not in data_dict:
                        print(f"Warning: Failed to grab frame from camera {serial}")
                        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(blank, f"Camera {serial} - Error", (400, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        frames.append(blank)
                        continue

                    # ZedCamera returns BGRA; drop alpha to get BGR for cv2
                    left_key = serial + "_left"
                    frame = data_dict["image"].get(left_key)
                    if frame is None:
                        frame = next(iter(data_dict["image"].values()))
                    frame = frame[..., :3]

                    self.frame_counts[serial] += 1
                    frame = self.add_camera_info(frame, serial, self.frame_counts[serial])
                    frames.append(frame)

                if not frames:
                    print("No frames captured!")
                    break

                display_frame = self.create_grid_layout(frames)
                last_display_frame = display_frame

                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    h, w = display_frame.shape[:2]
                    cv2.resizeWindow(window_name, w, h)
                    window_created = True

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\nStopping stream...")
                    if last_display_frame is not None:
                        self.save_grid_image(last_display_frame)
                    break

        except KeyboardInterrupt:
            print("\n\nStream interrupted by user")

        finally:
            self.cleanup()

    def stream_pointclouds(self):
        """Stream external camera point clouds to a Viser browser view."""
        try:
            import viser
        except ModuleNotFoundError as exc:
            raise RuntimeError("Point-cloud mode requires viser. Install it with `pip install viser`.") from exc

        print("\n" + "=" * 60)
        print("Starting Viser point-cloud stream...")
        print("=" * 60)
        print(f"\nOpen http://localhost:{self.viser_port} in a browser")
        print("Controls:")
        print("  Ctrl+C - Quit\n")

        server = viser.ViserServer(host="0.0.0.0", port=self.viser_port)
        server.scene.world_axes.visible = True
        self.start_time = time.time()
        pointcloud_handles = {}

        try:
            while True:
                for cam in self.zed_cameras:
                    serial = cam.serial_number
                    read_result = cam.read_camera()
                    if read_result is None:
                        print(f"Warning: Failed to grab point cloud from camera {serial}")
                        continue

                    data_dict, _ = read_result
                    pointclouds = data_dict.get("pointcloud") if data_dict is not None else None
                    if not pointclouds:
                        print(f"Warning: No point cloud returned from camera {serial}")
                        continue

                    left_key = serial + "_left"
                    pointcloud = pointclouds.get(left_key)
                    if pointcloud is None:
                        pointcloud = next(iter(pointclouds.values()))

                    points, colors = zed_pointcloud_to_viser_arrays(
                        pointcloud,
                        decimation=self.pc_decimation,
                        max_points=self.pc_max_points,
                        min_distance=self.pc_min_distance,
                        max_distance=self.pc_max_distance,
                    )
                    if points is None or len(points) == 0:
                        print(f"Warning: No valid point-cloud points from camera {serial}")
                        continue

                    self.frame_counts[serial] += 1
                    name = f"/zed_{serial}/pointcloud"
                    if name not in pointcloud_handles:
                        server.scene.add_frame(f"/zed_{serial}", show_axes=True, axes_length=0.1)
                        pointcloud_handles[name] = server.scene.add_point_cloud(
                            name,
                            points=points,
                            colors=colors,
                            point_size=self.pc_point_size,
                            point_shape="rounded",
                            precision="float32",
                        )
                    else:
                        pointcloud_handles[name].points = points
                        pointcloud_handles[name].colors = colors

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nPoint-cloud stream interrupted by user")

        finally:
            server.stop()
            self.cleanup()

    def save_grid_image(self, display_frame):
        """Save the current grid layout as a PNG file."""
        output_path = Path("robot_camera_views.png")
        try:
            cv2.imwrite(str(output_path), display_frame)
            print(f"Saved camera grid view to: {output_path}")
        except Exception as e:
            print(f"Failed to save image: {e}")

    def cleanup(self):
        """Clean up and close all cameras."""
        print("\nCleaning up...")
        cv2.destroyAllWindows()

        for cam in self.zed_cameras:
            cam.disable_camera()
            print(f"  Closed camera {cam.serial_number}")

        print("\nDone!")

    def run(self):
        """Main entry point."""
        if not self.detect_and_initialize():
            return False

        self.stream()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Stream all ZED camera feeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream with auto exposure (default)
  python scripts/cameras/test_camera.py

  # Stream with manual exposure at 50
  python scripts/cameras/test_camera.py --exposure 50

  # Darken the image
  python scripts/cameras/test_camera.py --exposure 20

  # Stream the external ZED point cloud in Viser
  python scripts/cameras/test_camera.py --pc
        """,
    )

    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="Manual exposure value (0-100). Omit for auto exposure.",
    )
    parser.add_argument(
        "--pc",
        action="store_true",
        help="Stream external camera point clouds to Viser instead of showing OpenCV images.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=8080,
        help="Port for the Viser point-cloud server.",
    )
    parser.add_argument(
        "--pc-resolution",
        type=int,
        nargs=2,
        default=DEFAULT_PC_RESOLUTION,
        metavar=("WIDTH", "HEIGHT"),
        help="ZED point-cloud resolution used in --pc mode.",
    )
    parser.add_argument(
        "--pc-decimation",
        type=int,
        default=2,
        help="Spatial decimation stride for Viser point-cloud updates.",
    )
    parser.add_argument(
        "--pc-max-points",
        type=int,
        default=100_000,
        help="Maximum number of points sent to Viser per camera.",
    )
    parser.add_argument(
        "--pc-min-distance",
        type=float,
        default=0.1,
        help="Minimum point distance in meters for Viser point-cloud filtering.",
    )
    parser.add_argument(
        "--pc-max-distance",
        type=float,
        default=5.0,
        help="Maximum point distance in meters for Viser point-cloud filtering.",
    )
    parser.add_argument(
        "--pc-point-size",
        type=float,
        default=0.005,
        help="Point size used by Viser in --pc mode.",
    )

    args = parser.parse_args()

    streamer = ZEDCameraStreamer(
        exposure=args.exposure,
        pointcloud=args.pc,
        viser_port=args.viser_port,
        pc_resolution=tuple(args.pc_resolution),
        pc_decimation=args.pc_decimation,
        pc_max_points=args.pc_max_points,
        pc_min_distance=args.pc_min_distance,
        pc_max_distance=args.pc_max_distance,
        pc_point_size=args.pc_point_size,
    )
    success = streamer.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
