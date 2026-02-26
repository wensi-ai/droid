#!/usr/bin/env python3
"""
Test script to stream all ZED camera feeds using pyzed.

This script:
1. Detects all connected ZED cameras
2. Opens and configures each camera
3. Streams live feeds from all cameras in a grid layout
4. Displays camera information (serial number, FPS, frame count)

Usage:
    python scripts/test_camera.py [--resolution {720p,1080p,2k}] [--fps FPS]

Controls:
    'q' or ESC - Quit
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import pyzed.sl as sl


class ZEDCameraStreamer:
    """Manages multiple ZED cameras for live streaming."""

    def __init__(self, resolution="720p", fps=30):
        self.cameras = []
        self.serial_numbers = []
        self.resolution = self._get_resolution(resolution)
        self.fps = fps
        self.frame_counts = {}
        self.start_time = time.time()

    def _get_resolution(self, resolution):
        """Convert resolution string to ZED SDK resolution."""
        resolution_map = {
            "720p": sl.RESOLUTION.HD720,  # 1280x720
            "1080p": sl.RESOLUTION.HD1080,  # 1920x1080
            "2k": sl.RESOLUTION.HD2K,  # 2208x1242
        }
        return resolution_map.get(resolution, sl.RESOLUTION.HD720)

    def detect_cameras(self):
        """Detect all connected ZED cameras."""
        print("\n" + "=" * 60)
        print("Detecting ZED cameras...")
        print("=" * 60)

        devices = sl.Camera.get_device_list()
        if not devices:
            print("ERROR: No ZED cameras detected!")
            return False

        print(f"\nFound {len(devices)} ZED camera(s):")
        for i, device in enumerate(devices):
            serial = device.serial_number
            camera_model = device.camera_model
            print(f"  [{i + 1}] Serial: {serial}, Model: {camera_model}")
            self.serial_numbers.append(serial)

        return True

    def initialize_cameras(self):
        """Initialize and open all detected cameras."""
        print("\n" + "=" * 60)
        print("Initializing cameras...")
        print("=" * 60)

        for serial in self.serial_numbers:
            try:
                # Create camera instance
                camera = sl.Camera()

                # Configure initialization parameters
                init_params = sl.InitParameters()
                init_params.camera_resolution = self.resolution
                init_params.camera_fps = self.fps
                init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth for faster streaming
                init_params.coordinate_units = sl.UNIT.METER
                init_params.set_from_serial_number(serial)

                # Open camera
                status = camera.open(init_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    print(f"  [ERROR] Failed to open camera {serial}: {status}")
                    continue

                # Get camera information
                cam_info = camera.get_camera_information()
                resolution = cam_info.camera_configuration.resolution
                fps = cam_info.camera_configuration.fps

                print(f"  [OK] Camera {serial}:")
                print(f"       Resolution: {resolution.width}x{resolution.height}")
                print(f"       FPS: {fps}")

                self.cameras.append(camera)
                self.frame_counts[serial] = 0

            except Exception as e:
                print(f"  [ERROR] Exception opening camera {serial}: {e}")

        if not self.cameras:
            print("\nERROR: No cameras were successfully initialized!")
            return False

        print(f"\nSuccessfully initialized {len(self.cameras)} camera(s)")
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
        # Create semi-transparent overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw info background
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Serial: {serial}", (20, 35), font, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 60), font, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 85), font, 0.6, (0, 255, 0), 2)

        return frame

    def save_frames(self, frames):
        """Save current frames to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("camera_frames") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (frame, serial) in enumerate(zip(frames, self.serial_numbers)):
            filename = output_dir / f"camera_{serial}_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"  Saved: {filename}")

        print(f"\nSaved {len(frames)} frame(s) to {output_dir}")

    def stream(self):
        """Main streaming loop."""
        if not self.cameras:
            print("ERROR: No cameras available for streaming!")
            return

        print("\n" + "=" * 60)
        print("Starting camera stream...")
        print("=" * 60)
        print("\nControls:")
        print("  'q' or ESC - Quit\n")

        # Create runtime parameters
        runtime_params = sl.RuntimeParameters()
        mats = [sl.Mat() for _ in self.cameras]

        # Window name
        window_name = f"ZED Camera Stream ({len(self.cameras)} cameras)"
        window_created = False

        # Reset start time
        self.start_time = time.time()

        try:
            while True:
                frames = []

                # Grab frames from all cameras
                for idx, (camera, serial) in enumerate(zip(self.cameras, self.serial_numbers)):
                    # Grab new frame
                    err = camera.grab(runtime_params)

                    if err == sl.ERROR_CODE.SUCCESS:
                        # Retrieve left camera image
                        camera.retrieve_image(mats[idx], sl.VIEW.LEFT)
                        # ZED SDK returns BGRA, just remove alpha channel (keep BGR for cv2)
                        frame = mats[idx].get_data()[:, :, :3]

                        # Update frame count
                        self.frame_counts[serial] += 1

                        # Add camera info overlay
                        frame = self.add_camera_info(frame, serial, self.frame_counts[serial])

                        frames.append(frame)
                    else:
                        print(f"Warning: Failed to grab frame from camera {serial}: {err}")
                        # Create blank frame as placeholder
                        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(
                            blank, f"Camera {serial} - Error", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                        )
                        frames.append(blank)

                if not frames:
                    print("No frames captured!")
                    break

                # Create grid layout
                display_frame = self.create_grid_layout(frames)

                # Create window after first frame (so it auto-sizes correctly)
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    # Resize window to match actual frame size
                    h, w = display_frame.shape[:2]
                    cv2.resizeWindow(window_name, w, h)
                    window_created = True

                # Display
                cv2.imshow(window_name, display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    print("\nStopping stream...")
                    break

        except KeyboardInterrupt:
            print("\n\nStream interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up and close all cameras."""
        print("\nCleaning up...")
        cv2.destroyAllWindows()

        for idx, camera in enumerate(self.cameras):
            serial = self.serial_numbers[idx] if idx < len(self.serial_numbers) else "unknown"
            camera.close()
            print(f"  Closed camera {serial}")

        print("\nDone!")

    def run(self):
        """Main entry point."""
        if not self.detect_cameras():
            return False

        if not self.initialize_cameras():
            return False

        self.stream()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Stream all ZED camera feeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream at 720p @ 30fps (default)
  python scripts/test_camera.py

  # Stream at 1080p @ 60fps
  python scripts/test_camera.py --resolution 1080p --fps 60

  # Stream at 2K @ 15fps
  python scripts/test_camera.py --resolution 2k --fps 15
        """,
    )

    parser.add_argument(
        "--resolution",
        type=str,
        choices=["720p", "1080p", "2k"],
        default="720p",
        help="Camera resolution (default: 720p)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )

    args = parser.parse_args()

    # Create and run streamer
    streamer = ZEDCameraStreamer(resolution=args.resolution, fps=args.fps)
    success = streamer.run()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
