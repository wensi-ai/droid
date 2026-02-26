#!/usr/bin/env python3
"""
Simple calibration script for third-person ZED cameras using Oculus VR control
Press Enter to start calibration, Enter again to stop
"""

import os
import sys
import time
import json
import threading
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from termcolor import colored
from droid.misc.parameters import hand_camera_id
from droid.robot_env import RobotEnv
from droid.controllers.oculus_controller import VRPolicy
# Import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print(colored("Error: pyzed not available. Please install ZED SDK.", "red"))
    sys.exit(1)

# ChArUco board parameters (from DROID)
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.018  # 18mm
CHARUCOBOARD_MARKER_SIZE = 0.013  # 13mm
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

# Create ChArUco board
CHARUCO_BOARD = aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=CHARUCOBOARD_CHECKER_SIZE,
    markerLength=CHARUCOBOARD_MARKER_SIZE,
    dictionary=ARUCO_DICT,
)

# Global flag for calibration control
calibration_active = False
calibration_lock = threading.Lock()


def detect_charuco_corners(image, camera_matrix, dist_coeffs):
    """
    Detect ChArUco corners - matching DROID's approach exactly
    Returns: (corners, ids, image_with_markers)
    """
    # Make a copy for visualization
    image_copy = image.copy()

    # Convert to grayscale (DROID uses grayscale)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create detector parameters - use CORNER_REFINE_SUBPIX like DROID
    detector_params = cv2.aruco.DetectorParameters_create()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Detect ArUco markers
    # import pdb; pdb.set_trace()
    corners, ids, rejected = aruco.detectMarkers(
        image=gray,
        dictionary=ARUCO_DICT,
        parameters=detector_params
    )

    # If no markers found, return early
    if ids is None or len(ids) == 0:
        # Draw rejected candidates for debugging
        if len(rejected) > 0:
            for i, rej in enumerate(rejected[:5]):  # Show first 5 rejected
                pts = rej.reshape(-1, 1, 2).astype(int)
                cv2.polylines(image_copy, [pts], True, (0, 0, 255), 2)
        return None, None, image_copy

    # Draw detected markers
    image_with_markers = aruco.drawDetectedMarkers(image_copy, corners, ids)

    # Refine detected markers (CRITICAL - like DROID does)
    corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
        gray,
        CHARUCO_BOARD,
        corners,
        ids,
        rejected,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        parameters=detector_params
    )

    if ids is None or len(ids) == 0:
        return None, None, image_with_markers

    # Interpolate ChArUco corners
    charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=CHARUCO_BOARD,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )

    # Check if interpolation was successful
    if not charuco_retval or charuco_corners is None:
        return None, None, image_with_markers

    # Draw ChArUco corners
    image_with_markers = aruco.drawDetectedCornersCharuco(
        image_with_markers, charuco_corners, charuco_ids
    )

    # Need at least 4 corners for PnP
    if len(charuco_corners) < 4:
        return None, None, image_with_markers

    return charuco_corners, charuco_ids, image_with_markers


def calibrate_hand_eye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
    """
    Solve hand-eye calibration using Tsai-Lenz method
    This is simplified version of cv2.calibrateHandEye

    For third-person camera:
    - Robot holds board (board attached to gripper)
    - Camera observes from fixed position
    - Solve: AX = XB where X is camera-to-base transform
    """
    n = len(R_gripper2base)
    if n < 3:
        return None, None

    # Use OpenCV's implementation for now (we can replace with pure math later)
    # This solves AX = XB problem
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI  # Tsai-Lenz method
    )

    return R_cam2base, t_cam2base


class SimpleZEDCalibrator:
    """Simple calibration system using Oculus VR control"""

    def __init__(self):
        # Components
        self.env = None   # RobotEnv
        self.controller = None  # VRPolicy
        self.zed1 = None
        self.zed2 = None
        self.zed3 = None  # Third camera (eye-in-hand)

        # Camera info
        self.cam1_serial = None
        self.cam1_intrinsics = None
        self.cam2_serial = None
        self.cam2_intrinsics = None
        self.cam3_serial = None
        self.cam3_intrinsics = None

        # Image containers (set to None until camera opens successfully)
        self.zed_image1 = None
        self.zed_image2 = None
        self.zed_image3 = None

        # List of successfully connected cameras:
        # each entry is (internal_cam_id, cam_key, serial, intrinsics, is_eye_in_hand)
        self.connected_cameras = []

        # Calibration data storage
        self.calibration_data = {
            'cam1': {'images': [], 'robot_poses': []},
            'cam2': {'images': [], 'robot_poses': []},
            'cam3': {'images': [], 'robot_poses': []}  # Eye-in-hand camera
        }

    def init_cameras(self):
        """Initialize ZED cameras - skips cameras that are not connected"""
        print(colored("Initializing ZED cameras...", "yellow"))

        self.connected_cameras = []

        # (camera_index, zed_attr, serial_attr, intrinsics_attr, image_attr, cam_key)
        camera_configs = [
            (0, 'zed1', 'cam1_serial', 'cam1_intrinsics', 'zed_image1', 'cam1'),
            (1, 'zed2', 'cam2_serial', 'cam2_intrinsics', 'zed_image2', 'cam2'),
            (2, 'zed3', 'cam3_serial', 'cam3_intrinsics', 'zed_image3', 'cam3'),
        ]

        for cam_idx, zed_attr, serial_attr, intrinsics_attr, image_attr, cam_key in camera_configs:
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.set_from_camera_id(cam_idx)
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30
            init_params.depth_mode = sl.DEPTH_MODE.NONE

            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(colored(f"Camera {cam_idx + 1} not available: {err} (skipping)", "yellow"))
                setattr(self, zed_attr, None)
                setattr(self, image_attr, None)
                continue

            cam_info = zed.get_camera_information()
            serial = str(cam_info.serial_number)
            calib = cam_info.camera_configuration.calibration_parameters.left_cam
            intrinsics = {
                'matrix': np.array([[calib.fx, 0, calib.cx],
                                   [0, calib.fy, calib.cy],
                                   [0, 0, 1]]),
                'distortion': np.array(calib.disto)
            }

            setattr(self, zed_attr, zed)
            setattr(self, serial_attr, serial)
            setattr(self, intrinsics_attr, intrinsics)
            setattr(self, image_attr, sl.Mat())

            internal_cam_id = cam_idx + 1
            is_eye_in_hand = (serial == hand_camera_id)
            self.connected_cameras.append((internal_cam_id, cam_key, serial, intrinsics, is_eye_in_hand))
            cam_type = "Eye-in-hand" if is_eye_in_hand else "Third-person"
            print(colored(f"✓ Camera {internal_cam_id} initialized (Serial: {serial}, {cam_type})", "green"))

        if not self.connected_cameras:
            print(colored("No cameras could be initialized!", "red"))
            return False

        self.zed_runtime = sl.RuntimeParameters()
        print(colored(f"✓ {len(self.connected_cameras)} camera(s) ready", "green"))
        return True

    def init_robot_and_oculus(self):
        """Initialize robot connection and Oculus VR controller using the DROID stack"""
        print(colored("Connecting to robot...", "yellow"))

        try:
            # camera_kwargs={} prevents MultiCameraWrapper from opening any ZED cameras,
            # avoiding hardware conflicts with the ZED cameras we open ourselves above.
            # do_reset=False so we control when the robot moves.
            self.env = RobotEnv(action_space="cartesian_velocity", camera_kwargs={}, do_reset=False)
            print(colored("✓ Robot connected via DROID stack", "green"))
        except Exception as e:
            print(colored(f"Failed to connect to robot: {e}", "red"))
            return False

        try:
            self.controller = VRPolicy()
            print(colored("✓ Oculus VR controller ready", "green"))
        except Exception as e:
            print(colored(f"Failed to initialize Oculus controller: {e}", "red"))
            return False

        # Move robot to home position
        print(colored("Moving robot to home position...", "yellow"))
        self.env.reset()
        print(colored("✓ Robot at home position", "green"))

        print(colored("\nOculus control ready. Hold the RIGHT GRIP button to move the robot.", "cyan"))
        return True

    def get_robot_pose(self):
        """Get current robot end-effector pose as [x,y,z,rx,ry,rz]"""
        state_dict, _ = self.env.get_state()
        # cartesian_position is already [x, y, z, roll, pitch, yaw] in euler angles
        return np.array(state_dict['cartesian_position'])

    def capture_images(self):
        """Capture images from all three cameras"""
        images = {}

        # Camera 1
        if self.zed1 and self.zed1.grab(self.zed_runtime) == sl.ERROR_CODE.SUCCESS:
            self.zed1.retrieve_image(self.zed_image1, sl.VIEW.LEFT)
            # ZED returns BGRA, convert to BGR for OpenCV
            bgra = self.zed_image1.get_data()
            images['cam1'] = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        # Camera 2
        if self.zed2 and self.zed2.grab(self.zed_runtime) == sl.ERROR_CODE.SUCCESS:
            self.zed2.retrieve_image(self.zed_image2, sl.VIEW.LEFT)
            # ZED returns BGRA, convert to BGR for OpenCV
            bgra = self.zed_image2.get_data()
            images['cam2'] = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        # Camera 3 (Eye-in-hand)
        if self.zed3 and self.zed3.grab(self.zed_runtime) == sl.ERROR_CODE.SUCCESS:
            self.zed3.retrieve_image(self.zed_image3, sl.VIEW.LEFT)
            # ZED returns BGRA, convert to BGR for OpenCV
            bgra = self.zed_image3.get_data()
            images['cam3'] = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        return images

    def run_calibration_loop(self, camera_id):
        """
        Main calibration loop - user controls with Oculus VR
        Robot control starts immediately (hold right grip button to move)
        First Enter: Start calibration data collection
        Second Enter: Stop and save calibration
        """
        global calibration_active

        # Determine if eye-in-hand based on serial number
        cam_serial = None
        if camera_id == 1:
            cam_serial = self.cam1_serial
        elif camera_id == 2:
            cam_serial = self.cam2_serial
        elif camera_id == 3:
            cam_serial = self.cam3_serial

        is_eye_in_hand = (cam_serial == hand_camera_id)

        print(colored(f"\n=== Calibrating Camera {camera_id} {'(Eye-in-hand)' if is_eye_in_hand else '(Third-person)'} ===", "cyan", attrs=['bold']))
        print(colored("Instructions:", "yellow"))

        if is_eye_in_hand:
            print("1. Place ChArUco board in a FIXED position (e.g., on table)")
            print("2. Hold RIGHT GRIP on Oculus to move robot so camera can see the board")
            print("3. Press ENTER to start collecting calibration data")
            print("4. Move robot to observe board from different angles")
            print("5. Press ENTER again to stop and compute calibration")
        else:
            print("1. Attach ChArUco board to robot gripper")
            print("2. Hold RIGHT GRIP on Oculus to position board about 1 foot from camera")
            print("3. Press ENTER to start collecting calibration data")
            print("4. Move robot slowly to show board from different angles")
            print("5. Press ENTER again to stop and compute calibration")
        print("")

        # Select camera parameters
        if camera_id == 1:
            cam_key = 'cam1'
            intrinsics = self.cam1_intrinsics
            cam_serial = self.cam1_serial
        elif camera_id == 2:
            cam_key = 'cam2'
            intrinsics = self.cam2_intrinsics
            cam_serial = self.cam2_serial
        else:  # camera_id == 3
            cam_key = 'cam3'
            intrinsics = self.cam3_intrinsics
            cam_serial = self.cam3_serial

        # Clear previous data
        self.calibration_data[cam_key] = {'images': [], 'robot_poses': []}

        # Create visualization window
        window_name = f"Camera {camera_id} - Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        # Robot control starts immediately - no ENTER needed
        print(colored("✓ Robot control active! Hold RIGHT GRIP on Oculus to move the robot.", "green"))

        # FIRST ENTER: Start calibration in a separate thread
        def wait_for_calibration_start():
            global calibration_active
            input("Press ENTER when ready to start collecting calibration data...")
            with calibration_lock:
                calibration_active = True
            print(colored("✓ Calibration data collection started! Press ENTER to stop.", "green"))

        calibration_start_thread = threading.Thread(target=wait_for_calibration_start)
        calibration_start_thread.start()

        # SECOND ENTER: Stop calibration (will be started after calibration begins)
        def wait_for_calibration_stop():
            global calibration_active
            # Wait for calibration to actually start first
            while not calibration_active:
                time.sleep(0.1)
            # Now wait for user to stopX
            input()  # Wait for Enter to stop
            with calibration_lock:
                calibration_active = False

        stop_thread = threading.Thread(target=wait_for_calibration_stop)
        stop_thread.start()

        # Main control loop at ~15 Hz (matches RobotEnv.control_hz)
        frame_count = 0
        samples_collected = 0
        calibration_complete = False
        status_active = False  # True once the 3-line status block has been initialised

        while not calibration_complete:
            # Get current robot state for VRPolicy
            state_dict, _ = self.env.get_state()
            vr_obs = {"robot_state": state_dict}

            # Get Oculus action (hold right grip to enable movement)
            action = self.controller.forward(vr_obs)

            # Step environment with 7-DOF cartesian velocity action
            self.env.step(action)

            # Capture images
            images = self.capture_images()

            # Always show the live camera feed
            if cam_key in images:
                image = images[cam_key]
                cv2.imshow(window_name, image)
                cv2.waitKey(1)

                with calibration_lock:
                    is_active = calibration_active

                if is_active:
                    # On first active frame, print the 3 placeholder status lines
                    if not status_active:
                        sys.stdout.write("  Capturing: -\n")
                        sys.stdout.write("  Detection: -\n")
                        sys.stdout.write("  Samples:   0\n")
                        sys.stdout.flush()
                        status_active = True

                    # Detect ChArUco corners
                    corners, ids, _ = detect_charuco_corners(
                        image,
                        intrinsics['matrix'],
                        intrinsics['distortion']
                    )

                    # Save every 10th frame when corners detected
                    if corners is not None and frame_count % 10 == 0:
                        robot_pose = self.get_robot_pose()
                        self.calibration_data[cam_key]['images'].append((corners, ids))
                        self.calibration_data[cam_key]['robot_poses'].append(robot_pose)
                        samples_collected += 1

                    # Build status strings
                    cam_list = ', '.join(images.keys())
                    if corners is None:
                        det_str = f"frame {frame_count}: no detection"
                    else:
                        det_str = f"frame {frame_count}: {len(corners)} corners detected"

                    # Overwrite the 3 status lines in-place
                    sys.stdout.write('\033[3A')
                    sys.stdout.write(f'\033[2K  Capturing: [{cam_list}]\n')
                    sys.stdout.write(f'\033[2K  Detection: {det_str}\n')
                    sys.stdout.write(f'\033[2K  Samples:   {samples_collected}\n')
                    sys.stdout.flush()

                frame_count += 1

            # Check if stop thread has finished (user pressed ENTER twice)
            if not stop_thread.is_alive():
                calibration_complete = True

            time.sleep(0.01)  # Small delay

        cv2.destroyWindow(window_name)

        # Wait for threads to complete if they haven't already
        if calibration_start_thread.is_alive():
            calibration_start_thread.join()
        if stop_thread.is_alive():
            stop_thread.join()

        print(colored(f"\nCollected {samples_collected} calibration samples", "green"))

        if samples_collected < 10:
            print(colored("Not enough samples for calibration!", "red"))
            return None

        # Compute calibration
        return self.compute_calibration(cam_key, cam_serial, intrinsics, is_eye_in_hand)

    def compute_calibration(self, cam_key, cam_serial, intrinsics, is_eye_in_hand=False):
        """
        Compute camera transformation from collected data
        For third-person: computes camera-to-base transform
        For eye-in-hand: computes camera-to-gripper transform
        """
        print(colored("Computing calibration...", "yellow"))

        data = self.calibration_data[cam_key]

        # Prepare data for hand-eye calibration
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for i, ((corners, ids), robot_pose) in enumerate(zip(data['images'], data['robot_poses'])):
            # Robot pose: [x,y,z,rx,ry,rz]
            pos = robot_pose[:3]
            euler = robot_pose[3:6]

            # Get board pose in camera frame using PnP
            obj_points = CHARUCO_BOARD.chessboardCorners[ids.flatten()]
            _, rvec, tvec = cv2.solvePnP(
                obj_points,
                corners,
                intrinsics['matrix'],
                intrinsics['distortion']
            )

            # Convert to rotation matrix (target to camera)
            R_t2c = cv2.Rodrigues(rvec)[0]
            t_t2c = tvec.flatten()
            R_target2cam.append(R_t2c)
            t_target2cam.append(t_t2c)

            if is_eye_in_hand:
                # Eye-in-hand: camera moves with gripper, board is fixed
                # We solve for camera-to-gripper transform
                R_g2b = R.from_euler('xyz', euler).as_matrix()
                t_g2b = pos
                R_gripper2base.append(R_g2b)
                t_gripper2base.append(t_g2b)
            else:
                # Third-person: camera is fixed, board moves with gripper
                # We solve for camera-to-base transform
                # Need inverse transform since board is on gripper
                R_g2b = R.from_euler('xyz', euler).as_matrix()
                t_g2b = -R_g2b.T @ pos  # Inverse transform
                R_gripper2base.append(R_g2b.T)
                t_gripper2base.append(t_g2b)

        # Solve hand-eye calibration
        if is_eye_in_hand:
            # Eye-in-hand calibration (AX = XB)
            # A = gripper motion, X = camera-to-gripper (what we want), B = board motion in camera
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            if R_cam2gripper is None:
                print(colored("Calibration failed!", "red"))
                return None

            # Convert to 6D pose
            euler = R.from_matrix(R_cam2gripper).as_euler('xyz')
            pose = np.concatenate([t_cam2gripper.flatten(), euler])

            m = intrinsics['matrix']
            print(colored("✓ Eye-in-hand calibration successful!", "green"))
            print(f"  Camera-to-gripper transform:")
            print(f"  Position (x,y,z): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            print(f"  Rotation (r,p,y): [{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]")
            print(f"  Intrinsics (fx,fy,cx,cy): [{m[0,0]:.2f}, {m[1,1]:.2f}, {m[0,2]:.2f}, {m[1,2]:.2f}]")
            print(f"  Distortion: {intrinsics['distortion'].tolist()}")

            return {
                'serial': cam_serial,
                'pose': pose.tolist(),
                'type': 'eye_in_hand',
                'timestamp': time.time(),
                'intrinsics': {
                    'matrix': intrinsics['matrix'].tolist(),
                    'distortion': intrinsics['distortion'].tolist()
                }
            }
        else:
            # Third-person calibration (standard)
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )

            if R_cam2base is None:
                print(colored("Calibration failed!", "red"))
                return None

            # Convert to 6D pose
            euler = R.from_matrix(R_cam2base).as_euler('xyz')
            pose = np.concatenate([t_cam2base.flatten(), euler])

            m = intrinsics['matrix']
            print(colored("✓ Third-person calibration successful!", "green"))
            print(f"  Camera-to-base transform:")
            print(f"  Position (x,y,z): [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
            print(f"  Rotation (r,p,y): [{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]")
            print(f"  Intrinsics (fx,fy,cx,cy): [{m[0,0]:.2f}, {m[1,1]:.2f}, {m[0,2]:.2f}, {m[1,2]:.2f}]")
            print(f"  Distortion: {intrinsics['distortion'].tolist()}")

            return {
                'serial': cam_serial,
                'pose': pose.tolist(),
                'type': 'third_person',
                'timestamp': time.time(),
                'intrinsics': {
                    'matrix': intrinsics['matrix'].tolist(),
                    'distortion': intrinsics['distortion'].tolist()
                }
            }

    def save_calibration(self, calibration_data):
        """Save calibration to JSON file with timestamp"""
        if not calibration_data:
            return

        # Create timestamped filename
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam_type = "eye_in_hand" if calibration_data['type'] == 'eye_in_hand' else "third_person"
        cam_serial = calibration_data['serial']

        # Save to timestamped file (unique for each run)
        output_dir = "calibrations"
        os.makedirs(output_dir, exist_ok=True)
        timestamped_file = os.path.join(output_dir, f"calibration_{cam_serial}_{cam_type}_{timestamp_str}.json")

        with open(timestamped_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(colored(f"✓ Calibration saved to {timestamped_file}", "green"))

        # Also update the main calibration file (optional - keeps latest calibration for each camera)
        main_file = os.path.join(output_dir, "zed_calibration_latest.json")

        # Load existing calibrations
        if os.path.exists(main_file):
            with open(main_file, 'r') as f:
                all_calibrations = json.load(f)
        else:
            all_calibrations = {}

        # Update with new calibration
        all_calibrations[calibration_data['serial']] = {
            'pose': calibration_data['pose'],
            'type': calibration_data['type'],
            'timestamp': calibration_data['timestamp'],
            'intrinsics': calibration_data.get('intrinsics', {})
        }

        # Save to main file
        with open(main_file, 'w') as f:
            json.dump(all_calibrations, f, indent=2)

        print(colored(f"✓ Latest calibration also saved to {main_file}", "green"))

    def run(self, camera_id=None):
        """Main execution flow"""
        print(colored("\n=== ZED Camera Calibration System (Oculus VR Control) ===", "cyan", attrs=['bold']))

        # Initialize cameras (skips ones that aren't connected)
        if not self.init_cameras():
            return

        # Display connected cameras
        print("\nConnected cameras:")
        for internal_cam_id, cam_key, serial, intrinsics, is_eye_in_hand in self.connected_cameras:
            display_idx = internal_cam_id - 1
            cam_type = "Eye-in-hand" if is_eye_in_hand else "Third-person"
            print(f"  {display_idx}: Serial {serial} - {cam_type}")

        if not self.init_robot_and_oculus():
            return

        # Build valid display indices (0-based) for connected cameras
        valid_choices = [str(cam[0] - 1) for cam in self.connected_cameras]
        connected_internal_ids = [cam[0] for cam in self.connected_cameras]

        if camera_id is not None:
            internal_cam_id = camera_id + 1
            if internal_cam_id not in connected_internal_ids:
                print(colored(f"Camera {camera_id} is not connected. Available: {valid_choices}", "red"))
                return
            print(f"\n{'='*50}")
            print(colored(f"Calibrating Camera {camera_id} only", "yellow"))
            calib = self.run_calibration_loop(internal_cam_id)
            if calib:
                self.save_calibration(calib)
        else:
            print("\nWhich camera would you like to calibrate?")
            for internal_cam_id, cam_key, serial, intrinsics, is_eye_in_hand in self.connected_cameras:
                display_idx = internal_cam_id - 1
                cam_type = "eye-in-hand" if is_eye_in_hand else "third-person"
                print(f"  {display_idx}: Camera {display_idx} ({cam_type})")
            print("  all: Calibrate all connected cameras")

            choice = input(f"Enter camera number ({'/'.join(valid_choices)}/all) [default: all]: ").strip().lower() or 'all'

            if choice == 'all':
                for internal_cam_id, cam_key, serial, intrinsics, is_eye_in_hand in self.connected_cameras:
                    print(f"\n{'='*50}")
                    calib = self.run_calibration_loop(internal_cam_id)
                    if calib:
                        self.save_calibration(calib)
            elif choice in valid_choices:
                internal_cam_id = int(choice) + 1
                print(f"\n{'='*50}")
                calib = self.run_calibration_loop(internal_cam_id)
                if calib:
                    self.save_calibration(calib)
            else:
                print(colored(f"Invalid choice! Please enter one of: {', '.join(valid_choices)}, all", "red"))
                return

        print(colored("\n✓ Calibration complete!", "green"))

    def cleanup(self):
        """Clean up resources"""
        if self.zed1:
            self.zed1.close()
            print("Closed Camera 1")
        if self.zed2:
            self.zed2.close()
            print("Closed Camera 2")
        if self.zed3:
            self.zed3.close()
            print("Closed Camera 3")
        cv2.destroyAllWindows()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ZED camera calibration with Oculus VR control")
    parser.add_argument("--camera", type=int, default=None,
                       help="Camera to calibrate by 0-based index (0, 1, or 2). Defaults to interactive selection.")
    args = parser.parse_args()

    calibrator = SimpleZEDCalibrator()

    try:
        calibrator.run(camera_id=args.camera)
    except KeyboardInterrupt:
        print(colored("\nCalibration interrupted", "yellow"))
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()
    
    
    