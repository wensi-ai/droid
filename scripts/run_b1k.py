# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from scripts.network_utils import WebsocketClientPolicy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

from droid.camera_utils.video_depth_anything import VideoDepthAnythingEstimator

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


def _compose_rollout_video_frame(
    rgb_external: np.ndarray,
    rgb_wrist: np.ndarray,
    *,
    depth_external: np.ndarray | None = None,
    depth_wrist: np.ndarray | None = None,
) -> np.ndarray:
    """Stack cameras vertically and, when available, place depth beside RGB."""
    rgb_column = np.concatenate([rgb_external, rgb_wrist], axis=0)
    if depth_external is None and depth_wrist is None:
        return rgb_column
    if depth_external is None or depth_wrist is None:
        raise ValueError("Both external and wrist depth frames are required for depth video output")

    depth_column = np.concatenate([depth_external, depth_wrist], axis=0)
    if depth_column.shape != rgb_column.shape:
        raise ValueError(
            f"RGB and depth video columns must have the same shape, got {rgb_column.shape} and {depth_column.shape}"
        )
    return np.concatenate([rgb_column, depth_column], axis=1)


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "38178251"  # e.g., "24514023"
    # right_camera_id: str = "39762559"  # e.g., "2425987687"
    wrist_camera_id: str = "16606959"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        "left"  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Camera parameters
    # Manual exposure value (0-100). Set to None to use auto exposure.
    exposure: int | None = 40

    # Replace the RGB policy inputs with Video Depth Anything visualizations.
    depth: bool = False
    # The local Video Depth Anything checkout. The default resolves to
    # ../Video-Depth-Anything relative to this DROID checkout.
    depth_model_root: str | None = None
    # Optional checkpoint override. By default, use
    # <depth_model_root>/checkpoints/video_depth_anything_<encoder>.pth.
    depth_checkpoint: str | None = None
    # Video Depth Anything encoder. The installed checkpoint uses the small model.
    depth_encoder: str = "vits"
    # Short-side inference resolution. 392 balances quality and rollout latency.
    depth_input_size: int = 392
    # Use full precision instead of the default CUDA mixed precision.
    depth_fp32: bool = False

    # Rollout parameters
    max_timesteps: int = 600

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    depth_estimator = None
    if args.depth:
        depth_estimator = VideoDepthAnythingEstimator(
            model_root=args.depth_model_root,
            checkpoint_path=args.depth_checkpoint,
            encoder=args.depth_encoder,
            input_size=args.depth_input_size,
            fp32=args.depth_fp32,
        )
        print(
            "Enabled Video Depth Anything "
            f"({args.depth_encoder}, input size {args.depth_input_size}, device {depth_estimator.device})."
        )

    # Initialize the Panda environment. Using joint position action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    # Set camera exposure
    for cam_id in [args.left_camera_id, args.wrist_camera_id]:
        cam = env.camera_reader.get_camera(cam_id)
        if args.exposure is None:
            cam.set_exposure(auto=True)
        else:
            cam.set_exposure(exposure_value=args.exposure, auto=False)
            print(f"Camera {cam_id}: set manual exposure to {args.exposure}")

    # Connect to the policy server
    policy_client = WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")
        # reset the policy client
        policy_client.reset()
        if depth_estimator is not None:
            depth_estimator.reset()

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                policy_external_image = curr_obs[f"{args.external_camera}_image"]
                policy_wrist_image = curr_obs["wrist_image"]
                if depth_estimator is not None:
                    policy_external_image, _ = depth_estimator.infer(
                        policy_external_image,
                        stream_name="external",
                    )
                    policy_wrist_image, _ = depth_estimator.infer(
                        policy_wrist_image,
                        stream_name="wrist",
                    )
                    if t_step == 0:
                        Image.fromarray(
                            np.concatenate([policy_external_image, policy_wrist_image], axis=1)
                        ).save("robot_depth_views.png")

                    video.append(
                        _compose_rollout_video_frame(
                            curr_obs[f"{args.external_camera}_image"],
                            curr_obs["wrist_image"],
                            depth_external=policy_external_image,
                            depth_wrist=policy_wrist_image,
                        )
                    )
                else:
                    video.append(
                        _compose_rollout_video_frame(
                            curr_obs[f"{args.external_camera}_image"],
                            curr_obs["wrist_image"],
                        )
                    )

                # # Visualize camera feeds (images are RGB; convert to BGR for cv2)
                # vis_external = curr_obs[f"{args.external_camera}_image"][..., ::-1]
                # vis_wrist = curr_obs["wrist_image"][..., ::-1]
                # combined_vis = np.concatenate([vis_external, vis_wrist], axis=1)
                # cv2.imshow("External | Wrist", combined_vis)
                # cv2.waitKey(1)

                # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                # and improve latency.
                external_obs_key = "external::external_camera_1::rgb"
                wrist_obs_key = "robot::robot:camera_link:Camera:0::rgb"
                if depth_estimator is not None:
                    # Depth robot configs use the simulator's depth_linear key
                    # names. These values are already encoded uint8 depth images,
                    # so the OpenPI server preserves them instead of interpreting
                    # them as raw metric depth.
                    external_obs_key = "external::external_camera_1::depth_linear"
                    wrist_obs_key = "robot::robot:camera_link:Camera:0::depth_linear"
                request_data = {
                    external_obs_key: image_tools.resize_with_pad(
                        policy_external_image, 224, 224
                    ),
                    wrist_obs_key: image_tools.resize_with_pad(
                        policy_wrist_image, 224, 224
                    ),
                    "robot::proprio": np.concatenate([curr_obs["joint_position"], curr_obs["gripper_position"]]),
                    "prompt": instruction,
                }
                # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                # Ctrl+C will be handled after the server call is complete
                with prevent_keyboard_interrupt():
                    # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                    action = policy_client.act(request_data)
                assert action.shape == (8,), f"Expected action of shape (8,) but got {action.shape}"

                # Binarize gripper action (open: 1 -> 0, close: -1 -> 1)
                if action[-1].item() < 0:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        video_suffix = "_rgb_depth" if depth_estimator is not None else ""
        save_filename = "video_" + timestamp + video_suffix
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "success": success,
                            "duration": t_step,
                            "video_filename": save_filename,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, wrist_image = None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"], robot_state["gripper_position"]]) * np.pi / 4

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
