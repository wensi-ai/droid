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

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


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

                video.append(
                    np.concatenate([curr_obs[f"{args.external_camera}_image"], curr_obs["wrist_image"]], axis=0)
                )

                # # Visualize camera feeds (images are RGB; convert to BGR for cv2)
                # vis_external = curr_obs[f"{args.external_camera}_image"][..., ::-1]
                # vis_wrist = curr_obs["wrist_image"][..., ::-1]
                # combined_vis = np.concatenate([vis_external, vis_wrist], axis=1)
                # cv2.imshow("External | Wrist", combined_vis)
                # cv2.waitKey(1)

                # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                # and improve latency.
                request_data = {
                    "external::external_camera_1::rgb": image_tools.resize_with_pad(
                        curr_obs[f"{args.external_camera}_image"], 224, 224
                    ),
                    "robot::robot:camera_link:Camera:0::rgb": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
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
        save_filename = "video_" + timestamp
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
