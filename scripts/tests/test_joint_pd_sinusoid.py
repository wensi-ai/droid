#!/usr/bin/env python3
"""Stream sinusoidal absolute joint targets and log robot tracking data.

Run from the DROID repo with the DROID conda environment active:

    conda activate droid
    python scripts/tests/test_joint_pd_sinusoid.py --yes

The script connects to the DROID robot server on the NUC by default. The NUC
should be running:

    conda activate droid
    python scripts/server/run_server.py

Each sample commands an absolute 7-DoF joint target, then records the target
and the measured robot state. By default the script excites one joint at a time
while holding the other joints at the reset pose. Logs are written as JSONL
during the run and as a compressed NPZ at the end for later PD calibration in
simulation.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from droid.misc.parameters import nuc_ip as configured_nuc_ip
except ImportError:
    configured_nuc_ip = None

try:
    from droid.robot_env import ROBOT_ENV_RESET_JOINTS
except ImportError:
    ROBOT_ENV_RESET_JOINTS = np.array(
        [0.0, -1.0 / 5.0 * math.pi, 0.0, -4.0 / 5.0 * math.pi, 0.0, 3.0 / 5.0 * math.pi, 0.0]
    )

DEFAULT_NUC_IP = configured_nuc_ip or "172.16.0.3"
PANDA_JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def parse_args():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Command sinusoidal absolute joint positions and log commanded/actual joint states."
    )
    parser.add_argument("--host", default=DEFAULT_NUC_IP, help="DROID robot server IP address.")
    parser.add_argument(
        "--backend",
        choices=["server", "local"],
        default="server",
        help="Use the zerorpc DROID server, or run directly against local Polymetis on the NUC.",
    )
    parser.add_argument(
        "--launch-controller",
        action="store_true",
        help="Ask the DROID server/local client to launch the Polymetis controller before connecting.",
    )
    parser.add_argument(
        "--no-launch-robot",
        action="store_true",
        help="Skip launch_robot(); use only if the server already has an initialized RobotInterface.",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Sinusoid duration in seconds per joint.")
    parser.add_argument("--hz", type=float, default=15.0, help="Command/log frequency in Hz.")
    parser.add_argument(
        "--read-delay",
        type=float,
        default=0.02,
        help="Seconds to wait after each command before reading actual joint pose.",
    )
    parser.add_argument("--frequency", type=float, default=0.45, help="Sinusoid frequency in Hz.")
    parser.add_argument(
        "--amplitude",
        default="0.225",
        help="Joint sine amplitude in radians. Use scalar, 7 comma-separated values, or one value per selected joint.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Joint sine phase in radians. Defaults to zero phase so each joint starts at center.",
    )
    parser.add_argument(
        "--joint-indices",
        default="0,1,2,3,4,5,6",
        help="Comma-separated zero-based joint indices to excite.",
    )
    parser.add_argument(
        "--between-joint-settle-seconds",
        type=float,
        default=1.0,
        help="Seconds to hold center after each joint segment before moving to the next joint.",
    )
    parser.add_argument(
        "--center-joints",
        default=None,
        help="Optional 7 comma-separated absolute joint center. Defaults to ROBOT_ENV_RESET_JOINTS.",
    )
    parser.add_argument(
        "--skip-initial-reset",
        action="store_true",
        help="Do not do the initial blocking move to the sinusoid center before streaming.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait after loading the joint policy and before streaming commands.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "logs" / "pd_calibration",
        help="Directory for JSONL and NPZ logs.",
    )
    parser.add_argument(
        "--run-name",
        default=f"joint_pd_sinusoid_{timestamp}",
        help="Base name for output files.",
    )
    parser.add_argument(
        "--no-return-to-center",
        action="store_true",
        help="Do not move back to the sine center after a normal finish or Ctrl-C.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive hardware-motion confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not connect to or command hardware; generate the intended targets only.",
    )
    return parser.parse_args()


def parse_float_vector(text, name):
    try:
        return np.array([float(part.strip()) for part in text.split(",") if part.strip() != ""], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse --{name}={text!r} as comma-separated floats.") from exc


def parse_index_vector(text):
    try:
        indices = np.array([int(part.strip()) for part in text.split(",") if part.strip() != ""], dtype=int)
    except ValueError as exc:
        raise ValueError(f"Could not parse --joint-indices={text!r} as comma-separated integers.") from exc
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices > 6):
        raise ValueError("--joint-indices must contain at least one index in [0, 6].")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("--joint-indices contains duplicates.")
    return indices


def expand_to_joints(values, joint_indices, name, default=None):
    result = np.zeros(7, dtype=float) if default is None else np.array(default, dtype=float)
    values = np.array(values, dtype=float)

    if len(values) == 1:
        result[joint_indices] = values[0]
    elif len(values) == 7:
        result[:] = values
    elif len(values) == len(joint_indices):
        result[joint_indices] = values
    else:
        raise ValueError(f"--{name} must be scalar, length 7, or length {len(joint_indices)}.")
    return result


def connect_robot(args):
    if args.backend == "server":
        from droid.misc.server_interface import ServerInterface

        robot = ServerInterface(ip_address=args.host, launch=False)
    else:
        from droid.franka.robot import FrankaRobot

        robot = FrankaRobot()

    if args.launch_controller:
        print("Launching Polymetis controller through DROID interface...")
        robot.launch_controller()
    if not args.no_launch_robot:
        print("Initializing RobotInterface...")
        robot.launch_robot()
    return robot


def get_robot_state(robot, dry_run, fallback_joints):
    if dry_run:
        state = {
            "joint_positions": fallback_joints.tolist(),
            "joint_velocities": np.zeros(7).tolist(),
            "joint_torques_computed": np.zeros(7).tolist(),
            "motor_torques_measured": np.zeros(7).tolist(),
            "prev_command_successful": True,
            "prev_controller_latency_ms": 0.0,
        }
        return state, {}
    return robot.get_robot_state()


def command_joints(robot, target, dry_run, blocking=False):
    if dry_run:
        return
    robot.update_joints(np.asarray(target, dtype=float), velocity=False, blocking=blocking)


def validate_motion(center, amplitude):
    low = center - np.abs(amplitude)
    high = center + np.abs(amplitude)
    if np.any(low < PANDA_JOINT_LOWER) or np.any(high > PANDA_JOINT_UPPER):
        bad = np.where((low < PANDA_JOINT_LOWER) | (high > PANDA_JOINT_UPPER))[0]
        details = ", ".join(
            f"joint {idx}: [{low[idx]:.3f}, {high[idx]:.3f}] outside "
            f"[{PANDA_JOINT_LOWER[idx]:.3f}, {PANDA_JOINT_UPPER[idx]:.3f}]"
            for idx in bad
        )
        raise ValueError(f"Requested sinusoid exceeds conservative Franka joint limits: {details}")


def validate_timing(args):
    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    if args.hz <= 0:
        raise ValueError("--hz must be positive.")
    if args.frequency <= 0:
        raise ValueError("--frequency must be positive.")
    if args.read_delay < 0:
        raise ValueError("--read-delay must be non-negative.")
    if args.read_delay >= 1.0 / args.hz:
        raise ValueError("--read-delay must be smaller than the command period 1 / --hz.")
    if args.between_joint_settle_seconds < 0:
        raise ValueError("--between-joint-settle-seconds must be non-negative.")


def confirm_hardware_motion(args, center, amplitude, frequency, joint_indices):
    if args.yes or args.dry_run:
        return

    print("\nThis will command absolute joint positions on the real robot.")
    print(f"Host/backend: {args.host} / {args.backend}")
    print(f"Duration/rate: {args.duration:.1f}s per joint at {args.hz:.1f} Hz")
    print(f"Estimated total command duration: {args.duration * len(joint_indices):.1f}s")
    print(f"Frequency: {frequency:.3f} Hz")
    print(f"Excited joints, one at a time: {joint_indices.tolist()}")
    print(f"Center: {np.array2string(center, precision=4)}")
    print(f"Amplitude: {np.array2string(amplitude, precision=4)}")
    response = input("Type 'move robot' to start: ").strip().lower()
    if response != "move robot":
        raise SystemExit("Aborted before commanding hardware.")


def write_metadata(path, args, center, amplitude, phase, joint_indices):
    metadata = {
        "created_wall_time": datetime.now().isoformat(),
        "host": args.host,
        "backend": args.backend,
        "trajectory_type": "sequential_per_joint_sinusoid",
        "duration_per_joint": args.duration,
        "total_command_duration": args.duration * len(joint_indices),
        "hz": args.hz,
        "read_delay": args.read_delay,
        "frequency": args.frequency,
        "joint_indices": joint_indices.tolist(),
        "between_joint_settle_seconds": args.between_joint_settle_seconds,
        "center_joint_positions": center.tolist(),
        "center_source": "robot_env.py:ROBOT_ENV_RESET_JOINTS" if args.center_joints is None else "--center-joints",
        "amplitude": amplitude.tolist(),
        "phase": phase.tolist(),
        "initial_reset": not args.skip_initial_reset,
        "launch_controller": args.launch_controller,
        "launch_robot": not args.no_launch_robot,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    args = parse_args()
    validate_timing(args)
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != "droid":
        print(f"WARNING: CONDA_DEFAULT_ENV is {conda_env!r}; expected 'droid'.")

    joint_indices = parse_index_vector(args.joint_indices)
    amplitude = expand_to_joints(parse_float_vector(args.amplitude, "amplitude"), joint_indices, "amplitude")

    if args.phase is None:
        phase = np.zeros(7, dtype=float)
    else:
        phase = expand_to_joints(parse_float_vector(args.phase, "phase"), joint_indices, "phase")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.run_name}.jsonl"
    npz_path = args.output_dir / f"{args.run_name}.npz"
    metadata_path = args.output_dir / f"{args.run_name}_metadata.json"

    robot = None
    initial_state = None
    center = np.array(ROBOT_ENV_RESET_JOINTS, dtype=float)
    if args.dry_run:
        pass
    else:
        robot = connect_robot(args)
        initial_state, _ = get_robot_state(robot, args.dry_run, np.zeros(7))

    if args.center_joints is not None:
        center = parse_float_vector(args.center_joints, "center-joints")
        if len(center) != 7:
            raise ValueError("--center-joints must contain exactly 7 values.")

    validate_motion(center, amplitude)
    confirm_hardware_motion(args, center, amplitude, args.frequency, joint_indices)
    write_metadata(metadata_path, args, center, amplitude, phase, joint_indices)

    if not args.skip_initial_reset:
        print("Moving to sinusoid center joint pose...")
        command_joints(robot, center, args.dry_run, blocking=True)

    if args.settle_seconds > 0 and not args.dry_run:
        print(f"Settling for {args.settle_seconds:.2f}s...")
        time.sleep(args.settle_seconds)

    period = 1.0 / args.hz
    steps_per_joint = int(math.ceil(args.duration * args.hz))
    rows = []
    stop_reason = "completed"

    print(f"Logging to {jsonl_path}")
    start_monotonic = time.monotonic()
    next_tick = start_monotonic
    step = 0

    try:
        with jsonl_path.open("w") as jsonl_file:
            for active_joint_order, active_joint in enumerate(joint_indices.tolist()):
                if active_joint_order > 0:
                    print(f"Returning to center before joint {active_joint}...")
                    command_joints(robot, center, args.dry_run, blocking=True)
                    if args.between_joint_settle_seconds > 0 and not args.dry_run:
                        time.sleep(args.between_joint_settle_seconds)

                print(f"Exciting joint {active_joint}...")
                next_tick = time.monotonic()
                segment_start = next_tick

                for joint_step in range(steps_per_joint):
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(next_tick - now)

                    command_monotonic = time.monotonic()
                    elapsed = command_monotonic - start_monotonic
                    segment_elapsed = command_monotonic - segment_start
                    target = center.copy()
                    target[active_joint] = center[active_joint] + amplitude[active_joint] * np.sin(
                        2.0 * math.pi * args.frequency * segment_elapsed + phase[active_joint]
                    )
                    command_joints(robot, target, args.dry_run, blocking=False)

                    if args.read_delay > 0 and not args.dry_run:
                        time.sleep(args.read_delay)
                    state, robot_timestamp = get_robot_state(robot, args.dry_run, target)
                    actual = np.array(state["joint_positions"], dtype=float)
                    row = {
                        "step": step,
                        "joint_step": joint_step,
                        "active_joint_order": active_joint_order,
                        "active_joint": active_joint,
                        "t_command_monotonic": elapsed,
                        "t_segment_monotonic": segment_elapsed,
                        "wall_time_command": time.time(),
                        "intended_action_joint_positions": target.tolist(),
                        "actual_joint_positions": actual.tolist(),
                        "actual_joint_velocities": state.get("joint_velocities"),
                        "joint_torques_computed": state.get("joint_torques_computed"),
                        "motor_torques_measured": state.get("motor_torques_measured"),
                        "prev_command_successful": state.get("prev_command_successful"),
                        "prev_controller_latency_ms": state.get("prev_controller_latency_ms"),
                        "robot_timestamp": robot_timestamp,
                    }
                    jsonl_file.write(json.dumps(row) + "\n")
                    jsonl_file.flush()
                    rows.append(row)

                    step += 1
                    next_tick += period
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\nInterrupted; preserving logs collected so far.")
    finally:
        if robot is not None and not args.no_return_to_center:
            print("Returning to center joint pose...")
            command_joints(robot, center, args.dry_run, blocking=True)

    if rows:
        np.savez_compressed(
            npz_path,
            intended_action_joint_positions=np.array(
                [row["intended_action_joint_positions"] for row in rows], dtype=float
            ),
            actual_joint_positions=np.array([row["actual_joint_positions"] for row in rows], dtype=float),
            actual_joint_velocities=np.array([row["actual_joint_velocities"] for row in rows], dtype=float),
            active_joint=np.array([row["active_joint"] for row in rows], dtype=int),
            active_joint_order=np.array([row["active_joint_order"] for row in rows], dtype=int),
            joint_step=np.array([row["joint_step"] for row in rows], dtype=int),
            t_command_monotonic=np.array([row["t_command_monotonic"] for row in rows], dtype=float),
            t_segment_monotonic=np.array([row["t_segment_monotonic"] for row in rows], dtype=float),
            wall_time_command=np.array([row["wall_time_command"] for row in rows], dtype=float),
            center_joint_positions=center,
            amplitude=amplitude,
            phase=phase,
            joint_indices=joint_indices,
            frequency=np.array(args.frequency, dtype=float),
            hz=np.array(args.hz, dtype=float),
            read_delay=np.array(args.read_delay, dtype=float),
            duration_per_joint=np.array(args.duration, dtype=float),
            between_joint_settle_seconds=np.array(args.between_joint_settle_seconds, dtype=float),
            stop_reason=np.array(stop_reason),
        )
        print(f"Wrote {len(rows)} samples to {jsonl_path}")
        print(f"Wrote arrays to {npz_path}")
    else:
        print("No samples were collected.")

    if initial_state is not None:
        print(f"Initial joints: {np.array2string(np.array(initial_state['joint_positions']), precision=4)}")
    print(f"Center joints: {np.array2string(center, precision=4)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
