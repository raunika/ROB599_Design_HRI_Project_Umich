import json
import socket
import threading
import sys
import time
from pathlib import Path

import yaml
from adafruit_servokit import ServoKit


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Gesture IDs arrive through the persistent SSH stdin session.
# Optional head-tracking input.
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Load actions.yaml from the same directory as this script, regardless of
# the directory from which the script is launched.
YAML_PATH = Path(__file__).resolve().with_name("actions.yaml")

# PCA9685 configuration (I2C).
PCA9685_ADDRESS = 0x40
PWM_FREQUENCY_HZ = 50

# Servo model assignment:
# - SG90: arms and eyelids
# - MG995: head pan and head tilt
#
# (Assuming "M995" means MG995.)
SERVO_MODELS = {
    "left_arm": "SG90",
    "right_arm": "SG90",
    "head_pan": "MG995",
    "head_tilt": "MG995",
    "left_eyelid": "SG90",
    "right_eyelid": "SG90",
}

# Logical angle span exposed by ServoKit. Both models are commonly sold as
# approximately 180-degree servos, but the usable mechanical range depends on
# the exact unit and linkage. The software limits below remain independently
# configurable for each mechanism.
SERVO_ACTUATION_RANGES = {
    "left_arm": 180,
    "right_arm": 180,
    "head_pan": 180,
    "head_tilt": 180,
    "left_eyelid": 180,
    "right_eyelid": 180,
}

# Conservative initial PWM pulse-width calibration in microseconds.
# These 1000-2000 us endpoints deliberately avoid the wider extremes sometimes
# quoted for SG90/MG995 servos, since clones vary and wider pulses can make a
# servo push against its mechanical stop. Calibrate each unit before widening.
SERVO_PULSE_WIDTHS_US = {
    "left_arm": (1000, 2000),       # SG90
    "right_arm": (1000, 2000),      # SG90
    "head_pan": (1000, 2000),      # MG995
    "head_tilt": (1000, 2000),     # MG995
    "left_eyelid": (1000, 2000),   # SG90
    "right_eyelid": (1000, 2000),  # SG90
}

# PCA9685 channel assignments. These are PCA9685 channel numbers, not Arduino
# GPIO pin numbers.
SERVO_CHANNELS = {
    "left_arm": 0,
    "right_arm": 1,
    "head_pan": 2,
    "head_tilt": 3,
    "left_eyelid": 4,
    "right_eyelid": 5,
}

# Current MOMO home positions.
HOME_POSITIONS = {
    "left_arm": 95,
    "right_arm": 75,
    "head_pan": 90,
    "head_tilt": 90,
    "left_eyelid": 90,
    "right_eyelid": 90,
}

# Mechanical command limits. These are the standard ServoKit degree limits.
# Tighten an individual min/max after physically calibrating that servo.
SERVO_LIMITS = {
    "left_arm": (0, 180),
    "right_arm": (0, 180),
    "head_pan": (0, 180),
    "head_tilt": (0, 180),
    "left_eyelid": (0, 180),
    "right_eyelid": (0, 180),
}

# Accept the older names used in earlier action files as aliases.
SERVO_ALIASES = {
    "head_yaw": "head_pan",
    "head_pitch": "head_tilt",
    "left_eye": "left_eyelid",
    "right_eye": "right_eyelid",
}


class ServoController:
    def __init__(self):
        print(
            f"[Servo] Initializing PCA9685 at I2C address "
            f"0x{PCA9685_ADDRESS:02X}, {PWM_FREQUENCY_HZ} Hz..."
        )
        self.kit = ServoKit(
            channels=16,
            address=PCA9685_ADDRESS,
            frequency=PWM_FREQUENCY_HZ,
        )

        # Configure each PCA9685 output's logical angle range and conservative
        # pulse-width endpoints for its assigned servo model.
        for name, channel in SERVO_CHANNELS.items():
            servo = self.kit.servo[channel]
            servo.actuation_range = SERVO_ACTUATION_RANGES[name]

            pulse_range = SERVO_PULSE_WIDTHS_US[name]
            if pulse_range is not None:
                min_pulse, max_pulse = pulse_range
                servo.set_pulse_width_range(min_pulse, max_pulse)

            print(
                f"[Servo] {name}: {SERVO_MODELS[name]} on PCA9685 "
                f"channel {channel}, {pulse_range[0]}-{pulse_range[1]} us"
            )

        self.actions = {}
        self.load_actions()

        self.llm_lock = threading.Lock()
        self.is_executing_llm = threading.Event()

        self.perception_data = {"yaw": HOME_POSITIONS["head_pan"],
                                "pitch": HOME_POSITIONS["head_tilt"]}
        self.perception_lock = threading.Lock()

        self.reset_to_home()
        print("[Servo] Ready.")

    def load_actions(self):
        with open(YAML_PATH, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        self.actions = data.get("actions", {})
        if not isinstance(self.actions, dict):
            raise ValueError("actions.yaml must contain an 'actions' mapping.")

        print(f"[Actions] Loaded {len(self.actions)} actions from {YAML_PATH}.")

    def normalize_servo_name(self, name):
        return SERVO_ALIASES.get(name, name)

    def set_servo(self, name, angle):
        name = self.normalize_servo_name(name)

        if name not in SERVO_CHANNELS:
            print(f"[Servo] Warning: Unknown servo '{name}'.")
            return

        try:
            angle = float(angle)
        except (TypeError, ValueError):
            print(f"[Servo] Warning: Invalid angle {angle!r} for {name}.")
            return

        minimum, maximum = SERVO_LIMITS[name]
        angle = max(minimum, min(maximum, angle))
        self.kit.servo[SERVO_CHANNELS[name]].angle = float(angle)

    def reset_to_home(self, exclude_head=False):
        for name, angle in HOME_POSITIONS.items():
            if exclude_head and name in ("head_pan", "head_tilt"):
                continue
            self.set_servo(name, angle)

    def _get_action(self, action_id):
        # YAML parsers may load numeric keys as integers or strings.
        return self.actions.get(str(action_id), self.actions.get(action_id))

    def execute_gesture_ids(self, gesture_ids):
        """Execute a list of gesture/action IDs in the order received."""
        if not isinstance(gesture_ids, list):
            gesture_ids = [gesture_ids]

        # Prevent overlapping gesture sequences.
        if not self.llm_lock.acquire(blocking=False):
            print("[Actions] Busy; ignoring overlapping gesture request.")
            return

        self.is_executing_llm.set()
        try:
            print(f"[Actions] Executing gesture IDs: {gesture_ids}")

            for action_id in gesture_ids:
                try:
                    action_id = int(action_id)
                except (TypeError, ValueError):
                    print(f"[Actions] Invalid gesture ID: {action_id!r}")
                    continue

                action = self._get_action(action_id)
                if not isinstance(action, dict):
                    print(f"[Actions] Gesture ID {action_id} not found in actions.yaml.")
                    continue

                # Supports the sequence-based actions.yaml structure.
                sequence = action.get("sequence", [])
                if not isinstance(sequence, list):
                    print(f"[Actions] Gesture {action_id} has an invalid sequence.")
                    continue

                for step in sequence:
                    if not isinstance(step, dict):
                        continue

                    angles = step.get("angles", {})
                    if isinstance(angles, dict):
                        for servo_name, angle in angles.items():
                            self.set_servo(servo_name, angle)

                    duration = step.get("duration", 0.1)
                    try:
                        time.sleep(max(0.0, float(duration)))
                    except (TypeError, ValueError):
                        time.sleep(0.1)

            print("[Actions] Sequence complete. Returning arms and eyelids to home.")
            self.reset_to_home(exclude_head=True)
            time.sleep(0.25)

        finally:
            self.is_executing_llm.clear()
            self.llm_lock.release()
            print("[Actions] Head tracking may resume.")

    def _dispatch_gesture_message(self, line):
        """
        Accept all formats used by the MOMO Mac-side sender:
          1) One integer per line: 3\n
          2) JSON list: [3, 1]
          3) Ollama-style JSON object: {"response": "...", "gesture": [3, 1]}
        """
        line = line.strip()
        if not line:
            return

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # The current Mac script sends one plain integer followed by newline.
            try:
                gesture_id = int(line)
            except ValueError:
                print(f"[SSH] Ignoring unrecognized message: {line!r}")
                return
            self.execute_gesture_ids([gesture_id])
            return

        if isinstance(payload, int):
            self.execute_gesture_ids([payload])
        elif isinstance(payload, list):
            self.execute_gesture_ids(payload)
        elif isinstance(payload, dict):
            gesture_ids = payload.get("gesture", payload.get("gestures", []))
            if gesture_ids:
                if not isinstance(gesture_ids, list):
                    gesture_ids = [gesture_ids]
                self.execute_gesture_ids(gesture_ids)
            else:
                print("[SSH] JSON received, but it contains no 'gesture' field.")
        else:
            print(f"[SSH] Unsupported JSON payload: {type(payload).__name__}")

    def ssh_stdin_listener(self):
        """Read newline-terminated gesture IDs arriving through the SSH session."""
        print("[SSH] Waiting for gesture IDs on stdin...")
        try:
            for line in sys.stdin:
                self._dispatch_gesture_message(line)
        except Exception as error:
            print(f"[SSH] Input error: {error}")
        finally:
            print("[SSH] Input stream closed. Returning servos home.")
            self.reset_to_home()

    def start_ssh_listener(self):
        """Run the servo controller as a persistent process launched over SSH."""
        threading.Thread(target=self.perception_listener, daemon=True).start()
        threading.Thread(target=self.tracking_loop, daemon=True).start()
        self.ssh_stdin_listener()

    def perception_listener(self):
        print(f"[Perception] UDP listener on {UDP_IP}:{UDP_PORT}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))

        while True:
            try:
                data, _ = sock.recvfrom(1024)
                packet = json.loads(data.decode("utf-8"))

                yaw = packet.get("yaw")
                pitch = packet.get("pitch")

                if yaw is not None and pitch is not None:
                    with self.perception_lock:
                        self.perception_data["yaw"] = yaw
                        self.perception_data["pitch"] = pitch

            except Exception as error:
                print(f"[Perception] Error reading packet: {error}")

    def tracking_loop(self):
        print("[Tracking] Head tracking loop running at approximately 2 Hz.")
        while True:
            if not self.is_executing_llm.is_set():
                with self.perception_lock:
                    yaw = self.perception_data["yaw"]
                    pitch = self.perception_data["pitch"]

                self.set_servo("head_pan", yaw)
                self.set_servo("head_tilt", pitch)

            time.sleep(0.5)

    def start(self):
        # Standalone mode: run perception-based head tracking only.
        threading.Thread(target=self.perception_listener, daemon=True).start()
        self.tracking_loop()


if __name__ == "__main__":
    controller = ServoController()
    try:
        if "--ssh-listener" in sys.argv:
            controller.start_ssh_listener()
        else:
            controller.start()
    except KeyboardInterrupt:
        print("\\n[System] Shutting down cleanly...")
        controller.reset_to_home()
