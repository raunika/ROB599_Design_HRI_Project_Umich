import json
import socket
import threading
import time
import serial
import yaml
from adafruit_servokit import ServoKit

# --- CONFIGURATION ---
SERIAL_PORT = "/dev/ttyACM0"  # Adjust to /dev/ttyUSB0 if needed
BAUD_RATE = 115200
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
YAML_PATH = "actions.yaml"

# Servo Channel Mapping
SERVO_CHANNELS = {
    "left_arm": 0,
    "right_arm": 1,
    "head_yaw": 2,
    "head_pitch": 3,
    "left_eye": 4,
    "right_eye": 5,
}

# Center/Default positions (Degrees)
CENTER_POSITIONS = {
    "left_arm": 90,
    "right_arm": 90,
    "head_yaw": 90,
    "head_pitch": 90,
    "left_eye": 0,
    "right_eye": 0,
}

class ServoController:
    def __init__(self):
        # Initialize PCA9685 (16 channels, default I2C address 0x40)
        self.kit = ServoKit(channels=16)
        self.load_actions()
        
        # Thread synchronization
        self.llm_lock = threading.Lock()
        self.is_executing_llm = False

        # Latest perception data container
        self.perception_data = {"yaw": 90, "pitch": 90}
        self.perception_lock = threading.Lock()

        # Initialize servos to center
        self.reset_to_center()

    def load_actions(self):
        with open(YAML_PATH, "r") as f:
            data = yaml.safe_load(f)
            self.actions = data.get("actions", {})

    def set_servo(self, name, angle):
        if name in SERVO_CHANNELS:
            # Clamp angle between 0 and 180
            angle = max(0, min(180, angle))
            self.kit.servo[SERVO_CHANNELS[name]].angle = angle

    def reset_to_center(self, exclude_head=False):
        """Reset servos to center, optionally excluding head pitch/yaw for smooth tracking."""
        for name, angle in CENTER_POSITIONS.items():
            if exclude_head and name in ["head_yaw", "head_pitch"]:
                continue
            self.set_servo(name, angle)

    def execute_llm_actions(self, action_ids):
        with self.llm_lock:
            self.is_executing_llm = True
            print(f"[LLM] Executing actions: {action_ids}")

            for act_id in action_ids:
                action = self.actions.get(int(act_id))
                if not action:
                    print(f"[LLM] Warning: Action ID {act_id} not found in YAML.")
                    continue

                for step in action.get("sequence", []):
                    angles = step.get("angles", {})
                    duration = step.get("duration", 0.1)
                    for servo_name, angle in angles.items():
                        self.set_servo(servo_name, angle)
                    time.sleep(duration)

            # After LLM finishes: return all servos except pitch/yaw to center
            print("[LLM] Actions completed. Centering body/arms/eyes...")
            self.reset_to_center(exclude_head=True)

            # Wait 0.25 seconds before resuming perception tracking
            time.sleep(0.25)
            self.is_executing_llm = False
            print("[LLM] Resumed perception tracking.")

    def serial_listener(self):
        print(f"[Serial] Listening on {SERIAL_PORT}...")
        while True:
            try:
                with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
                    while True:
                        line = ser.readline().decode("utf-8").strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if isinstance(data, list):
                                    # Trigger LLM action sequence
                                    threading.Thread(
                                        target=self.execute_llm_actions, 
                                        args=(data,), 
                                        daemon=True
                                    ).start()
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                print(f"[Serial] Connection error: {e}. Retrying in 3 seconds...")
                time.sleep(3)

    def perception_listener(self):
        print(f"[Perception] UDP server listening on {UDP_IP}:{UDP_PORT}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                packet = json.loads(data.decode("utf-8"))
                
                # Rule 9: Extract only yaw and pitch, ignore user_name, in_frame, is_happy
                yaw = packet.get("yaw")
                pitch = packet.get("pitch")
                
                if yaw is not None and pitch is not None:
                    with self.perception_lock:
                        self.perception_data["yaw"] = yaw
                        self.perception_data["pitch"] = pitch
            except Exception as e:
                print(f"[Perception] Error reading packet: {e}")

    def tracking_loop(self):
        print("[Tracking] Head tracking loop started (2Hz rate target)...")
        while True:
            # Only track if LLM is not currently executing an override action
            if not self.is_executing_llm:
                with self.perception_lock:
                    target_yaw = self.perception_data["yaw"]
                    target_pitch = self.perception_data["pitch"]
                
                self.set_servo("head_yaw", target_yaw)
                self.set_servo("head_pitch", target_pitch)
            
            # Perception runs at ~2Hz (update every 0.5 seconds)
            time.sleep(0.5)

    def start(self):
        # Start background threads
        threading.Thread(target=self.serial_listener, daemon=True).start()
        threading.Thread(target=self.perception_listener, daemon=True).start()
        
        # Run tracking loop on main thread
        self.tracking_loop()

if __name__ == "__main__":
    controller = ServoController()
    try:
        controller.start()
    except KeyboardInterrupt:
        print("Shutting down cleanly...")
        controller.reset_to_center()