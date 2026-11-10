import time
import yaml
from adafruit_servokit import ServoKit

# --- CONFIGURATION ---
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

class ServoDebugController:
    def __init__(self):
        # Initialize PCA9685 (16 channels, default I2C address 0x40)
        print("[Init] Initializing ServoKit...")
        self.kit = ServoKit(channels=16)
        self.load_actions()
        
        # Initialize servos to center
        print("[Init] Resetting all servos to center positions...")
        self.reset_to_center()

    def load_actions(self):
        try:
            with open(YAML_PATH, "r") as f:
                data = yaml.safe_load(f) or {}
                self.actions = data.get("actions", {})
            print(f"[YAML] Loaded {len(self.actions)} action(s) from {YAML_PATH}.")
        except Exception as e:
            print(f"[YAML] Warning: Could not load {YAML_PATH}: {e}")
            self.actions = {}

    def set_servo(self, name, angle):
        if name in SERVO_CHANNELS:
            # Clamp angle between 0 and 180
            angle = max(0, min(180, angle))
            self.kit.servo[SERVO_CHANNELS[name]].angle = angle
            print(f"  -> Set servo '{name}' (Channel {SERVO_CHANNELS[name]}) to {angle}°")
        else:
            print(f"[Error] Unknown servo name: '{name}'")

    def reset_to_center(self):
        for name, angle in CENTER_POSITIONS.items():
            self.set_servo(name, angle)

    def execute_action(self, action_id):
        action = self.actions.get(int(action_id))
        if not action:
            print(f"[Actions] Warning: Action ID {action_id} not found in YAML.")
            return

        print(f"[Actions] Executing Action ID {action_id}...")
        for step in action.get("sequence", []):
            angles = step.get("angles", {})
            duration = step.get("duration", 0.1)
            for servo_name, angle in angles.items():
                self.set_servo(servo_name, angle)
            time.sleep(duration)
        print("[Actions] Sequence finished. Centering body/arms/eyes...")
        self.reset_to_center()

    def run_interactive_loop(self):
        print("\n" + "="*40)
        print(" SERVO DEBUG CONTROLLER READY")
        print("="*40)
        print("Commands:")
        print("  <action_id>       : Run action from YAML (e.g., '1' or '2')")
        print("  set <servo> <ang> : Set specific servo (e.g., 'set left_arm 45')")
        print("  center            : Reset all servos to center")
        print("  list              : Show available action IDs")
        print("  quit / exit       : Exit program")
        print("="*40 + "\n")

        while True:
            try:
                user_input = input("Debug> ").strip()
                if not user_input:
                    continue
                
                parts = user_input.split()
                cmd = parts[0].lower()

                if cmd in ["quit", "exit"]:
                    print("Exiting...")
                    break
                
                elif cmd == "center":
                    self.reset_to_center()
                
                elif cmd == "list":
                    print(f"Available Action IDs: {list(self.actions.keys())}")
                
                elif cmd == "set" and len(parts) == 3:
                    servo_name = parts[1]
                    try:
                        angle = float(parts[2])
                        self.set_servo(servo_name, angle)
                    except ValueError:
                        print("[Error] Angle must be a valid number.")
                
                elif cmd.isdigit():
                    self.execute_action(int(cmd))
                
                else:
                    print(f"[Error] Unknown command or format: '{user_input}'")

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"[Error] {e}")

if __name__ == "__main__":
    controller = ServoDebugController()
    try:
        controller.run_interactive_loop()
    finally:
        print("Safely centering servos before exit...")
        controller.reset_to_center()