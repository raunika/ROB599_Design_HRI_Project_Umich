
import time
import inspect

# Python 3.12 compatibility
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from pyfirmata import Arduino, SERVO, util


# ==========================================
# CONFIGURATION
# ==========================================

ARDUINO_PORT = "/dev/cu.usbmodem2101"

SERVO_PINS = {
    "head_pan": 5,
    "head_tilt": 3,
    "left_arm": 10,
    "right_arm": 9,
    "left_eyelid": 6,
    "right_eyelid": 11,
}

HOME_ANGLES = {
    "head_pan": 90,
    "head_tilt": 90,
    "left_arm": 95,
    "right_arm": 75,
    "left_eyelid": 90,
    "right_eyelid": 90,
}

# Absolute physical angle limits
SERVO_LIMITS = {
    "head_pan": (70, 110),
    "head_tilt": (70, 110),
    "left_arm": (70, 120),
    "right_arm": (50, 110),
    "left_eyelid": (70, 110),
    "right_eyelid": (70, 110),
}

# Track current positions
current_angles = {}


# ==========================================
# CONNECT TO ARDUINO
# ==========================================

print("\nConnecting to Arduino...")

board = Arduino(ARDUINO_PORT)
iterator = util.Iterator(board)
iterator.start()

time.sleep(1)

servos = {}

for name, pin in SERVO_PINS.items():
    board.digital[pin].mode = SERVO
    servos[name] = board.digital[pin]

time.sleep(0.5)


# ==========================================
# INITIALIZE HOME POSITIONS
# ==========================================

print("\nMoving all servos to their home positions...")

# Avoid sending conflicting commands to a shared pin
for name, angle in HOME_ANGLES.items():
    pin = SERVO_PINS[name]

    if not any(
        SERVO_PINS[other] == pin
        and list(SERVO_PINS).index(other) < list(SERVO_PINS).index(name)
        for other in SERVO_PINS
    ):
        servos[name].write(angle)

    current_angles[name] = angle

time.sleep(1)

print("All servos initialized.")


# ==========================================
# SERVO MOVEMENT
# ==========================================

def move_servo(name, offset):

    home = HOME_ANGLES[name]
    target = home + offset

    minimum, maximum = SERVO_LIMITS[name]

    if not minimum <= target <= maximum:
        print(
            f"Movement rejected. Target: {target} degrees. "
            f"Allowed range: {minimum}-{maximum}."
        )
        return

    pin = SERVO_PINS[name]

    # Shared pins cannot independently control two servos
    other_servos = [
        other for other in SERVO_PINS
        if SERVO_PINS[other] == pin and other != name
    ]

    if other_servos:
        print(
            f"WARNING: {name} shares pin D{pin} with "
            f"{', '.join(other_servos)}."
        )
        print("These servos cannot be independently controlled.")
        return

    servos[name].write(target)
    current_angles[name] = target

    print(
        f"{name}: Home {home}° | "
        f"Offset {offset:+}° | Target {target}°"
    )


# ==========================================
# CALIBRATION INTERFACE
# ==========================================

def main():

    names = list(SERVO_PINS.keys())
    selected = None

    print("\n========== MOMO SERVO CALIBRATION ==========")

    print("\nAvailable servos:")

    for i, name in enumerate(names, start=1):
        print(
            f"{i}. {name:15} | Pin D{SERVO_PINS[name]} | "
            f"Home: {HOME_ANGLES[name]}°"
        )

    print("\nCommands:")
    print("  1-6  = Select servo")
    print("  +10  = Move 10 degrees above home")
    print("  -10  = Move 10 degrees below home")
    print("  0    = Return to home")
    print("  q    = Quit")

    try:
        while True:

            command = input("\nCommand: ").strip().lower()

            if command == "q":
                break

            if command.isdigit() and command in [
                str(i) for i in range(1, len(names) + 1)
            ]:
                selected = names[int(command) - 1]
                print(
                    f"\nSelected: {selected} "
                    f"(Home: {HOME_ANGLES[selected]}°)"
                )
                continue

            if selected is None:
                print("Select a servo first.")
                continue

            try:
                offset = int(command)
                move_servo(selected, offset)

            except ValueError:
                print("Enter a servo number, an offset (+10/-10), 0, or q.")

    except KeyboardInterrupt:
        print("\nCalibration interrupted.")

    finally:
        print("\nReturning servos to home positions...")

        # Return each unique physical pin to its home position.
        handled_pins = set()

        for name, angle in HOME_ANGLES.items():
            pin = SERVO_PINS[name]

            if pin not in handled_pins:
                servos[name].write(angle)
                handled_pins.add(pin)

        time.sleep(1)
        board.exit()

        print("Arduino disconnected.")


if __name__ == "__main__":
    main()
