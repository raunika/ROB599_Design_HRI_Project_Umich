
import time
import inspect

from pyfirmata import Arduino, SERVO, util
# pip install pyfirmata if not found

# ============================================================
# ARDUINO CONFIGURATION
# ============================================================

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

current_angles = HOME_ANGLES.copy()

SMOOTH_STEPS = 35


# ============================================================
# GESTURE DEFINITIONS
# ============================================================

GESTURES = {
    1: "Blink",
    2: "Nod",
    3: "Head shake",
    4: "Right arm raise",
    5: "Left arm raise",
    6: "Both arms up",
    7: "Delayed two-arm raise",
}


# Each gesture contains sequential steps.
# Each step: (servo angles to update, duration in seconds)

ARDUINO_GESTURES = {

    1: [  # Blink
        ({"left_eyelid": 135, "right_eyelid": 35}, 0.8),
        ({"left_eyelid": 90, "right_eyelid": 90}, 0.8),
	({"left_eyelid": 135, "right_eyelid": 35}, 0.8),
        ({"left_eyelid": 90, "right_eyelid": 90}, 0.8),
    ],

    2: [  # Nod
        ({"head_tilt": 110}, 0.3),
        ({"head_tilt": 90}, 0.3),
        ({"head_tilt": 110}, 0.3),
        ({"head_tilt": 90}, 0.3),
    ],

    3: [  # Head shake
        ({"head_pan": 70}, 0.3),
        ({"head_pan": 110}, 0.3),
        ({"head_pan": 70}, 0.3),
        ({"head_pan": 90}, 0.3),
    ],

    4: [  # Right arm raise
        ({"right_arm": 170}, 0.3),
        ({"right_arm": 75}, 0.3),
    ],

    5: [  # Left arm raise
        ({"left_arm": 20}, 0.3),
        ({"left_arm": 95}, 0.3),
    ],

    6: [  # Both arms up
        ({"left_arm": 20, "right_arm": 170}, 0.4),
        ({"left_arm": 95, "right_arm": 75}, 0.4),
    ],

    7: [  # Delayed two-arm raise
        ({"left_arm": 20}, 0.3),
        ({"right_arm": 170}, 0.3),
        ({"left_arm": 95, "right_arm": 75}, 0.3),
    ],
}


# ============================================================
# SMOOTH SERVO MOVEMENT
# ============================================================

def smooth_move_servos(servos, targets, duration,
                       steps=SMOOTH_STEPS):

    """
    Smoothly interpolate servo positions using
    smoothstep interpolation.

    duration: Total movement duration in seconds.
    steps: Number of intermediate positions.
    """

    start_angles = {
        name: current_angles[name]
        for name in targets
    }

    delay = duration / steps

    for step in range(1, steps + 1):

        t = step / steps

        # Smooth acceleration and deceleration
        progress = 3 * t**2 - 2 * t**3

        for name, target in targets.items():

            start = start_angles[name]

            angle = start + (target - start) * progress

            servos[name].write(int(round(angle)))

        time.sleep(delay)

    # Update commanded servo positions
    for name, target in targets.items():
        current_angles[name] = target


# ============================================================
# ARDUINO CONNECTION
# ============================================================

def connect_arduino():

    # Python 3.12 compatibility for PyFirmata
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    print("\nConnecting to MOMO's Arduino...")

    board = Arduino(ARDUINO_PORT)

    iterator = util.Iterator(board)
    iterator.start()

    time.sleep(1)

    servos = {}

    # Configure servo pins
    for name, pin in SERVO_PINS.items():

        board.digital[pin].mode = SERVO

        servos[name] = board.digital[pin]

    # Initialize home positions
    for name, angle in HOME_ANGLES.items():

        servos[name].write(angle)

        current_angles[name] = angle

    time.sleep(0.5)

    print("Arduino connected.")
    print("All servos initialized to home positions.")

    return board, servos, iterator


# ============================================================
# GESTURE EXECUTION
# ============================================================

def execute_gesture(servos, gesture_id):

    if gesture_id not in ARDUINO_GESTURES:
        print(f"Unknown gesture: {gesture_id}")
        return

    print(
        f"\n[MOMO GESTURE] "
        f"{gesture_id}: {GESTURES[gesture_id]}"
    )

    gesture_steps = ARDUINO_GESTURES[gesture_id]

    for angles, duration in gesture_steps:

        smooth_move_servos(
            servos=servos,
            targets=angles,
            duration=duration
        )


def execute_all_gestures(servos):

    print("\n================================")
    print("       MOMO GESTURE DEMO")
    print("================================")

    for gesture_id in ARDUINO_GESTURES:

        execute_gesture(servos, gesture_id)

        # Small pause between gestures
        time.sleep(0.5)


# ============================================================
# RETURN TO HOME POSITION
# ============================================================

def return_home(servos):

    print("\nReturning MOMO to home position...")

    smooth_move_servos(
        servos=servos,
        targets=HOME_ANGLES,
        duration=0.6
    )

    print("MOMO is back at home position.")


# ============================================================
# CLEANUP
# ============================================================

def close_arduino(board, servos):

    try:

        return_home(servos)

        time.sleep(0.5)

    finally:

        board.exit()

        print("\nArduino connection closed.")


# ============================================================
# MAIN
# ============================================================

def main():

    board = None
    servos = None

    try:

        board, servos, iterator = connect_arduino()

        print("\nStarting gesture sequence in 2 seconds...")

        time.sleep(2)

        # Run through every gesture once
        execute_all_gestures(servos)

        print("\nAll gestures completed!")

    except KeyboardInterrupt:

        print("\nGesture sequence interrupted by user.")

    except Exception as e:

        print(f"\nError: {e}")

    finally:

        if board is not None and servos is not None:

            close_arduino(board, servos)


if __name__ == "__main__":
    main()
