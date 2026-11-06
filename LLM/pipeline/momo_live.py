import subprocess
import signal
import sys

from momo_pipeline import run_pipeline, TTS_OUTPUT

MIC = "bluez_source.58_18_62_20_F9_D3.headset_head_unit"
RECORDING = "/nvmeroot/momo/test_live.wav"


def main():
    print("\n========== MOMO LIVE ==========")
    print("Press ENTER to start listening.")
    print("Speak, then press ENTER to finish.")
    print("Type q and press ENTER to quit.\n")

    while True:
        command = input("[MOMO] Ready? ")

        if command.lower() == "q":
            print("[MOMO] Goodbye!")
            break

        print("\n[MOMO] Listening... Speak now.")

        recorder = subprocess.Popen([
            "parecord",
            "--device=" + MIC,
            "--file-format=wav",
            "--rate=16000",
            "--channels=1",
            RECORDING
        ])

        input("[MOMO] Press ENTER when you finish speaking...")

        recorder.send_signal(signal.SIGINT)
        recorder.wait()

        try:
            run_pipeline(RECORDING)

            print("\n[MOMO] Speaking...")
            subprocess.run(["paplay", TTS_OUTPUT], check=True)

        except Exception as error:
            print("\n[MOMO] Error:", error)

        print("\n[MOMO] Ready for the next interaction.\n")


if __name__ == "__main__":
    main()
