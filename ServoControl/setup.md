
---

# Raspberry Pi Setup & Configuration Guide

This guide covers setting up your robot controller script to run automatically on boot and configuring your Raspberry Pi to withstand unexpected power loss.

---

## 1. Raspberry Pi Systemd Configuration (Run on Startup)

To configure the script to start automatically when the Raspberry Pi boots up, follow these steps:

1. Create a systemd service file at `/etc/systemd/system/robot.service`:
```bash
sudo nano /etc/systemd/system/robot.service

```


2. Paste the following configuration (adjust paths and user name if your user isn't `pi`):
```ini
[Unit]
Description=Robot Servo Controller & Tracking Daemon
After=network.target local-fs.target i2c.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ServoControl
ExecStart=/usr/bin/python3 /home/pi/ServoControl/servo_controller.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

```


3. Save and close the file (`Ctrl+O`, `Enter`, then `Ctrl+X` in nano).
4. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable robot.service
sudo systemctl start robot.service

```



---

## 2. Handling Unexpected Power Loss & SD Card Protection

Because sudden power loss (unplugging the Pi abruptly) can corrupt SD cards, follow these best practices on your Raspberry Pi:

* **Enable I2C Interface:** Ensure I2C is enabled so the PCA9685 can be reached on boot:
```bash
sudo raspi-config

```


Navigate to *Interface Options* -> *I2C* and enable it.
* **Read-Only Root Filesystem (Recommended):** To completely prevent SD card corruption from hard unplugs, configure Raspberry Pi OS to run in **OverlayFS (Read-Only Root)** mode:
```bash
sudo raspi-config

```


Navigate to *Performance Options* -> *Overlay File System* and enable it.
*(Note: If you enable read-only mode, any logs or local files modified at runtime will not persist across reboots unless written to a separate tmpfs or data partition).*
* **Graceful Servo Parking:** The controller script includes exception handling (`KeyboardInterrupt` / safe shutdown exits) to set servos back to neutral when terminated properly via system controls.