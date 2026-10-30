import serial
import time

"""
    A wrapper class which allows python to talk to the arduino board. 
    The class currently assumes a usb connection exists between this computer and the board.
"""
class StewartPlatform():
    def __init__(self, port, baudrate=115200):
        self.serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1,
            write_timeout=1,
        )

        # Allow the Arduino to reset and initialize.
        time.sleep(2)
        self.serial.reset_input_buffer()
        self.reset()

    """
        given a string to send to the arduino, send it. does this work for multi-character strings?
        this function should also be a halting operation which waits until the arduino sends back "OK".
    """
    def send_command(self, command):
        # Send one character supported by handleCommand().
        if not isinstance(command, str):
            raise ValueError("command must be a string")

        if not command:
            raise ValueError("command cannot be empty")
        
        if not self.serial.is_open:
            raise serial.SerialException("Serial connection is closed")
        
        message = (command + "\n").encode("ascii")
        self.serial.write(message)
        self.serial.flush()

        # get the response. this halts the code until the arduino confirms the platform has moved.
        response = self.serial.readline()
        response = response.decode("utf-8", errors="replace").strip()
        if response.startswith("ERR"):
            raise RuntimeError(
                f"Arduino rejected command {command!r}: {response}"
            )
        return response

    def send_motion_command(self, command, poll_interval=0.05):
        """
        Send a motion command and wait until the platform finishes moving.
        """
        response = self.send_command(command)

        if response != "OK":
            return response

        # Wait until the Arduino reports that motion is complete.
        while True:
            status = self.get_status()

            # Extract the MOTION field from the status response.
            fields = dict(
                item.split("=", 1)
                for item in status.split()
                if "=" in item
            )

            motion = fields.get("MOTION")

            if motion is None:
                raise RuntimeError(f"Could not read MOTION from status: {status}")

            if motion == "IDLE":
                return response

            time.sleep(poll_interval)

    """
        resets the platform back to its original location
    """
    def reset(self):
        return self.send_motion_command("h")

    """
        returns a string representing the current status of the platform as recieved by the platform
    """
    def get_status(self):
        return self.send_command("STATUS")

    """
        closes the connection to the platform
    """
    def close(self):
        self.send_command("l") # turn off the laser before disconnecting
        self.reset()
        if self.serial.is_open:
            self.serial.close()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
