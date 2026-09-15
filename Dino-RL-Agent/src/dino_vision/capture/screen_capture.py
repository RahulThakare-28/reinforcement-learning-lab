# dino_vision/capture/screen_capture.py

import tkinter as tk
import pyautogui
import numpy as np



class ScreenCapture:
    """
    Select and continuously capture a rectangular region of the screen.
    """

    def __init__(self):
        self.region = None
        self.running = False

    def select_region(self):
        """
        Open a movable/resizable selection window.

        Move the window over the Dino game,
        resize it as required,
        then press Enter to confirm.
        """

        root = tk.Tk()
        root.title("Dino Vision - Select Game Region")

        # Small initial window
        root.geometry("800x300+300+200")

        # Make the window visually identifiable
        root.attributes("-alpha", 0.3)
        root.attributes("-topmost", True)

        canvas = tk.Canvas(
            root,
            bg="black",
            highlightthickness=3,
            highlightbackground="red"
        )
        canvas.pack(fill="both", expand=True)

        confirmed = False

        def confirm(event=None):
            nonlocal confirmed

            x = root.winfo_rootx()
            y = root.winfo_rooty()
            width = root.winfo_width()
            height = root.winfo_height()

            self.region = {
                "left": x,
                "top": y,
                "width": width,
                "height": height,
            }

            confirmed = True
            root.destroy()

        def cancel(event=None):
            root.destroy()

        root.bind("<Return>", confirm)
        root.bind("<Escape>", cancel)

        root.mainloop()

        return self.region if confirmed else None

    def capture(self):
        """
        Capture the currently selected region.

        Returns:
            numpy.ndarray: RGB image
        """

        if self.region is None:
            raise RuntimeError(
                "No capture region selected. "
                "Call select_region() first."
            )

        r = self.region

        screenshot = pyautogui.screenshot(
            region=(
                r["left"],
                r["top"],
                r["width"],
                r["height"],
            )
        )

        return np.array(screenshot)

    def start(self):
        """
        Start continuous screen capture.

        Yields:
            numpy.ndarray: RGB frame
        """

        if self.region is None:
            raise RuntimeError(
                "No capture region selected. "
                "Call select_region() first."
            )

        self.running = True

        while self.running:
            yield self.capture()

    def stop(self):
        """Stop continuous capture."""
        self.running = False

