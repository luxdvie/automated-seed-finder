#!/usr/bin/env python3
"""Display monitor numbers on each screen for identification."""

import cv2
import numpy as np
from mss import mss


def create_monitor_image(monitor_index: int, width: int, height: int) -> np.ndarray:
    """Create an image with a big monitor number."""
    # Create black image (smaller than full screen)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw the monitor number
    text = str(monitor_index)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(width, height) / 200
    thickness = max(8, int(font_scale * 5))

    # Get text size for centering
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Center the text
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    # Draw white text
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    # Draw resolution subtitle
    subtitle = f"{width} x {height}"
    sub_scale = font_scale / 3
    sub_thickness = max(2, int(sub_scale * 3))
    (sub_w, sub_h), _ = cv2.getTextSize(subtitle, font, sub_scale, sub_thickness)
    sub_x = (width - sub_w) // 2
    sub_y = y + text_height // 2 + 30
    cv2.putText(img, subtitle, (sub_x, sub_y), font, sub_scale, (128, 128, 128), sub_thickness)

    return img


def main():
    with mss() as sct:
        monitors = sct.monitors

        print(f"Found {len(monitors) - 1} monitor(s):")
        for i, mon in enumerate(monitors):
            if i == 0:
                continue  # Skip the "all monitors" entry
            print(f"  Monitor {i}: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})")

        print("\nDisplaying monitor numbers for 3 seconds...")
        print("Press any key to close early.\n")

        # Window size (not fullscreen)
        win_width = 600
        win_height = 400

        # Create windows for each monitor
        windows = []
        for i, mon in enumerate(monitors):
            if i == 0:
                continue

            window_name = f"Monitor {i}"

            # Create image at window size
            img = create_monitor_image(i, win_width, win_height)

            # Calculate center position on this monitor
            center_x = mon["left"] + (mon["width"] - win_width) // 2
            center_y = mon["top"] + (mon["height"] - win_height) // 2

            # Create and position window
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, img)
            cv2.moveWindow(window_name, center_x, center_y)
            windows.append(window_name)

        # Wait for 3 seconds or keypress
        cv2.waitKey(3000)

        # Clean up
        cv2.destroyAllWindows()

        print("Done!")


if __name__ == "__main__":
    main()
