#!/usr/bin/env python3
"""Interactive tool to extract template images from screenshots.

Usage:
    python scripts/extract_templates.py --input ocr/examples/Augur-NoShiftingEarth.jpeg --output templates/nightlords/Augur.png
    python scripts/extract_templates.py --input-dir ocr/examples --output-dir templates/nightlords --auto-detect nightlord
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# Nightlord mapping from filename to app ID
NIGHTLORD_MAPPING = {
    "Augur": "4_Maris",
    "GapingJaw": "2_Adel",
    "SentientPest": "3_Gnoster",
    "Tricephalos": "1_Gladius",
    "DarkdriftKnight": "6_Fulghor",
    "EquilibrousBeast": "5_Libra",
    "Balancers": "9_Balancers",
    "Dreglord": "10_Greg",
    "FissureInTheFog": "7_Caligo",
    "NightAspect": "8_Heolstor",
}


class TemplateExtractor:
    """Interactive template extraction tool."""

    def __init__(self):
        self.image = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi = None
        self.window_name = "Template Extractor - Draw ROI"

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

            # Calculate ROI
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            if x2 - x1 > 5 and y2 - y1 > 5:
                self.roi = (x1, y1, x2, y2)
                print(f"Selected ROI: {self.roi}")

    def extract_interactive(self, image_path: str, output_path: str) -> bool:
        """Interactively extract a template from an image.

        Args:
            image_path: Path to source image.
            output_path: Path to save extracted template.

        Returns:
            True if extraction successful.
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Error: Could not load image {image_path}")
            return False

        # Create a resized version for display if too large
        max_display_size = 1200
        h, w = self.image.shape[:2]
        scale = min(max_display_size / w, max_display_size / h, 1.0)

        if scale < 1.0:
            display_image = cv2.resize(self.image, None, fx=scale, fy=scale)
        else:
            display_image = self.image.copy()
            scale = 1.0

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\nInstructions:")
        print("  - Click and drag to select the template region")
        print("  - Press 's' to save the selection")
        print("  - Press 'r' to reset selection")
        print("  - Press 'q' or ESC to quit without saving\n")

        while True:
            # Draw current selection
            img_display = display_image.copy()

            if self.start_point and self.end_point:
                cv2.rectangle(
                    img_display,
                    self.start_point,
                    self.end_point,
                    (0, 255, 0),
                    2
                )

            cv2.imshow(self.window_name, img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and self.roi:
                # Save the selection
                x1, y1, x2, y2 = self.roi

                # Scale back to original coordinates
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

                template = self.image[y1:y2, x1:x2]

                # Ensure output directory exists
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(output_path, template)
                print(f"Saved template to: {output_path}")
                cv2.destroyAllWindows()
                return True

            elif key == ord('r'):
                # Reset selection
                self.roi = None
                self.start_point = None
                self.end_point = None
                print("Selection reset")

            elif key == ord('q') or key == 27:  # ESC
                print("Cancelled")
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
        return False

    def auto_extract_nightlord(
        self,
        image_path: str,
        output_dir: str
    ) -> bool:
        """Auto-extract nightlord icon from a screenshot.

        Uses the expected nightlord position to extract the icon.

        Args:
            image_path: Path to source screenshot.
            output_dir: Directory to save templates.

        Returns:
            True if extraction successful.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False

        # Get filename to determine nightlord name
        filename = Path(image_path).stem
        nightlord_name = filename.split("-")[0]

        if nightlord_name not in NIGHTLORD_MAPPING:
            print(f"Unknown nightlord in filename: {nightlord_name}")
            return False

        # Expected position in 1000x1000 coordinate system
        # Nightlord is at (111, 900) in the coordinate system
        nightlord_x = 111
        nightlord_y = 900

        # Estimate map region (right ~50% of screen)
        h, w = image.shape[:2]
        map_x_start = int(w * 0.45)
        map_region = image[:, map_x_start:]

        map_h, map_w = map_region.shape[:2]

        # Scale coordinates to map region
        scale_x = map_w / 1000
        scale_y = map_h / 1000

        center_x = int(nightlord_x * scale_x)
        center_y = int(nightlord_y * scale_y)

        # Extract region around the nightlord position
        icon_size = int(min(map_w, map_h) * 0.08)  # ~8% of map size
        half_size = icon_size // 2

        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(map_w, center_x + half_size)
        y2 = min(map_h, center_y + half_size)

        template = map_region[y1:y2, x1:x2]

        if template.size == 0:
            print(f"Error: Empty template region")
            return False

        # Save template
        output_path = Path(output_dir) / f"{nightlord_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), template)
        print(f"Extracted {nightlord_name} -> {output_path}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract template images from screenshots"
    )
    parser.add_argument(
        "--input", "-i",
        help="Input image path for single extraction"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output template path for single extraction"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory for batch extraction"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for batch extraction"
    )
    parser.add_argument(
        "--auto-detect",
        choices=["nightlord", "spawn"],
        help="Auto-detect mode for batch extraction"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive mode for extraction"
    )

    args = parser.parse_args()

    extractor = TemplateExtractor()

    if args.input and args.output:
        # Single file extraction
        if args.interactive:
            extractor.extract_interactive(args.input, args.output)
        else:
            print("Use --interactive for manual selection")
            return

    elif args.input_dir and args.output_dir:
        # Batch extraction
        input_dir = Path(args.input_dir)

        if not input_dir.exists():
            print(f"Error: Input directory does not exist: {input_dir}")
            return

        # Find all images
        image_files = list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return

        print(f"Found {len(image_files)} images")

        for image_path in image_files:
            if args.auto_detect == "nightlord":
                extractor.auto_extract_nightlord(str(image_path), args.output_dir)
            elif args.interactive:
                output_name = image_path.stem + ".png"
                output_path = Path(args.output_dir) / output_name
                extractor.extract_interactive(str(image_path), str(output_path))
            else:
                print(f"Skipping {image_path} (no auto-detect mode specified)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
