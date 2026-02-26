#!/usr/bin/env python3
"""
Generate a ChArUco board PDF for printing
"""

import argparse
import cv2
import numpy as np
try:
    import cv2.aruco as aruco
except (ImportError, AttributeError):
    from cv2 import aruco

PAPER_SIZES = {
    "LETTER": (215.9, 279.4),  # 8.5 x 11 inches in mm
    "A4":     (210.0, 297.0),  # mm
}

def generate_charuco_board(paper_size="LETTER"):
    """Generate and save ChArUco board as PDF and PNG"""

    # Board parameters - adjust these to match your needs
    SQUARES_X = 14  # Number of chessboard squares in X direction
    SQUARES_Y = 9   # Number of chessboard squares in Y direction
    SQUARE_LENGTH_MM = 18  # Size of each square in mm
    MARKER_LENGTH_MM = 13  # Size of ArUco markers in mm

    # Paper dimensions in mm
    paper_size = paper_size.upper()
    if paper_size not in PAPER_SIZES:
        raise ValueError(f"Unknown paper size '{paper_size}'. Choose from: {list(PAPER_SIZES.keys())}")
    PAPER_WIDTH_MM, PAPER_HEIGHT_MM = PAPER_SIZES[paper_size]

    # DPI for high quality printing
    DPI = 300

    # Convert mm to inches (for DPI calculation)
    MM_TO_INCH = 1.0 / 25.4  # 1 inch = 25.4 mm

    # Calculate exact pixel sizes based on physical dimensions
    # This ensures 1:1 scale when printed
    SQUARE_LENGTH_PX = int(SQUARE_LENGTH_MM * MM_TO_INCH * DPI)
    MARKER_LENGTH_PX = int(MARKER_LENGTH_MM * MM_TO_INCH * DPI)

    print(f"Generating ChArUco board (paper: {paper_size}):")
    print(f"  Grid: {SQUARES_X}x{SQUARES_Y} squares")
    print(f"  Square size: {SQUARE_LENGTH_MM}mm ({SQUARE_LENGTH_PX}px at {DPI} DPI)")
    print(f"  Marker size: {MARKER_LENGTH_MM}mm ({MARKER_LENGTH_PX}px at {DPI} DPI)")

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    print(f"  Dictionary: DICT_5X5_250")

    # Create ChArUco board
    board = aruco.CharucoBoard_create(
        squaresX=SQUARES_X,
        squaresY=SQUARES_Y,
        squareLength=float(SQUARE_LENGTH_PX),
        markerLength=float(MARKER_LENGTH_PX),
        dictionary=dictionary
    )

    # Calculate board size
    board_width_px = SQUARES_X * SQUARE_LENGTH_PX
    board_height_px = SQUARES_Y * SQUARE_LENGTH_PX

    # Add margins (20mm on each side)
    MARGIN_MM = 20
    MARGIN_PX = int(MARGIN_MM * MM_TO_INCH * DPI)

    # Total image size with margins
    img_width = board_width_px + 2 * MARGIN_PX
    img_height = board_height_px + 2 * MARGIN_PX

    print(f"  Board size: {board_width_px}x{board_height_px}px")
    print(f"  Image size with margins: {img_width}x{img_height}px")

    # Generate the board image
    board_img = board.draw((board_width_px, board_height_px))

    # Convert to 3-channel if grayscale
    if len(board_img.shape) == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    # Create white background with margins
    final_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # Place board in center
    y_offset = MARGIN_PX
    x_offset = MARGIN_PX
    final_img[y_offset:y_offset+board_height_px, x_offset:x_offset+board_width_px] = board_img

    # Add text information at the top
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3

    # Title
    text = "ChArUco Board - DICT_5X5_250"
    cv2.putText(final_img, text, (MARGIN_PX, MARGIN_PX//2),
                font, font_scale, (0, 0, 0), font_thickness)

    # Info text at bottom
    info_lines = [
        f"Grid: {SQUARES_X}x{SQUARES_Y} | Square: {SQUARE_LENGTH_MM}mm | Marker: {MARKER_LENGTH_MM}mm",
        f"Dictionary: DICT_5X5_250 | Print at 100% scale"
    ]

    y_pos = img_height - MARGIN_PX//3
    for line in reversed(info_lines):
        text_size = cv2.getTextSize(line, font, font_scale*0.5, font_thickness//2)[0]
        x_pos = (img_width - text_size[0]) // 2
        cv2.putText(final_img, line, (x_pos, y_pos),
                   font, font_scale*0.5, (0, 0, 0), font_thickness//2)
        y_pos -= int(MARGIN_PX * 0.4)

    # Convert to PDF using PIL if available
    try:
        from PIL import Image

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        # Create PIL image
        pil_img = Image.fromarray(img_rgb)

        # Save as PDF
        output_pdf = "charuco_board_5x5_250.pdf"

        # Set DPI info for correct printing size
        pil_img.save(output_pdf, "PDF", resolution=DPI, save_all=True)
        print(f"Saved as: {output_pdf}")
        print(f"\nIMPORTANT: Print at 100% scale (no fit to page)")

    except ImportError:
        print("\nNote: Install Pillow to generate PDF: pip install Pillow")
        print("PNG file saved successfully. You can print it at 100% scale.")

    # Also save a smaller preview version
    preview_scale = 0.3
    preview_img = cv2.resize(final_img, None, fx=preview_scale, fy=preview_scale)
    cv2.imwrite("charuco_board_5x5_250_preview.png", preview_img)
    print("Preview saved as: charuco_board_5x5_250_preview.png")

    # Display board info
    print(f"\nBoard Information:")
    print(f"  Dictionary: DICT_5X5_250")
    print(f"  Total markers in board: {(SQUARES_X-1)*(SQUARES_Y-1)//2}")
    print(f"  Board physical size: {SQUARES_X*SQUARE_LENGTH_MM}mm x {SQUARES_Y*SQUARE_LENGTH_MM}mm")
    print(f"  Recommended viewing distance: 30-100cm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a ChArUco board PDF for printing")
    parser.add_argument(
        "--paper-size",
        choices=list(PAPER_SIZES.keys()),
        default="LETTER",
        help="Paper size for PDF output (default: LETTER)",
    )
    args = parser.parse_args()
    generate_charuco_board(paper_size=args.paper_size)
    print("\nDone!")