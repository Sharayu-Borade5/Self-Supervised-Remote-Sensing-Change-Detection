import cv2
import numpy as np

def visualize_change(before, after, score):
    overlay = cv2.addWeighted(before, 0.5, after, 0.5, 0)
    cv2.putText(
        overlay,
        f"Change Score: {score:.3f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    return overlay
