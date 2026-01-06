FOCAL_LENGTH = 700
KNOWN_WIDTH = 0.5  # meters

def estimate_distance(bbox_width):
    return (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
