def get_direction(x_center, frame_width):
    if x_center < frame_width / 3:
        return "Left"
    elif x_center > 2 * frame_width / 3:
        return "Right"
    return "Center"
