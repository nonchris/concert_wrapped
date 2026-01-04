import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def moderate_color(hex_color: str, saturation_factor=1, brightness_factor=0.5):
    """
    Make a color less extreme by:
    - Reducing saturation (always)
    - Brightening if too dark, darkening if too bright

    Parameters:
    - hex_color: hex string
    - saturation_factor: how much saturation to keep (0-1, lower = more gray)
    - brightness_factor: how much to push toward middle brightness (0-1)
                        0 = move fully to middle gray
                        1 = keep original brightness
    """
    rgb = mcolors.to_rgb(hex_color)
    hsv = mcolors.rgb_to_hsv(rgb)

    # Always reduce saturation
    hsv[1] *= saturation_factor

    # Push brightness toward middle (0.5)
    middle = 0.5
    hsv[2] = hsv[2] + (middle - hsv[2]) * (1 - brightness_factor)

    rgb_new = mcolors.hsv_to_rgb(hsv)
    return rgb_new


if __name__ == "__main__":
    # Example usage
    original = "#FF5733"
    toned_down = desaturate_color(original, factor=0.5)  # 50% saturation

    # Use in a plot
    plt.plot([1, 2, 3], [1, 2, 3], color=toned_down, linewidth=3)
    plt.show()
