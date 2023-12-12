import numpy as np


def slidingWindow(image_size, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                  x_range=(0, 1), y_range=(0, 1), scale=1.5):
    windows = []
    h, w = image_size[1], image_size[0]
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * h)):
        win_width = int(init_size[0] + (scale * (y - (y_range[0] * h))))
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        if y + win_height > int(y_range[1] * h) or win_width > w:
            break
        x_step = int((1 - x_overlap) * win_width)
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            windows.append((x, y, x + win_width, y + win_height))

    return windows


def myslidingWindow(image_size, init_size=(64, 64), x_overlap=0.5, y_step=0.05,
                    x_range=(0, 1), y_range=(0, 1), scale=1.5):
    windows = []
    h, w = image_size[1], image_size[0]
    # win_width = int(init_size[0])
    # win_height = int(init_size[1])
    for y in range(int(y_range[0] * h), int(y_range[1] * h), int(y_step * init_size[1])):
        win_height = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        win_width = int(init_size[1] + (scale * (y - (y_range[0] * h))))
        if y + win_height > int(y_range[1] * h) or win_height > h:
            break
        x_step = int((x_overlap) * init_size[0])
        for x in range(int(x_range[0] * w), int(x_range[1] * w), x_step):
            # win_width = int(init_size[0] + (scale * (x - (x_range[0] * w))))
            if x+win_width > int(x_range[1] * w) or win_width > w:
                break
            windows.append((x, y, x + win_width, y + win_height))
    return windows
