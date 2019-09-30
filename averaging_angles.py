from typing import List

import numpy as np

ANGLES = {"top": [0],
          "right": [90],
          "bottom": [180],
          "left": [270],
          "vertical": [0, 180],
          "horizontal": [90, 270]}


def get_cakes_to_average(averaging_type: str, num_cakes: int, starting_angle: int) -> List[int]:
    desired_angles = np.array(ANGLES[averaging_type])
    desired_cakes = list(((desired_angles - starting_angle) % 360) / 360 * num_cakes + 1)
    cakes_to_average = []
    for cake in desired_cakes:
        cakes_to_average.append(int(cake))
        cakes_to_average.append(int(cake + 1))
        cakes_to_average.append(int(cake - 1))
    return cakes_to_average
