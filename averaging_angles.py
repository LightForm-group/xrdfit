from typing import List, Union

import numpy as np

ANGLES = {"top": [0],
          "right": [90],
          "bottom": [180],
          "left": [270],
          "vertical": [0, 180],
          "horizontal": [90, 270]}


def get_cakes_to_average(averaging_type: str, total_cakes: int, starting_angle: int,
                         num_cakes_to_average: int) -> Union[None, List[int]]:
    if averaging_type not in ANGLES:
        print("Data not loaded. {} is an unknown averaging type. "
              "Use one of: {}".format(averaging_type, ANGLES.keys()))
        return None

    desired_angles = np.array(ANGLES[averaging_type])
    desired_cakes = list(((desired_angles - starting_angle) % 360) / 360 * total_cakes + 1)
    cakes_to_average = []
    for cake_center in desired_cakes:
        for cake_number in range(num_cakes_to_average):
            cake_to_add = int(cake_center - int(num_cakes_to_average/2) + cake_number)
            cake_to_add = cake_to_add % total_cakes
            cakes_to_average.append(cake_to_add)
    return cakes_to_average
