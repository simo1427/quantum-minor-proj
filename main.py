from image_effects import *
from timing_curves import *


if __name__ == "__main__":
    animate_images(
        'media/ocean.png',
        'media/grass.png',
        partial_swap,
        frames=60,
        timing_curve=linear,
        use_statevector=True,
        animate_back=True,
        alpha=1.0,
        i=0,
        j=1,
        on_registers=True
    )
