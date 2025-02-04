import numpy as np
import pytest

NAngle = 24
all_angles = np.arange(1, NAngle+1)

@pytest.mark.parametrize("ANGLE", all_angles)
@pytest.mark.parametrize("ANGLE_WALL", all_angles)
def test_reflect_angle_id(ANGLE, ANGLE_WALL):
    """
    """

    from pytideR.algos import reflect_angle_id

    angle_reflected = reflect_angle_id(ANGLE, ANGLE_WALL)
    assert isinstance(angle_reflected, int)
