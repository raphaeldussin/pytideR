import numpy as np


def reflect_angle_id(angle_incident, angle_wall, NAngle=24):
    """ compute reflected angle """
    TwoPi = 8.0*np.arctan(1.0)
    Angle_size = TwoPi / (float(NAngle))
    NAngle_d2 = (NAngle / 2)

    angle_relatif = np.mod(angle_incident -angle_wall + NAngle, NAngle)
    if (0 < np.mod(angle_incident - angle_wall + NAngle, NAngle) < NAngle_d2):
        a_r = 2 * angle_wall - angle_incident
        a_r = np.mod(a_r + NAngle, NAngle)
    else:
        a_r = None
    return a_r


