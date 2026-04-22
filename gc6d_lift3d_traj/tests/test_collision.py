import numpy as np

from gc6d_lift3d_traj.planner.collision import CollisionConfig, check_table_collision


def test_table_collision():
    cfg = CollisionConfig(table_z=0.0, table_tolerance=0.001)
    pos_ok = np.array([[0, 0, 0.1], [0, 0, 0.2]], dtype=np.float32)
    pos_bad = np.array([[0, 0, -0.01], [0, 0, 0.2]], dtype=np.float32)
    assert not check_table_collision(pos_ok, cfg)
    assert check_table_collision(pos_bad, cfg)

