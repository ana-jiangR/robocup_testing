"""Shared vision floor for every robotics skill in this repo. Not a skill.

    field.py      the swap point: Field and CameraFieldTransform
    camera.py     opening a camera, locking exposure / white balance
    intrinsics.py load, save and guess the camera matrix
    charuco.py    *measure* the camera matrix, from a ChArUco board
    paths.py      repo_root(), so calib/ resolves whatever the cwd is
    planview.py   drawing the field: video overlay and top-down panel

A skill depends on vision-core. vision-core never depends on a skill.
Import from the submodules; there is no re-exported surface to keep in sync.
"""
