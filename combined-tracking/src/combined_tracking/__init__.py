"""Track the robot's AprilTag and the ball together. Not skill 1 or 2 -- an
integration of both.

Unlike tag-tracking and ball-tracking, which only depend on vision-core, this
one depends on *both* of them and reuses their detection/filter/drawing code
directly (TagTracker, BallTracker, draw_tag_marks, draw_ball_marks, ...)
rather than reimplementing any of it. tag-tracking and ball-tracking still do
not depend on each other -- only this integration sits above both.

track_combined.py opens the camera once and runs both detectors against the
same frame, so `uv run track` and `uv run track-ball` keep working exactly as
they always have, independently, for whichever one thing you actually need.
"""
