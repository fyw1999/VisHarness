from visharness.agent_loop.trajectory_state import VisionTrajectoryState


def test_initial_image_is_named_img_0():
    image = object()

    state = VisionTrajectoryState.from_initial_image(image)

    assert state.images["img_0"]["image"] is image
