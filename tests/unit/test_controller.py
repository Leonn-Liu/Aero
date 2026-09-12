from aero.decoding.controller import AdaptiveGammaController

def test_controller_dynamics():
    controller = AdaptiveGammaController(
        initial_gamma=4,
        min_gamma=1,
        max_gamma=8,
        window_size=5,
        high_threshold=0.75,
        low_threshold=0.40
    )
    assert controller.current_gamma == 4

    for _ in range(5):
        gamma = controller.update(accepted_count=1, draft_count=4)
    assert controller.current_gamma == 3

    for _ in range(10):
        gamma = controller.update(accepted_count=4, draft_count=4)
    assert controller.current_gamma == 5

    for _ in range(25):
        controller.update(accepted_count=0, draft_count=8)
    assert controller.current_gamma == 1

    for _ in range(40):
        controller.update(accepted_count=8, draft_count=8)
    assert controller.current_gamma == 8

    controller.reset()
    assert controller.current_gamma == 4
    assert len(controller.history) == 0

if __name__ == "__main__":
    test_controller_dynamics()
    print("test_controller PASSED")
