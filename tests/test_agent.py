import unittest

from agent import BSRState, DynamicBSRAgent


class DynamicBSRAgentTest(unittest.TestCase):
    def test_recommendation_decreases_when_urgency_increases(self) -> None:
        agent = DynamicBSRAgent()
        calm_state = BSRState(1_000, 8.0, 0.2, 0.40)
        urgent_state = BSRState(70_000, 35.0, 1.1, 0.50)

        calm_periodicity = agent.recommend_periodicity(calm_state)
        urgent_periodicity = agent.recommend_periodicity(urgent_state)

        self.assertGreater(calm_periodicity, urgent_periodicity)

    def test_recommendation_is_clamped(self) -> None:
        agent = DynamicBSRAgent(min_periodicity_ms=5, max_periodicity_ms=80)
        extreme_state = BSRState(200_000, 120.0, 10.0, 0.20)

        recommendation = agent.recommend_periodicity(extreme_state)

        self.assertGreaterEqual(recommendation, 5)
        self.assertLessEqual(recommendation, 80)


if __name__ == "__main__":
    unittest.main()
