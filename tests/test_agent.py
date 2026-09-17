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

    def test_recommendation_hits_lower_clamp(self) -> None:
        agent = DynamicBSRAgent(min_periodicity_ms=5, max_periodicity_ms=80)
        extreme_state = BSRState(200_000, 400.0, 100.0, 0.90)

        recommendation = agent.recommend_periodicity(extreme_state)

        self.assertEqual(recommendation, 5)

    def test_high_load_makes_recommendation_more_aggressive(self) -> None:
        agent = DynamicBSRAgent()
        normal_load = BSRState(30_000, 18.0, 0.4, 0.80)
        high_load = BSRState(30_000, 18.0, 0.4, 0.90)

        normal_periodicity = agent.recommend_periodicity(normal_load)
        high_load_periodicity = agent.recommend_periodicity(high_load)

        self.assertGreater(normal_periodicity, high_load_periodicity)

    def test_constructor_validates_bounds_and_target_latency(self) -> None:
        with self.assertRaises(ValueError):
            DynamicBSRAgent(min_periodicity_ms=90, max_periodicity_ms=80)
        with self.assertRaises(ValueError):
            DynamicBSRAgent(min_periodicity_ms=0, max_periodicity_ms=80)
        with self.assertRaises(ValueError):
            DynamicBSRAgent(target_latency_ms=0)


if __name__ == "__main__":
    unittest.main()
