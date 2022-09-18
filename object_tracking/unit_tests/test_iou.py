import unittest
from utils.utils import compute_iou

class TestComputeIOU(unittest.TestCase):
    def test_commutative(self):
        bbox1 = [30, 15, 90, 80]
        bbox2 = [60, 70, 79, 92]
        self.assertEqual(compute_iou(bbox1, bbox2), compute_iou(bbox1, bbox2))

    def test_not_intersect(self):
        bbox1 = [30, 15, 90, 80]
        bbox2 = [120, 95, 79, 92]
        self.assertEqual(compute_iou(bbox1, bbox2), 0.0)
        bbox2 = [130, 70, 79, 92]
        self.assertEqual(compute_iou(bbox1, bbox2), 0.0)

    def test_intersect1(self):
        bbox1 = [30, 15, 90, 80]
        bbox2 = [60, 70, 79, 92]
        self.assertEqual(compute_iou(bbox1, bbox2), 0.11566933991363355)

    def test_partial_overlap(self):
        bbox1 = [30, 15, 90, 80]
        bbox2 = [40, 40, 20, 20]
        self.assertEqual(compute_iou(bbox1, bbox2), 0.05555555555555555)

    def test_full_overlap(self):
        bbox1 = [40, 40, 20, 20]
        bbox2 = [40, 40, 20, 20]
        self.assertEqual(compute_iou(bbox1, bbox2), 1.0)

    # def test_input_validation(self):
    #     bbox1 = [-10, -13, 20, 40]
    #     bbox2 = [100, -13, 20, 40]
    #     with self.assertRaises(AssertionError) as cm:
    #         compute_iou(bbox1, bbox2)
    #     self.assertEqual(cm.exception, "bbox1 coordinates must be positive")

    #     bbox1 = [10, 13, 20, 40]
        


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComputeIOU)
    unittest.TextTestRunner(verbosity=2).run(suite)

