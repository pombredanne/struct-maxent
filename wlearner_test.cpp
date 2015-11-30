#include <map>
#include "gtest/gtest.h"
#include "wlearner.hpp"
#include "constants.hpp"

// Test Feature for TreeLearner class.
class TreeLearnerTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    point1 = new Point(1);
    point1->AddRawFeature(1);
    point1->AddRawFeature(-1.0);
    point1->AddRawFeature(0.0);
    point1->AddRawFeature(123);
    point1->SetProbWeight(0.1);

    point2 = new Point(2);
    point2->AddRawFeature(1);
    point2->AddRawFeature(-0.5);
    point2->AddRawFeature(123);
    point2->AddRawFeature(3);
    point2->SetProbWeight(1.1);

    point3 = new Point(3);
    point3->AddRawFeature(-1);
    point3->AddRawFeature(0.0);
    point3->AddRawFeature(40);
    point3->AddRawFeature(-3);
    point3->SetProbWeight(0.3);

    point4 = new Point(4);
    point4->AddRawFeature(-1);
    point4->AddRawFeature(0.1);
    point4->AddRawFeature(-40);
    point4->AddRawFeature(1);
    point4->SetProbWeight(2.1);

    point5 = new Point(5);
    point5->AddRawFeature(0.1);
    point5->AddRawFeature(-0.1);
    point5->AddRawFeature(0.0);
    point5->AddRawFeature(1);
    point5->SetProbWeight(0.8);

    point6 = new Point(6);
    point6->AddRawFeature(0.1);
    point6->AddRawFeature(-0.1);
    point6->AddRawFeature(1.1);
    point6->AddRawFeature(1);
    point6->SetProbWeight(0.1);

    point7 = new Point(7);
    point7->AddRawFeature(-1);
    point7->AddRawFeature(0.1);
    point7->AddRawFeature(2);
    point7->AddRawFeature(-3);
    point7->SetProbWeight(0.1);

    point8 = new Point(8);
    point8->AddRawFeature(-1);
    point8->AddRawFeature(0.1);
    point8->AddRawFeature(2);
    point8->AddRawFeature(0);
    point8->SetProbWeight(0.2);

    point9 = new Point(9);
    point9->AddRawFeature(-10);
    point9->AddRawFeature(-1.1);
    point9->AddRawFeature(0.5);
    point9->AddRawFeature(1);
    point9->SetProbWeight(0.2);

    point10 = new Point(10);
    point10->AddRawFeature(-10);
    point10->AddRawFeature(-1.1);
    point10->AddRawFeature(1.5);
    point10->AddRawFeature(1);
    point10->SetProbWeight(1.2);

    point11 = new Point(11);
    point11->AddRawFeature(-1);
    point11->AddRawFeature(0.1);
    point11->AddRawFeature(2);
    point11->AddRawFeature(-4);
    point11->SetProbWeight(0.6);

    point12 = new Point(12);
    point12->AddRawFeature(-1);
    point12->AddRawFeature(0.1);
    point12->AddRawFeature(2);
    point12->AddRawFeature(0);
    point12->SetProbWeight(0.2);

    std::map<double, double> feature1_map;
    feature1_map[-10.0] = -3.0;
    feature1_map[-1.0] = 0.0;
    feature1_map[0.1] = 2.0;
    feature1_map[1.0] = 2.0;
    vtot.push_back(feature1_map);
    std::map<double, double> feature2_map;
    feature2_map[-1.1] = -0.7;
    feature2_map[-1.0] = -0.7;
    feature2_map[-0.5] = 0.05;
    feature2_map[-0.1] = 0.05;
    feature2_map[0.0] = 0.05;
    feature2_map[0.1] = 3.0;
    feature2_map[2.0] = 3.0;
    vtot.push_back(feature2_map);
    std::map<double, double> feature3_map;
    feature3_map[-40.0] = 1.3;
    feature3_map[0.0] = 1.3;
    feature3_map[0.5] = 1.3;
    feature3_map[1.1] = 1.3;
    feature3_map[1.5] = 4.0;
    feature3_map[2.0] = 4.0;
    feature3_map[3.0] = 4.0;
    feature3_map[40.0] = 124.0;
    feature3_map[123.0] = 124.0;
    vtot.push_back(feature3_map);
    std::map<double, double> feature4_map;
    feature4_map[-4.0] = -1.0;
    feature4_map[-3.0] = -1.0;
    feature4_map[0.0] = 5.0;
    feature4_map[1.0] = 5.0;
    feature4_map[3.0] = 5.0;
    feature4_map[4.0] = 5.0;
    feature4_map[123.0] = 124.0;
    vtot.push_back(feature4_map);

    node = new Node();
    node->AddPoint(point1);
    node->AddPoint(point2);
    node->AddPoint(point3);
    node->AddPoint(point4);
    node->AddPoint(point5);
    node->AddPoint(point6);
    node->AddPoint(point7);
    node->AddPoint(point8);
    node->AddPoint(point9);
    node->AddPoint(point10);
    node->AddPoint(point11);
    node->AddPoint(point12);
    node->AddSample(point3);
    node->AddSample(point6);
    node->AddSample(point6);
    node->AddSample(point7);
    node->AddSample(point7);
    node->AddSample(point8);
    node->AddSample(point10);
    node->AddSample(point11);

    space = new Space();
    space->AddPoint(*point1);
    space->AddPoint(*point2);
    space->AddPoint(*point3);
    space->AddPoint(*point4);
    space->AddPoint(*point5);
    space->AddPoint(*point6);
    space->AddPoint(*point7);
    space->AddPoint(*point8);
    space->AddPoint(*point9);
    space->AddPoint(*point10);
    space->AddPoint(*point11);
    space->AddPoint(*point12);
    space->Finalize();

    sample.push_back(point3);
    sample.push_back(point6);
    sample.push_back(point6);
    sample.push_back(point7);
    sample.push_back(point7);
    sample.push_back(point8);
    sample.push_back(point10);
    sample.push_back(point11);

  }
  Point *point1;
  Point *point2;
  Point *point3;
  Point *point4;
  Point *point5;
  Point *point6;
  Point *point7;
  Point *point8;
  Point *point9;
  Point *point10;
  Point *point11;
  Point *point12;
  Space *space;
  Sample sample;
  Node *node;
  std::vector< std::map<double, double> > vtot;
  TreeLearner *tlearner;
};

// Tests that tree complexity is computed correctly.
TEST_F(TreeLearnerTest, TestTreeComplexity) {
  tlearner = new TreeLearner(4, 0.5, 0.1, vtot);
  EXPECT_NEAR(1.6512874658240806, tlearner->TreeComplexity(11, 239),
	      gTolerance);
  EXPECT_NEAR(2.839056879562188, tlearner->TreeComplexity(13, 75),
	      gTolerance);
  tlearner = new TreeLearner(10, 0.5, 0.1, vtot);
  EXPECT_NEAR(1.9446339760875195, tlearner->TreeComplexity(11, 239),
	      gTolerance);
  EXPECT_NEAR(3.3434072396876133, tlearner->TreeComplexity(13, 75),
	      gTolerance);

}

// Tests that gradient is computed correctly.
TEST_F(TreeLearnerTest, TestGradient) {
  tlearner = new TreeLearner(4, 0.5, 0.1, vtot);
  EXPECT_NEAR(0.0, tlearner->Gradient(11, 239, 0.5),
	      gTolerance);
  EXPECT_NEAR(0.0, tlearner->Gradient(11, 239, -0.5),
	      gTolerance);
  EXPECT_NEAR(-0.5743562670879597, tlearner->Gradient(11, 239, -1.5),
	      gTolerance);
  EXPECT_NEAR(10.5743562670879596, tlearner->Gradient(11, 239, 11.5),
	      gTolerance);
  tlearner = new TreeLearner(4, 0.3, 0.7, vtot);
  EXPECT_NEAR(0.0, tlearner->Gradient(13, 75, 1.5),
	      gTolerance);
  EXPECT_NEAR(0.0, tlearner->Gradient(13, 75, -1.5),
	      gTolerance);
  EXPECT_NEAR(0.9482829361313436, tlearner->Gradient(13, 75, 2.5),
	      gTolerance);
  EXPECT_NEAR(-1.9482829361313436, tlearner->Gradient(13, 75, -3.5),
	      gTolerance);
}

// Tests building threshold to weights map
TEST_F(TreeLearnerTest, TestBuildThresholdToWeightsMap) {
  tlearner = new TreeLearner(4, 0.5, 0.1, vtot);
  std::map<double, std::pair<double, int> > ttow =
    tlearner->BuildThresholdToWeightsMap(node, 0);
  EXPECT_EQ(3, ttow.size());
  EXPECT_NEAR(1.4, ttow[-3].first, gTolerance);
  EXPECT_EQ(1, ttow[-3].second);
  EXPECT_NEAR(3.5, ttow[0].first, gTolerance);
  EXPECT_EQ(5, ttow[0].second);
  EXPECT_NEAR(2.1, ttow[2].first, gTolerance);
  EXPECT_EQ(2, ttow[2].second);
  ttow = tlearner->BuildThresholdToWeightsMap(node, 1);
  EXPECT_EQ(3, ttow.size());
  EXPECT_NEAR(1.5, ttow[-0.7].first, gTolerance);
  EXPECT_EQ(1, ttow[-0.7].second);
  EXPECT_NEAR(2.3, ttow[0.05].first, gTolerance);
  EXPECT_EQ(3, ttow[0.05].second);
  EXPECT_NEAR(3.2, ttow[3].first, gTolerance);
  EXPECT_EQ(4, ttow[3].second);
  ttow = tlearner->BuildThresholdToWeightsMap(node, 2);
  EXPECT_EQ(3, ttow.size());
  EXPECT_NEAR(3.3, ttow[1.3].first, gTolerance);
  EXPECT_EQ(2, ttow[1.3].second);
  EXPECT_NEAR(2.3, ttow[4].first, gTolerance);
  EXPECT_EQ(5, ttow[4].second);
  EXPECT_NEAR(1.4, ttow[124].first, gTolerance);
  EXPECT_EQ(1, ttow[124].second);
  ttow = tlearner->BuildThresholdToWeightsMap(node, 3);
  EXPECT_EQ(3, ttow.size());
  EXPECT_NEAR(1.0, ttow[-1].first, gTolerance);
  EXPECT_EQ(4, ttow[-1].second);
  EXPECT_NEAR(5.9, ttow[5].first, gTolerance);
  EXPECT_EQ(4, ttow[5].second);
  EXPECT_NEAR(0.1, ttow[124].first, gTolerance);
  EXPECT_EQ(0, ttow[124].second);
}

// Tests that growing tree is performed correctly.
TEST_F(TreeLearnerTest, TestGrowTree) {
  tlearner = new TreeLearner(4, 0.5, 0.1, vtot);
  Node *left_child;
  Node *right_child;
  tlearner->GrowTree(node, 0, 1, 1, &left_child, &right_child);
  EXPECT_NEAR(1.0, left_child->GetValue(), gTolerance);
  EXPECT_NEAR(0.0, right_child->GetValue(), gTolerance);
  EXPECT_EQ(node->PointsBegin(), node->PointsEnd());
  EXPECT_EQ(node->SamplesBegin(), node->SamplesEnd());
  int left_ids[6] = {1, 2, 5, 6, 9, 10};
  int i = 0;
  for (std::vector<Point*>::iterator it = left_child->PointsBegin();
       it != left_child->PointsEnd(); it++) {
    Point *point = *it;
    EXPECT_EQ(left_ids[i], point->GetId());
    i++;
  }
  int right_ids[6] = {3, 4, 7, 8, 11, 12};
  i = 0;
  for (std::vector<Point*>::iterator it = right_child->PointsBegin();
       it != right_child->PointsEnd(); it++) {
    Point *point = *it;
    EXPECT_EQ(right_ids[i], point->GetId());
    i++;
  }
  int left_sample_ids[3] = {6, 6, 10};
  i = 0;
  for (std::vector<Point*>::iterator it = left_child->SamplesBegin();
       it != left_child->SamplesEnd(); it++) {
    Point *point = *it;
    EXPECT_EQ(left_sample_ids[i], point->GetId());
    i++;
  }
  int right_sample_ids[5] = {3, 7, 7, 8, 11};
  i = 0;
  for (std::vector<Point*>::iterator it = right_child->SamplesBegin();
       it != right_child->SamplesEnd(); it++) {
    Point *point = *it;
    EXPECT_EQ(right_sample_ids[i], point->GetId());
    i++;
  }  
}

// Tests that Tree Learner finds best thresholds correctly
TEST_F(TreeLearnerTest, TestBestThreshold) {
  tlearner = new TreeLearner(4, 0.01, 0.01, vtot);
  double th;
  double grad;
  double val;
  double diff;
  node->SetValue(0.0);
  tlearner->BestThreshold(0, node, 0.0, 7.0, 8, 1, &th, &grad, &val, &diff);
  EXPECT_NEAR(-3.0, th, gTolerance);
  EXPECT_NEAR(-0.03347294734416998, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(-0.075, diff, gTolerance);
  node->SetValue(1.0);
  tlearner->BestThreshold(0, node, 2.0, 14.0, 16, 4, &th, &grad, &val, &diff);
  EXPECT_NEAR(-3.0, th, gTolerance);
  EXPECT_NEAR(1.9930020375126904, grad, gTolerance);
  EXPECT_NEAR(1.0, val, gTolerance);
  EXPECT_NEAR(2.0375, diff, gTolerance);
  node->SetValue(0.0);
  tlearner->BestThreshold(1, node, 0.0, 7.0, 8, 1, &th, &grad, &val, &diff);
  EXPECT_NEAR(-0.7, th, gTolerance);
  EXPECT_NEAR(-0.04775866162988426, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(-0.08928571428571427, diff, gTolerance);
  node->SetValue(1.0);
  tlearner->BestThreshold(1, node, -2.0, 14.0, 16, 4, &th, &grad, &val, &diff);
  EXPECT_NEAR(-0.7, th, gTolerance);
  EXPECT_NEAR(-2.0001448946555476, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(-2.044642857142857, diff, gTolerance);
  node->SetValue(0.0);
  tlearner->BestThreshold(2, node, 0.0, 7.0, 8, 1, &th, &grad, &val, &diff);
  EXPECT_NEAR(1.3, th, gTolerance);
  EXPECT_NEAR(-0.1799015187727414, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(-0.22142857142857142, diff, gTolerance);
  node->SetValue(0.0);
  tlearner->BestThreshold(2, node, 0.1, 7.0, 8, 500, &th, &grad, &val, &diff);
  EXPECT_NEAR(1.3, th, gTolerance);
  EXPECT_NEAR(0.0, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(-0.12142857142857141, diff, gTolerance);
  node->SetValue(0.0);
  tlearner->BestThreshold(3, node, 0.0, 7.0, 8, 1, &th, &grad, &val, &diff);
  EXPECT_NEAR(-1.0, th, gTolerance);
  EXPECT_NEAR(0.31561580448702714, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);

  Node *left_child;
  Node *right_child;
  tlearner->GrowTree(node, -1.0, 3, 0.0, &left_child, &right_child);
  tlearner->BestThreshold(0, left_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(0.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);  
  tlearner->BestThreshold(1, left_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(3.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);  
  tlearner->BestThreshold(2, left_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(124.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(0.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);
  tlearner->BestThreshold(0, right_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(2.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(1.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);  
  tlearner->BestThreshold(1, right_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(3.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(1.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);  
  tlearner->BestThreshold(2, right_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(124.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(1.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);  
  tlearner->BestThreshold(3, right_child, 0.35714285714285715, 7.0, 8, 3,
			  &th, &grad, &val, &diff);
  EXPECT_NEAR(124.0, th, gTolerance);
  EXPECT_NEAR(0.30762160510080794, grad, gTolerance);
  EXPECT_NEAR(1.0, val, gTolerance);
  EXPECT_NEAR(0.35714285714285715, diff, gTolerance);
}

// Tests that Tree Learner trains a tree feature correctly.
TEST_F(TreeLearnerTest, TestTrain) {
  tlearner = new TreeLearner(4, 0.01, 0.01, vtot);
  double gradient;
  Feature *feature;
  tlearner->Train(*space, sample, &feature, &gradient);
  EXPECT_NEAR(0.5, feature->GetSampleExpectation(), gTolerance);
  EXPECT_NEAR(6.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
  EXPECT_NEAR(0.31561580448702714, gradient, gTolerance);
  EXPECT_NEAR(3.1527052655830015, feature->Complexity(), gTolerance);
  EXPECT_EQ(3, dynamic_cast<TreeFeature*>(feature)->TreeSize());

  vtot.clear();
  std::map<double, double> feature_map;
  feature_map[-1.0] = 0.0;
  feature_map[1.0] = 2.0;
  vtot.push_back(feature_map);
  vtot.push_back(feature_map);

  tlearner = new TreeLearner(2, 0.0, 0.0, vtot);

  point1 = new Point(1);
  point2 = new Point(2);
  point3 = new Point(3);
  point4 = new Point(4);

  point1->AddRawFeature(-1);
  point1->AddRawFeature(-1);
  point2->AddRawFeature(-1);
  point2->AddRawFeature(1);
  point3->AddRawFeature(1);
  point3->AddRawFeature(-1);
  point4->AddRawFeature(1);
  point4->AddRawFeature(1);

  point1->SetProbWeight(1);
  point2->SetProbWeight(1);
  point3->SetProbWeight(0);
  point4->SetProbWeight(1);

  space = new Space();
  space->AddPoint(*point1);
  space->AddPoint(*point2);
  space->AddPoint(*point3);
  space->AddPoint(*point4);
  space->Finalize();
  
  sample.clear();
  sample.push_back(point3);
  
  tlearner->Train(*space, sample, &feature, &gradient);
  EXPECT_NEAR(1.0, feature->GetSampleExpectation(), gTolerance);
  EXPECT_NEAR(0.0, feature->GetUnnormalizedPopulationExpectation(),
	      gTolerance);
  EXPECT_NEAR(-1.0, gradient, gTolerance);
  EXPECT_NEAR(5.522542525380642, feature->Complexity(), gTolerance);
  EXPECT_EQ(5, dynamic_cast<TreeFeature*>(feature)->TreeSize());
}

class MonomialLearnerTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    point1 = new Point(1);
    point1->AddRawFeature(1);
    point1->AddRawFeature(-1);
    point1->AddRawFeature(0.5);
    point1->SetProbWeight(0.5);

    point2 = new Point(2);
    point2->AddRawFeature(0.5);
    point2->AddRawFeature(-0.5);
    point2->AddRawFeature(1);
    point2->SetProbWeight(1.5);

    point3 = new Point(3);
    point3->AddRawFeature(-1);
    point3->AddRawFeature(0.0);
    point3->AddRawFeature(1);
    point3->SetProbWeight(2.0);

    point4 = new Point(4);
    point4->AddRawFeature(-1);
    point4->AddRawFeature(0.1);
    point4->AddRawFeature(0.0);
    point4->SetProbWeight(1.0);

    space = new Space();
    space->AddPoint(*point1);
    space->AddPoint(*point2);
    space->AddPoint(*point3);
    space->AddPoint(*point4);
    space->Finalize();
    
    sample.push_back(point2);
    sample.push_back(point3);
    sample.push_back(point3);
    
  }
  MonomialLearner *mlearner;
  Point *point1;
  Point *point2;
  Point *point3;
  Point *point4;
  Space *space;
  Sample sample;
  std::vector<double> point_values;
  std::vector<double> sample_values;
};

// Tests that monomial complexity is computed correctly.
TEST_F(MonomialLearnerTest, TestMonomialComplexity) {
  mlearner = new MonomialLearner(4, 0.5, 0.1, 2);
  EXPECT_NEAR(0.34059931932994864, mlearner->MonomialComplexity(5, 239),
	      gTolerance);
  EXPECT_NEAR(0.549199900266618, mlearner->MonomialComplexity(13, 239),
	      gTolerance);
  mlearner = new MonomialLearner(10, 0.5, 0.1, 1);
  EXPECT_NEAR(0.5540860454230366, mlearner->MonomialComplexity(5, 75),
	      gTolerance);
  EXPECT_NEAR(0.8934369025871959, mlearner->MonomialComplexity(13, 75),
	      gTolerance);
}

// Tests that gradient is computed correctly.
TEST_F(MonomialLearnerTest, TestGradient) {
  mlearner = new MonomialLearner(4, 0.5, 0.1, 2);
  EXPECT_NEAR(0.0, mlearner->Gradient(5, 239, 0.1),
	      gTolerance);
  EXPECT_NEAR(0.0, mlearner->Gradient(13, 239, -0.1),
	      gTolerance);
  EXPECT_NEAR(-0.2297003403350257, mlearner->Gradient(5, 239, -0.5),
	      gTolerance);
  EXPECT_NEAR(11.125400049866691, mlearner->Gradient(13, 239, 11.5),
	      gTolerance);
  mlearner = new MonomialLearner(4, 0.3, 0.7, 1.0);
  EXPECT_NEAR(0.0, mlearner->Gradient(13, 75, 0.5),
	      gTolerance);
  EXPECT_NEAR(0.0, mlearner->Gradient(13, 75, -0.5),
	      gTolerance);
  EXPECT_NEAR(0.09202792479051292, mlearner->Gradient(13, 75, 1.0),
	      gTolerance);
  EXPECT_NEAR(-0.5920279247905129, mlearner->Gradient(13, 75, -1.5),
	      gTolerance);
}

// Tests that finding best feature works correctly.
TEST_F(MonomialLearnerTest, TestBestFeature) {
  point_values.push_back(0.5);
  point_values.push_back(1.5);
  point_values.push_back(2.0);
  point_values.push_back(1.0);

  sample_values.push_back(1.0);
  sample_values.push_back(1.0);
  sample_values.push_back(1.0);

  mlearner = new MonomialLearner(3, 0.1, 0.01, 1.0);
  double grad;
  int feature;
  double population_expectation;
  double sample_expectation;
  mlearner->BestFeature(point_values, sample_values, *space, sample,
			5.0, 0, &grad, &feature, &population_expectation,
			&sample_expectation);
  EXPECT_NEAR(-0.154419149779556, grad, gTolerance);
  EXPECT_EQ(2, feature);
  EXPECT_NEAR(0.75, population_expectation, gTolerance);
  EXPECT_NEAR(1.0, sample_expectation, gTolerance);

  point_values[0] *= 0.5;
  point_values[1] *= 1.0;
  point_values[2] *= 1.0;
  point_values[3] *= 0.0;

  sample_values[0] *= 1.0;
  sample_values[1] *= 1.0;
  sample_values[2] *= 1.0;

  mlearner->BestFeature(point_values, sample_values, *space, sample,
			5.0, 1, &grad, &feature, &population_expectation,
			&sample_expectation);
  EXPECT_NEAR(0.16897040093882765, grad, gTolerance);
  EXPECT_EQ(0, feature);
  EXPECT_NEAR(-0.2, population_expectation, gTolerance);
  EXPECT_NEAR(-0.5, sample_expectation, gTolerance);

  point_values[0] *= 1.0;
  point_values[1] *= 0.5;
  point_values[2] *= -1.0;
  point_values[3] *= -1.0;

  sample_values[0] *= 0.5;
  sample_values[1] *= -1.0;
  sample_values[2] *= -1.0;

  mlearner->BestFeature(point_values, sample_values, *space, sample,
			5.0, 2, &grad, &feature, &population_expectation,
			&sample_expectation);
  EXPECT_NEAR(0.11676961926324889, grad, gTolerance);
  EXPECT_EQ(2, feature);
  EXPECT_NEAR(-0.225, population_expectation, gTolerance);
  EXPECT_NEAR(-0.5, sample_expectation, gTolerance);
}


// Tests that training monomial feature works correctly.
TEST_F(MonomialLearnerTest, TestTrain) {
 mlearner = new MonomialLearner(3, 0.1, 0.01, 1.0);
 Feature *feature;
 double gradient;
 mlearner->Train(*space, sample, &feature, &gradient);
 EXPECT_NEAR(0.16897040093882765, gradient, gTolerance);
 EXPECT_EQ(2, dynamic_cast<MonomialFeature*>(feature)->GetPower());

 Point *test_point1 = new Point(1);
 test_point1->AddRawFeature(2);
 test_point1->AddRawFeature(1);
 test_point1->AddRawFeature(1);

 Point *test_point2 = new Point(2);
 test_point2->AddRawFeature(1);
 test_point2->AddRawFeature(2);
 test_point2->AddRawFeature(1);

 Point *test_point3 = new Point(3);
 test_point3->AddRawFeature(1);
 test_point3->AddRawFeature(1);
 test_point3->AddRawFeature(2);

 EXPECT_NEAR(2, feature->FeatureMap(test_point1), gTolerance);
 EXPECT_NEAR(1, feature->FeatureMap(test_point2), gTolerance);
 EXPECT_NEAR(2, feature->FeatureMap(test_point3), gTolerance);
}
