#include "gtest/gtest.h"
#include "constants.hpp"
#include "tree.hpp"
#include "space.hpp"

// Tests that functionality related to adding, getting and clearing
// points is performing correctly. This inludes getting probability
// weights and sample counts for points in the node.
TEST(TreeTest, TestAddingGettingClearingPointsAndSamples) {
  Point *point = new Point(1);
  Point *point2 = new Point(7);
  Point *point3 = new Point(99);
  point->SetProbWeight(2.0);
  point2->SetProbWeight(0.0);
  point3->SetProbWeight(3.0);
  Node *node = new Node();
  EXPECT_NEAR(0.0, node->GetPopulationWeight(), gTolerance);
  node->AddPoint(point);
  EXPECT_NEAR(2.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(0, node->GetSampleCount());
  node->AddPoint(point2);
  EXPECT_NEAR(2.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(0, node->GetSampleCount());
  node->AddPoint(point3);
  EXPECT_NEAR(5.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(0, node->GetSampleCount());
  int ids[3] = {1, 7, 99};
  int i = 0;
  for (std::vector<Point*>::iterator it = node->PointsBegin();
       it != node->PointsEnd(); it++) {
    EXPECT_EQ(ids[i], (*it)->GetId());
    i++;
  }
  EXPECT_TRUE(node->SamplesBegin() == node->SamplesEnd());
  node->ClearPoints();
  EXPECT_NEAR(0.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_TRUE(node->PointsBegin() == node->PointsEnd());
  
  EXPECT_EQ(0, node->GetSampleCount());
  node->AddSample(point);
  EXPECT_NEAR(0.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(1, node->GetSampleCount());
  node->AddSample(point);
  EXPECT_NEAR(0.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(2, node->GetSampleCount());
  node->AddSample(point2);
  EXPECT_NEAR(0.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(3, node->GetSampleCount());
  node->AddSample(point3);
  EXPECT_NEAR(0.0, node->GetPopulationWeight(), gTolerance);
  EXPECT_EQ(4, node->GetSampleCount());
  int ids2[4] = {1, 1, 7, 99};
  i = 0;
  for (std::vector<Point*>::iterator it = node->SamplesBegin();
       it != node->SamplesEnd(); it++) {
    EXPECT_EQ(ids2[i], (*it)->GetId());
    i++;
  }
  EXPECT_TRUE(node->PointsBegin() == node->PointsEnd());  
  node->ClearSamples();
  EXPECT_EQ(0, node->GetSampleCount());
  EXPECT_TRUE(node->SamplesBegin() == node->SamplesEnd());
}

// Tests that functionality related to adding children and accessing them.
// This includes checking and setting thresholds and values of the node
// as well as checking if a particular node is a leaf.
TEST(TreeTest, TestChildren) {
  Node *node = new Node();
  EXPECT_TRUE(node->IsLeaf());
  Node *left_child = new Node();
  node->SetLeftChild(left_child);
  EXPECT_FALSE(node->IsLeaf());
  EXPECT_TRUE(left_child->IsLeaf());
  EXPECT_EQ(left_child, node->GetLeftChild());
  EXPECT_EQ(NULL, node->GetRightChild());
  Node *right_child = new Node();
  node->SetRightChild(right_child);
  EXPECT_FALSE(node->IsLeaf());
  EXPECT_TRUE(right_child->IsLeaf());
  EXPECT_EQ(right_child, node->GetRightChild());
  EXPECT_EQ(left_child, node->GetLeftChild());

  node->SetThreshold(0.5);
  node->SetFeature(1);
  left_child->SetValue(2.0);
  right_child->SetValue(-1.0);
  EXPECT_NEAR(2.0, left_child->GetValue(), gTolerance);
  EXPECT_NEAR(-1.0, right_child->GetValue(), gTolerance);
  Point *point = new Point(1);
  point->AddRawFeature(1.0);
  point->AddRawFeature(0.0);
  point->AddRawFeature(2.0);
  EXPECT_EQ(left_child, node->Child(point));
  Point *point2 = new Point(2);
  point2->AddRawFeature(-1.0);
  point2->AddRawFeature(1.0);
  point2->AddRawFeature(-2.0);
  EXPECT_EQ(right_child, node->Child(point2));
}
