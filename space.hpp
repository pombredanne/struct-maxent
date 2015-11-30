#include <vector>

#ifndef SPACE_HPP
#define SPACE_HPP

// This class represents a point in space to which
// DMaxEnt assigns probabilities. Inital probabilitstic weight is 1.0.
// Please remember to finalize your point once you are done adding features
// otherwise behavior is undefined. Note that probabilistic weight
// of finalzied points is allowed to be modified.
// Sample usage:
//   Point point = new Point(id);
//   point.AddRawFeature(value1);
//   point.AddRawFeature(value2);
//   point.Finalize();
//   double raw_feature_value = point.GetRawFeature(index);
//   double probabilistic_weight = point.GetProbWeight();
//   point.SetProbWeight(new_probabilistic_weight);
class Point{
public:
  Point(int id);
  int GetId();
  double GetRawFeature(int index);
  int AddRawFeature(double value);
  double GetProbWeight();
  void SetProbWeight(double value);
  int NumRawFeatures();
  void Finalize();
private:
  int id;
  bool finalized;
  double probability_weight;
  std::vector<double> raw_features;
};


// This class represents underlying space X. Space is a set of points.
// Each point in the space has a unique identifier (integer key).
// Once you are done building your space please finalize it
// otherwise behaviour is undefined. 
// DMaxEnt fits probability density over this space. Sample Usage:
//   Space X = new Space();
//   X.AddPoint(key1, point1);
//   X.AddPoint(key2, point2);
//   ...
//   X.Finalize();
//   X.GetPoint(some_key);
class Space{
public:
  Space();
  int AddPoint(Point &point);
  Point& GetPoint(int key);
  void Finalize();
  int NumPoints();
  typedef std::vector<Point>::iterator SpaceIterator;
  SpaceIterator begin();
  SpaceIterator end();
private:
  bool finalized;
  std::vector<Point> points;
};

// An example is a pointer to a point in space
typedef Point* Example;

// A sample is a vector of examples
typedef std::vector<Example> Sample;

#endif
