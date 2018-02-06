#include <boost/tuple/tuple.hpp>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <fstream>
#include <vector>
// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT FT;
typedef Kernel::Point_3 Point;
// Data type := index, followed by the point, followed by three integers that
// define the Red Green Blue color of the point.
typedef boost::tuple<int, Point, int, int, int> IndexedPointWithColorTuple;
#pragma once


float brute_force_min_dist(IndexedPointWithColorTuple lhs, std::vector<IndexedPointWithColorTuple> rhs);
float brute_force_mean_dist(std::vector<IndexedPointWithColorTuple> lhs, std::vector<IndexedPointWithColorTuple> rhs);
bool brute_force_thresh_dist_detect(IndexedPointWithColorTuple lhs, std::vector<IndexedPointWithColorTuple> rhs, float thresh);
int brute_force_thresh_dist_counter(std::vector<IndexedPointWithColorTuple> lhs,
                                    std::vector<IndexedPointWithColorTuple> rhs,
                                    float thresh);
float computeFscore(std::vector<IndexedPointWithColorTuple> lhs,
                    std::vector<IndexedPointWithColorTuple> rhs,
                    float thresh,
                    float& acc,
                    float& recall);

std::vector<IndexedPointWithColorTuple> resample_reconstruction_points(std::vector<IndexedPointWithColorTuple>& recon,
                                                                       std::vector<IndexedPointWithColorTuple> gt,
                                                                       float thresh);
void brute_force_thresh_dist_detect(IndexedPointWithColorTuple lhs,
                                    std::vector<IndexedPointWithColorTuple>& rhs,
                                    float thresh,
                                    std::vector<Point>& out);
float meanDist(Point l, Point r);
void find_bounding_volume(Point& min_p, Point& max_p, std::vector<IndexedPointWithColorTuple> input);
std::vector<IndexedPointWithColorTuple> brute_force_delete_distant_outliers(std::vector<IndexedPointWithColorTuple>& recon,
                                                                            std::vector<IndexedPointWithColorTuple> gt,
                                                                            float thresh);
