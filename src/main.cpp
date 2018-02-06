// #include "pcl\registration\icp.h"
#include "Utils.h"
#include <fstream>
#include <vector>
using namespace std;

int main(int argc, char* argv[]) {
    std::string recon_file = "data/machine_vision/blackfly_pmvs.xyz";
    std::string gt_file = "data/machine_vision/blackfly_agisoft_gt.xyz";
    // Reads a .xyz point set file in points.
    // As the point is the second element of the tuple (that is with index 1)
    // we use a property map that accesses the 1st element of the tuple.
    std::string out_path = recon_file.substr(0, recon_file.size() - 4) + "-fs_resampled3.xyz";
    std::string out_path_log = recon_file.substr(0, recon_file.size() - 4) + "-info3.txt";
    std::vector<IndexedPointWithColorTuple> points_recon, points_gt;
    std::ifstream stream_recon(recon_file), stream_gt(gt_file);

    CGAL::read_xyz_points(
        stream_recon, std::back_inserter(points_recon),
        CGAL::Nth_of_tuple_property_map<1, IndexedPointWithColorTuple>());

    CGAL::read_xyz_points(
        stream_gt, std::back_inserter(points_gt),
        CGAL::Nth_of_tuple_property_map<1, IndexedPointWithColorTuple>());

    int r_size = points_recon.size();
    int g_size = points_gt.size();

    cout << r_size << endl;
    cout << g_size << endl;
    float threshold = 0.008;  // zbog skale koja nije u mm
    float mean_acc = 0.5;
    mean_acc = brute_force_mean_dist(points_recon, points_gt);

    // std::vector<IndexedPointWithColorTuple> points_recon_outl = brute_force_delete_distant_outliers(points_recon,
    // points_gt,
    // mean_acc  / 2);

    std::vector<IndexedPointWithColorTuple> points_recon_resamp = resample_reconstruction_points(points_recon,
                                                                                                 points_gt,
                                                                                                 mean_acc  / 2);

    // for (int i = 0; i < points_recon.size(); i++) {
    // Point p = get<1>(points_recon[i]);
    // output.push_back(p);
    // }

    ofstream outFile(out_path);
    ofstream outFile_log(out_path_log);


    CGAL::write_xyz_points(outFile, points_recon_resamp.begin(), points_recon_resamp.end(),
                           CGAL::Nth_of_tuple_property_map<1, IndexedPointWithColorTuple>());

    float acc, rec;
    float fscore = computeFscore(points_recon_resamp, points_gt, threshold, acc, rec);

    outFile_log << "3D reconstruction Fscore evaluation. " << std::endl;
    outFile_log << "Recon File: " << recon_file << " " << "Ground truth file: " << gt_file << std::endl;
    outFile_log << "Mean distance reconstruction to ground truth: " << mean_acc << std::endl;
    outFile_log << "Fscore: " << fscore << std::endl;
    outFile_log << "Precision: " << acc << " %" << std::endl;
    outFile_log << "Recall: " << rec << " %" << std::endl;
    outFile_log << "Threshold: " << threshold << std::endl;

    outFile_log.close();


    cout << fscore << endl;


    return EXIT_SUCCESS;
}
