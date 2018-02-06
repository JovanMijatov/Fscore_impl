#include "Utils.h"
using namespace std;

float brute_force_min_dist(IndexedPointWithColorTuple lhs, std::vector<IndexedPointWithColorTuple> rhs) {
    float m_dist = INT_MAX;
    vector<IndexedPointWithColorTuple>::iterator brhs = rhs.begin();
    vector<IndexedPointWithColorTuple>::iterator erhs = rhs.end();


    while (brhs != erhs) {
        Point l, r;
        l = get<1>(lhs);
        r = get<1>(*brhs);
        float c_dist =
            sqrt(pow((l[0] - r[0]), 2) + pow((l[1] - r[1]), 2) + pow((l[2] - r[2]), 2));
        if (c_dist < m_dist) {
            m_dist = c_dist;
        }

        ++brhs;
    }

    return m_dist;
}

bool brute_force_thresh_dist_detect(IndexedPointWithColorTuple lhs, std::vector<IndexedPointWithColorTuple> rhs, float thresh) {
    float m_dist = INT_MAX;
    vector<IndexedPointWithColorTuple>::iterator brhs = rhs.begin();
    vector<IndexedPointWithColorTuple>::iterator erhs = rhs.end();

    while (brhs != erhs) {
        Point l, r;
        l = get<1>(lhs);
        r = get<1>(*brhs);
        float c_dist =
            sqrt(pow((l[0] - r[0]), 2) + pow((l[1] - r[1]), 2) + pow((l[2] - r[2]), 2));
        if (c_dist < thresh) {
            return true;
        }
        ++brhs;
    }

    return false;
}

void brute_force_thresh_dist_detect(IndexedPointWithColorTuple lhs,
                                    std::vector<IndexedPointWithColorTuple>& rhs,
                                    float thresh,
                                    std::vector<Point>& out) {
    float m_dist = INT_MAX;
    vector<IndexedPointWithColorTuple>::iterator brhs = rhs.begin();
    vector<IndexedPointWithColorTuple>::iterator erhs = rhs.end();
    Point l, r;
    l = get<1>(lhs);

    std::vector<IndexedPointWithColorTuple> out_rhs;

    while (brhs != erhs) {
        r = get<1>(*brhs);
        float c_dist = meanDist(l, r);
        if (c_dist < thresh) {
            out.push_back(r);
            // rhs.erase(brhs);
            brhs++;
        } else {
            IndexedPointWithColorTuple t = boost::make_tuple(0, r, 0, 0, 0);
            out_rhs.push_back(t);
            brhs++;
        }
    }
    rhs = out_rhs;
}

float meanDist(Point l, Point r) {
    return sqrt(pow((l[0] - r[0]), 2) + pow((l[1] - r[1]), 2) + pow((l[2] - r[2]), 2));
}

int brute_force_thresh_dist_counter(std::vector<IndexedPointWithColorTuple> lhs,
                                    std::vector<IndexedPointWithColorTuple> rhs,
                                    float thresh) {
    vector<IndexedPointWithColorTuple>::iterator blhs = lhs.begin();
    vector<IndexedPointWithColorTuple>::iterator elhs = lhs.end();

    int size = lhs.size(), counter = 0, ctr = 0;

    float m_dist = 0;
    while (blhs != elhs) {
        bool cond = brute_force_thresh_dist_detect(*blhs, rhs, thresh);
        if (cond) {
            ctr++;
        }

        ++blhs;
        counter++;
        float percent = float(100 * counter) / size;
        if (counter % 100 == 0) {
            cout << percent << endl;
        }
    }
    return ctr;
}

float brute_force_mean_dist(std::vector<IndexedPointWithColorTuple> lhs, std::vector<IndexedPointWithColorTuple> rhs) {
    vector<IndexedPointWithColorTuple>::iterator blhs = lhs.begin();
    vector<IndexedPointWithColorTuple>::iterator elhs = lhs.end();

    int size = lhs.size(), counter = 0;

    float m_dist = 0;
    while (blhs != elhs) {
        float c_dist = brute_force_min_dist(*blhs, rhs);
        m_dist += c_dist;

        ++blhs;
        counter++;
        float percent = float(100 * counter) / size;
        if (counter % 100 == 0) {
            cout << percent << endl;
        }
    }
    m_dist /= size;
    return m_dist;
}

float computeFscore(std::vector<IndexedPointWithColorTuple> lhs,
                    std::vector<IndexedPointWithColorTuple> rhs,
                    float thresh,
                    float& acc,
                    float& recall) {
    int acc_thresh = brute_force_thresh_dist_counter(lhs, rhs, thresh);
    int recall_thresh = brute_force_thresh_dist_counter(rhs, lhs, thresh);

    int lhs_size = lhs.size();
    int rhs_size = rhs.size();

    acc = float(100 * (float)(acc_thresh) / lhs_size);
    recall = float(100 * (float)(recall_thresh) / rhs_size);

    float F = 2 * acc * recall / (acc + recall);
    return F;
}

std::vector<IndexedPointWithColorTuple> resample_reconstruction_points(std::vector<IndexedPointWithColorTuple>& recon,
                                                                       std::vector<IndexedPointWithColorTuple> gt,
                                                                       float thresh) {
    std::vector<Point> detected;
    std::vector<IndexedPointWithColorTuple> output;

    vector<IndexedPointWithColorTuple>::iterator bgt = gt.begin();
    vector<IndexedPointWithColorTuple>::iterator egt = gt.end();

    int count = 0;

    float x = 0, y = 0, z = 0;
    while (bgt != egt) {
        brute_force_thresh_dist_detect(*bgt, recon, thresh, detected);
        if (detected.size()) {
            for (int i = 0; i < detected.size(); i++) {
                Point c = detected[i];
                x += c[0];
                y += c[1];
                z += c[2];
            }
            x /= detected.size();
            y /= detected.size();
            z /= detected.size();
            Point p(x, y, z);
            IndexedPointWithColorTuple ipwcp = boost::make_tuple(0, p, 0, 0, 0);
            recon.push_back(ipwcp);
            x = 0, y = 0, z = 0;
            detected.clear();
        }
        bgt++;
        count++;

        float percent = float(100 * count) / gt.size();
        if (count % 100 == 0) {
            cout << percent << endl;
        }
    }
    output = recon;
    return output;
}

std::vector<IndexedPointWithColorTuple> brute_force_delete_distant_outliers(std::vector<IndexedPointWithColorTuple>& recon,
                                                                            std::vector<IndexedPointWithColorTuple> gt,
                                                                            float thresh) {
    std::vector<Point> detected;
    std::vector<IndexedPointWithColorTuple> output;

    vector<IndexedPointWithColorTuple>::iterator bgt = gt.begin();
    vector<IndexedPointWithColorTuple>::iterator egt = gt.end();

    int count = 0;

    float x = 0, y = 0, z = 0;
    while (bgt != egt) {
        brute_force_thresh_dist_detect(*bgt, recon, thresh, detected);
        if (detected.size()) {
            for (int i = 0; i < detected.size(); i++) {
                Point c = detected[i];
                IndexedPointWithColorTuple ipwcp = boost::make_tuple(0, c, 0, 0, 0);
                output.push_back(ipwcp);
            }
            detected.clear();
        }
        bgt++;
        count++;

        float percent = float(100 * count) / gt.size();
        if (count % 100 == 0) {
            cout << percent << endl;
        }
    }
    return output;
}

void find_bounding_volume(Point& min_p, Point& max_p, std::vector<IndexedPointWithColorTuple> input) {
    vector<IndexedPointWithColorTuple>::iterator binput = input.begin();
    vector<IndexedPointWithColorTuple>::iterator einput = input.end();
    float min_x = INT_MAX, min_y = INT_MAX, min_z = INT_MAX, max_x = INT_MIN, max_y = INT_MIN, max_z = INT_MIN;

    while (binput != einput) {
        Point p = get<1>(*binput);
        if (p[0] < min_x) {
            min_x = p[0];
        } else if (p[0] > max_x) {
            max_x = p[0];
        } else if (p[1] < min_y) {
            min_y = p[1];
        } else if (p[1] > max_y) {
            max_y = p[1];
        } else if (p[2] < min_z) {
            min_z = p[1];
        } else if (p[2] > max_z) {
            max_z = p[2];
        }
        binput++;
    }
    Point min(min_x, min_y, min_z);
    Point max(max_x, max_y, max_z);
    min_p = min;
    max_p = max;
}

void erase_bounding_volume_outliers(std::vector<IndexedPointWithColorTuple>& recon, Point min, Point max) {
    std::vector<IndexedPointWithColorTuple> temp;
    vector<IndexedPointWithColorTuple>::iterator binput = recon.begin();
    vector<IndexedPointWithColorTuple>::iterator einput = recon.end();

    while (binput != einput) {
        Point p = get<1>(*binput);
        if ((p[0] < min[0]) || (p[0] > max[0]) || (p[1] < min[1]) || (p[1] > max[1]) || (p[2] < min[2]) || (p[2] > max[2])) {
            binput++;
        } else {
            IndexedPointWithColorTuple ip = boost::make_tuple(0, p, 0, 0, 0);
            temp.push_back(ip);
            binput++;
        }
    }
    recon = temp;
}
