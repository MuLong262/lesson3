#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

// ---------- CSV 简单解析 ----------

vector<string> splitCSV(const string &line) {
    vector<string> cols;
    string cur;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); i++) {
        char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur += '"';
                i++;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            cols.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    cols.push_back(cur);
    return cols;
}

string trim(const string &s) {
    size_t l = 0, r = s.size();
    while (l < r && (s[l] == ' ' || s[l] == '\t' || s[l] == '\r' || s[l] == '\n')) l++;
    while (r > l && (s[r - 1] == ' ' || s[r - 1] == '\t' || s[r - 1] == '\r' || s[r - 1] == '\n')) r--;
    return s.substr(l, r - l);
}

int findCol(const vector<string> &header, const string &name) {
    for (int i = 0; i < (int)header.size(); i++) {
        if (trim(header[i]) == name) return i;
    }
    return -1;
}

// ---------- 标准化 ----------

struct Standardizer {
    vector<double> mean;
    vector<double> std;
};

Standardizer fitStandardizer(const vector<vector<double>> &X) {
    int n = X.size();
    int p = X[0].size();
    Standardizer st;
    st.mean.assign(p, 0.0);
    st.std.assign(p, 0.0);

    for (int j = 0; j < p; j++) {
        double sum = 0;
        for (int i = 0; i < n; i++) sum += X[i][j];
        st.mean[j] = sum / n;
    }

    for (int j = 0; j < p; j++) {
        double sq = 0;
        for (int i = 0; i < n; i++) {
            double d = X[i][j] - st.mean[j];
            sq += d * d;
        }
        st.std[j] = sqrt(sq / n);
        if (st.std[j] < 1e-12) st.std[j] = 1.0;
    }
    return st;
}

vector<vector<double>> standardize(const vector<vector<double>> &X, const Standardizer &st) {
    int n = X.size();
    int p = X[0].size();
    vector<vector<double>> Z(n, vector<double>(p));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            Z[i][j] = (X[i][j] - st.mean[j]) / st.std[j];
    return Z;
}

// ---------- Elastic Net：坐标下降法 ----------

inline double softThreshold(double z, double t) {
    if (z > t) return z - t;
    if (z < -t) return z + t;
    return 0.0;
}

struct ElasticNetModel {
    vector<double> coef;
    double intercept = 0.0;
};

ElasticNetModel fitElasticNet(const vector<vector<double>> &X, const vector<double> &y,
                              double alpha, double l1Ratio,
                              int maxIter = 5000, double tol = 1e-6) {
    int n = X.size();
    int p = X[0].size();

    vector<double> beta(p, 0.0);
    double intercept = 0.0;

    double yMean = 0.0;
    for (double v : y) yMean += v;
    yMean /= n;

    vector<double> xjSq(p, 0.0);
    for (int j = 0; j < p; j++)
        for (int i = 0; i < n; i++)
            xjSq[j] += X[i][j] * X[i][j];

    for (int iter = 0; iter < maxIter; iter++) {
        double maxChange = 0.0;

        for (int j = 0; j < p; j++) {
            double rho = 0.0;
            for (int i = 0; i < n; i++) {
                double residual = y[i] - intercept;
                for (int k = 0; k < p; k++) {
                    if (k != j) residual -= X[i][k] * beta[k];
                }
                rho += X[i][j] * residual;
            }

            double old = beta[j];
            double denom = xjSq[j] + 1e-12;

            double l1Penalty = alpha * l1Ratio * n;
            double l2Denom = denom + alpha * (1.0 - l1Ratio) * n;

            beta[j] = softThreshold(rho, l1Penalty) / l2Denom;
            maxChange = max(maxChange, fabs(beta[j] - old));
        }

        // 更新截距
        double resSum = 0.0;
        for (int i = 0; i < n; i++) {
            double pred = intercept;
            for (int j = 0; j < p; j++) pred += X[i][j] * beta[j];
            resSum += y[i] - pred;
        }
        intercept += resSum / n;

        if (maxChange < tol) break;
    }

    ElasticNetModel model;
    model.coef = beta;
    model.intercept = intercept;
    return model;
}

// ---------- 主程序 ----------

int main() {
    string path = "netflix_titles.csv";
    ifstream fin(path);
    if (!fin.is_open()) {
        cerr << "Cannot open file: " << path << "\n"
             << "Please put netflix_titles.csv in the same directory.\n";
        return 1;
    }

    string headerLine;
    if (!getline(fin, headerLine)) return 1;

    vector<string> header = splitCSV(headerLine);

    int idxType = findCol(header, "type");
    int idxCountry = findCol(header, "country");
    int idxRating = findCol(header, "rating");
    int idxRelease = findCol(header, "release_year");
    int idxDuration = findCol(header, "duration");

    if (idxRelease < 0 || idxDuration < 0) {
        cerr << "Required columns not found.\n";
        return 1;
    }

    vector<vector<double>> rawX;
    vector<double> rawY;

    string line;
    while (getline(fin, line)) {
        vector<string> cols = splitCSV(line);
        if ((int)cols.size() <= max({idxType, idxCountry, idxRating, idxRelease, idxDuration}))
            continue;

        string type = trim(cols[idxType]);
        string country = trim(cols[idxCountry]);
        string rating = trim(cols[idxRating]);
        string releaseStr = trim(cols[idxRelease]);
        string durationStr = trim(cols[idxDuration]);

        if (releaseStr.empty() || durationStr.empty()) continue;

        double releaseYear = stod(releaseStr);

        // duration 形如 "90 min" / "2 Seasons"
        double duration = 0.0;
        bool validDuration = false;
        istringstream ds(durationStr);
        if (ds >> duration) validDuration = true;
        if (!validDuration) continue;

        double isMovie = (type == "Movie") ? 1.0 : 0.0;
        double isUS = (country.find("United States") != string::npos) ? 1.0 : 0.0;
        double isTVMA = (rating == "TV-MA") ? 1.0 : 0.0;

        rawX.push_back({releaseYear, isMovie, isUS, isTVMA});
        rawY.push_back(duration);
    }
    fin.close();

    if (rawX.empty()) {
        cerr << "No valid rows.\n";
        return 1;
    }

    // 标准化特征
    Standardizer stX = fitStandardizer(rawX);
    vector<vector<double>> X = standardize(rawX, stX);

    // 标准化标签
    double yMean = 0.0, yStd = 1.0;
    for (double v : rawY) yMean += v;
    yMean /= rawY.size();

    double sq = 0.0;
    for (double v : rawY) sq += (v - yMean) * (v - yMean);
    yStd = sqrt(sq / rawY.size());
    if (yStd < 1e-12) yStd = 1.0;

    vector<double> y(rawY.size());
    for (size_t i = 0; i < rawY.size(); i++)
        y[i] = (rawY[i] - yMean) / yStd;

    // Elastic Net 超参数
    double alpha = 0.1;   // 正则化强度
    double l1Ratio = 0.5; // 0.5 表示 L1 和 L2 各占一半

    ElasticNetModel model = fitElasticNet(X, y, alpha, l1Ratio);

    // 还原到原始尺度
    vector<double> origCoef(4);
    vector<string> names = {"release_year", "is_movie", "is_us", "is_tv_ma"};

    for (int j = 0; j < 4; j++)
        origCoef[j] = model.coef[j] * yStd / stX.std[j];

    double origIntercept = model.intercept * yStd + yMean;
    for (int j = 0; j < 4; j++)
        origIntercept -= origCoef[j] * stX.mean[j];

    cout << "Elastic Net on Netflix Titles Dataset\n";
    cout << "Samples: " << rawX.size() << "\n";
    cout << "alpha=" << alpha << ", l1_ratio=" << l1Ratio << "\n\n";

    for (int j = 0; j < 4; j++)
        cout << names[j] << ": " << fixed << setprecision(4) << origCoef[j] << "\n";

    cout << "intercept: " << origIntercept << "\n\n";

    // 简单评估 MSE / RMSE
    double mse = 0.0;
    for (size_t i = 0; i < rawX.size(); i++) {
        double pred = origIntercept;
        for (int j = 0; j < 4; j++) pred += rawX[i][j] * origCoef[j];
        double err = pred - rawY[i];
        mse += err * err;
    }
    mse /= rawX.size();

    cout << "Train MSE:  " << fixed << setprecision(4) << mse << "\n";
    cout << "Train RMSE: " << sqrt(mse) << "\n";
    return 0;
}