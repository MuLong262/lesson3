#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm> // 必须包含，用于 std::swap

using namespace std;

// ==========================================
// 数学工具函数 (高斯消元法)
// ==========================================

// 使用高斯消元法求解线性方程组 Ax = b
std::vector<double> solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& b, int n) {
    std::vector<double> x(n, 0.0);
    
    for (int i = 0; i < n; i++) {
        // 部分主元选取，提高数值稳定性
        int maxIdx = i;
        for (int j = i + 1; j < n; j++) {
            if (abs(A[j][i]) > abs(A[maxIdx][i])) {
                maxIdx = j;
            }
        }
        if (maxIdx != i) {
            swap(A[i], A[maxIdx]);
            swap(b[i], b[maxIdx]);
        }
        
        double pivot = A[i][i];
        if (abs(pivot) < 1e-10) continue; 

        // 归一化主元行
        for (int j = i; j < n; j++) {
            A[i][j] /= pivot;
        }
        b[i] /= pivot;
        
        // 消去其他行
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = i; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
        x[i] = b[i];
    }
    return x;
}


// ==========================================
// Lasso回归模型类
// ==========================================

// 软阈值函数 (Soft Thresholding)
inline double softThreshold(double z, double lambda) {
    if (z > lambda) return z - lambda;
    if (z < -lambda) return z + lambda;
    return 0.0;
}

class LassoRegression {
private:
    std::vector<double> coefficients; 
    double intercept;                 

    // Z-score 标准化特征矩阵
    void standardizeFeatures(std::vector<std::vector<double>>& X, std::vector<double>& means, std::vector<double>& stds) {
        int n = X.size();
        if (n == 0) return;
        int p = X[0].size();
        
        means.assign(p, 0.0);
        stds.assign(p, 0.0);
        
        // 1. 计算均值
        for (int j = 0; j < p; j++) {
            for (int i = 0; i < n; i++) {
                means[j] += X[i][j];
            }
            means[j] /= n;
        }
        
        // 2. 计算标准差并标准化
        for (int j = 0; j < p; j++) {
            double sqSum = 0.0;
            for (int i = 0; i < n; i++) {
                sqSum += pow(X[i][j] - means[j], 2);
            }
            stds[j] = sqrt(sqSum / n);
            if (stds[j] == 0.0) stds[j] = 1.0; 

            // 原地标准化
            for (int i = 0; i < n; i++) {
                X[i][j] = (X[i][j] - means[j]) / stds[j];
            }
        }
    }

    // Z-score 标准化标签
    void standardizeLabel(std::vector<double>& y, double& mean, double& std) {
        int n = y.size();
        if (n == 0) return;
        
        double sum = 0.0;
        for (double val : y) sum += val;
        mean = sum / n;
        
        double sqSum = 0.0;
        for (double val : y) sqSum += pow(val - mean, 2);
        std = sqrt(sqSum / n);
        if (std == 0.0) std = 1.0;
        
        for (double& val : y) {
            val = (val - mean) / std;
        }
    }

    // 反标准化系数，还原到原始尺度
    void inverseStandardizeCoefficients(const std::vector<double>& means, const std::vector<double>& stds, double y_mean, double y_std) {
        int p = coefficients.size();
        std::vector<double> new_coeffs(p);
        
        for (int j = 0; j < p; j++) {
            new_coeffs[j] = coefficients[j] * (y_std / stds[j]);
        }
        
        intercept = intercept * y_std + y_mean;
        for (int j = 0; j < p; j++) {
            intercept -= new_coeffs[j] * means[j];
        }
        
        coefficients = new_coeffs;
    }

public:
    LassoRegression() : intercept(0.0) {}

    // 拟合模型 (坐标下降法 Coordinate Descent)
    void fit(const std::vector<std::vector<double>>& X_orig, const std::vector<double>& y_orig, double lambda, bool normalize = true) {
        int n = X_orig.size();
        int p = X_orig[0].size();
        
        std::vector<std::vector<double>> X = X_orig;
        std::vector<double> y = y_orig;
        
        double y_mean = 0.0, y_std = 1.0;
        std::vector<double> x_means, x_stds;

        if (normalize) {
            standardizeFeatures(X, x_means, x_stds);
            standardizeLabel(y, y_mean, y_std);
        }
        
        // 预计算每个特征的平方和 (X_j^T * X_j)
        std::vector<double> xj_sq(p, 0.0);
        for (int j = 0; j < p; j++) {
            for (int i = 0; i < n; i++) {
                xj_sq[j] += X[i][j] * X[i][j];
            }
        }
        
        coefficients.assign(p, 0.0);
        intercept = 0.0; 
        
        double tol = 1e-6; // 收敛阈值
        int max_iter = 10000;
        
        for (int iter = 0; iter < max_iter; iter++) {
            bool converged = true;
            
            // 更新截距 (在标准化数据上，截距为0，此处为了代码完整性保留)
            if (!normalize) {
                double res_sum = 0.0;
                for (int i = 0; i < n; i++) {
                    double pred = intercept;
                    for (int j = 0; j < p; j++) pred += X[i][j] * coefficients[j];
                    res_sum += (y[i] - pred);
                }
                intercept += res_sum / n; 
            }
            
            double max_diff = 0.0;
            
            // 坐标下降更新每个系数
            for (int j = 0; j < p; j++) {
                // 计算残差向量 R_j = y - intercept - sum_{k!=j} X_k * beta_k
                double rho_j = 0.0;
                for (int i = 0; i < n; i++) {
                    double residual_i = y[i] - intercept;
                    for (int k = 0; k < p; k++) {
                        if (k != j) residual_i -= X[i][k] * coefficients[k];
                    }
                    rho_j += X[i][j] * residual_i;
                }
                
                double old_coef = coefficients[j];
                // 核心步骤：软阈值操作
                coefficients[j] = softThreshold(rho_j, n * lambda) / xj_sq[j];
                
                if (std::abs(coefficients[j] - old_coef) > max_diff) {
                    max_diff = std::abs(coefficients[j] - old_coef);
                }
            }
            
            if (max_diff < tol) break;
        }
        
        // 反标准化
        if (normalize) {
            inverseStandardizeCoefficients(x_means, x_stds, y_mean, y_std);
        } else {
            // 若不标准化，计算截距
            double sum_y = 0, sum_xb = 0;
            for (int i = 0; i < n; i++) {
                sum_y += y_orig[i];
                double pred = 0;
                for (int j = 0; j < p; j++) pred += X_orig[i][j] * coefficients[j];
                sum_xb += pred;
            }
            intercept = (sum_y - sum_xb) / n;
        }
    }
    
    // 预测
    std::vector<double> predict(const std::vector<std::vector<double>>& X) {
        int n = X.size();
        int p = coefficients.size();
        std::vector<double> y_pred(n, intercept);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                y_pred[i] += X[i][j] * coefficients[j];
            }
        }
        return y_pred;
    }
    
    const std::vector<double>& getCoefficients() const { return coefficients; }
    double getIntercept() const { return intercept; }
};

// ==========================================
// 主函数
// ==========================================
int main() {
    // 构造模拟数据: y = 2*x1 + 3*x2 - 1*x3 + 5 + noise
    vector<vector<double>> X = {
        {1.0, 2.0, 3.0},
        {2.0, 3.0, 1.0},
        {3.0, 1.0, 2.0},
        {4.0, 5.0, 4.0},
        {5.0, 4.0, 5.0},
        {2.5, 3.5, 2.5},
        {3.5, 2.5, 3.5},
        {1.5, 4.5, 1.5},
        {4.5, 1.5, 4.5},
        {3.0, 3.0, 3.0}
    };
    vector<double> y = {14.5, 12.1, 13.2, 27.5, 27.0, 17.5, 18.2, 17.0, 19.5, 17.5};

    // 训练 Lasso 回归
    cout << "--- Lasso Regression ---" << endl;
    LassoRegression lasso;
    lasso.fit(X, y, 0.5); // lambda = 0.5 (Lasso通常对lambda较敏感，可以调小观察系数是否变稀疏)
    
    vector<double> lasso_coeffs = lasso.getCoefficients();
    for (int i = 0; i < lasso_coeffs.size(); i++) {
        cout << "Coefficient " << i << ": " << fixed << setprecision(4) << lasso_coeffs[i] << endl;
    }
    cout << "Intercept: " << fixed << setprecision(4) << lasso.getIntercept() << endl;
    
    // 预测并打印结果
    vector<double> y_pred_lasso = lasso.predict(X);
    cout << "\nPredictions: ";
    for (double val : y_pred_lasso) {
        cout << fixed << setprecision(4) << val << " ";
    }
    cout << endl;

    return 0;
}