// ==========================================
// LassoRegression.h
// Lasso回归（L1正则化）C++ 实现 (使用坐标下降法 Coordinate Descent)
// ==========================================
#ifndef LASSO_REGRESSION_H
#define LASSO_REGRESSION_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>

// 软阈值函数 (Soft Thresholding)
inline double softThreshold(double z, double lambda) {
    if (z > lambda) return z - lambda;
    if (z < -lambda) return z + lambda;
    return 0.0;
}

class LassoRegression {
private:
    std::vector<double> coefficients; // 回归系数
    double intercept;                 // 截距
    std::vector<double> featureMeans;
    std::vector<double> featureStds;
    double labelMean;
    double labelStd;
    bool isStandardized;

    std::vector<std::vector<double>> standardizeFeatures(const std::vector<std::vector<double>>& X) {
        int n = X.size();
        int p = X[0].size();
        featureMeans.assign(p, 0.0);
        featureStds.assign(p, 0.0);
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += X[i][j];
            featureMeans[j] = sum / n;
            double sqSum = 0.0;
            for (int i = 0; i < n; i++) sqSum += std::pow(X[i][j] - featureMeans[j], 2);
            featureStds[j] = std::sqrt(sqSum / n);
            if (featureStds[j] == 0.0) featureStds[j] = 1.0;
        }
        std::vector<std::vector<double>> X_scaled(n, std::vector<double>(p));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                X_scaled[i][j] = (X[i][j] - featureMeans[j]) / featureStds[j];
            }
        }
        return X_scaled;
    }
    
    std::vector<double> standardizeLabel(const std::vector<double>& y) {
        int n = y.size();
        double sum = 0.0;
        for (double val : y) sum += val;
        labelMean = sum / n;
        double sqSum = 0.0;
        for (double val : y) sqSum += std::pow(val - labelMean, 2);
        labelStd = std::sqrt(sqSum / n);
        if (labelStd == 0.0) labelStd = 1.0;
        std::vector<double> y_scaled(n);
        for (int i = 0; i < n; i++) y_scaled[i] = (y[i] - labelMean) / labelStd;
        return y_scaled;
    }
    
    void inverseStandardizeCoefficients(int p) {
        std::vector<double> new_coeffs(p);
        for (int j = 0; j < p; j++) {
            new_coeffs[j] = coefficients[j] * labelStd / featureStds[j];
        }
        intercept = intercept * labelStd + labelMean;
        for (int j = 0; j < p; j++) intercept -= new_coeffs[j] * featureMeans[j];
        coefficients = new_coeffs;
    }

public:
    LassoRegression() : intercept(0.0), labelMean(0.0), labelStd(1.0), isStandardized(true) {}
    
    // 拟合模型 (使用坐标下降法)
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double lambda, bool normalize = true) {
        int n = X.size();
        int p = X[0].size();
        isStandardized = normalize;
        
        std::vector<std::vector<double>> X_train = X;
        std::vector<double> y_train = y;
        
        if (normalize) {
            X_train = standardizeFeatures(X);
            y_train = standardizeLabel(y);
        }
        
        // 预计算每个特征的内积 (X_j^T * X_j)
        std::vector<double> xj_sq(p, 0.0);
        for (int j = 0; j < p; j++) {
            for (int i = 0; i < n; i++) {
                xj_sq[j] += X_train[i][j] * X_train[i][j];
            }
        }
        
        coefficients.assign(p, 0.0);
        intercept = 0.0; // 标准化后截距为0
        
        double tol = 1e-6; // 收敛阈值
        int max_iter = 10000;
        
        for (int iter = 0; iter < max_iter; iter++) {
            bool converged = true;
            
            // 更新截距
            double res_sum = 0.0;
            for (int i = 0; i < n; i++) {
                double pred = 0.0;
                for (int j = 0; j < p; j++) pred += X_train[i][j] * coefficients[j];
                res_sum += (y_train[i] - pred);
            }
            intercept += res_sum / n; // 标准化数据中截距初始为0，此步可省略但保留更稳健
            
            double max_diff = 0.0;
            
            // 坐标下降更新每个系数
            for (int j = 0; j < p; j++) {
                double rho_j = 0.0;
                for (int i = 0; i < n; i++) {
                    double residual_i = y_train[i] - intercept;
                    for (int k = 0; k < p; k++) {
                        if (k != j) residual_i -= X_train[i][k] * coefficients[k];
                    }
                    rho_j += X_train[i][j] * residual_i;
                }
                
                double old_coef = coefficients[j];
                coefficients[j] = softThreshold(rho_j, n * lambda) / xj_sq[j];
                
                if (std::abs(coefficients[j] - old_coef) > max_diff) {
                    max_diff = std::abs(coefficients[j] - old_coef);
                }
            }
            
            if (max_diff < tol) break;
        }
        
        if (normalize) {
            inverseStandardizeCoefficients(p);
        } else {
            double sum_y = 0, sum_xb = 0;
            for (int i = 0; i < n; i++) {
                sum_y += y[i];
                double pred = 0;
                for (int j = 0; j < p; j++) pred += X[i][j] * coefficients[j];
                sum_xb += pred;
            }
            intercept = (sum_y - sum_xb) / n;
        }
    }
    
    std::vector<double> predict(const std::vector<std::vector<double>>& X) {
        int n = X.size();
        int p = coefficients.size();
        std::vector<double> y_pred(n);
        for (int i = 0; i < n; i++) {
            y_pred[i] = intercept;
            for (int j = 0; j < p; j++) y_pred[i] += X[i][j] * coefficients[j];
        }
        return y_pred;
    }
    
    const std::vector<double>& getCoefficients() const { return coefficients; }
    double getIntercept() const { return intercept; }
};

#endif // LASSO_REGRESSION_H