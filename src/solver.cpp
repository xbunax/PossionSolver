#include "../hdr/solver.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

using namespace Eigen;
using json = nlohmann::json;

double N1(double s, double t) { return 0.25 * (1 - s) * (1 - t); }

double N2(double s, double t) { return 0.25 * (1 + s) * (1 - t); }

double N3(double s, double t) { return 0.25 * (1 + s) * (1 + t); }

double N4(double s, double t) { return 0.25 * (1 - s) * (1 + t); }

double dN1_ds(double s, double t) { return -0.25 * (1 - t); }

double dN1_dt(double s, double t) { return -0.25 * (1 - s); }

double dN2_ds(double s, double t) { return 0.25 * (1 - t); }

double dN2_dt(double s, double t) { return -0.25 * (1 + s); }

double dN3_ds(double s, double t) { return 0.25 * (1 + t); }

double dN3_dt(double s, double t) { return 0.25 * (1 + s); }

double dN4_ds(double s, double t) { return -0.25 * (1 + t); }

double dN4_dt(double s, double t) { return 0.25 * (1 - s); }

double gauss_integrate_2d(std::function<double(double, double)> f, int n,
                          double xa, double xb, double ya, double yb) {
  std::vector<double> x = {-0.774596669241483, 0.0, 0.774596669241483};
  std::vector<double> wx = {0.555555555555556, 0.888888888888889,
                            0.555555555555556};
  double integral = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double xi = 0.5 * (xb - xa) * x[i] + 0.5 * (xa + xb);
      double eta = 0.5 * (yb - ya) * x[j] + 0.5 * (ya + yb);
      integral += wx[i] * wx[j] * f(xi, eta);
    }
  }
  return 0.25 * (xb - xa) * (yb - ya) * integral;
}

void apply_boundary_conditions(MatrixXd &K, VectorXd &F, int n_e_x, int n_e_y,
                               double u_left, double u_right, double u_top,
                               double u_bottom) {
  int n_x = n_e_x + 1;
  int n_y = n_e_y + 1;

  // 左边界 (x = 0)
  for (int j = 0; j < n_y; ++j) {
    int node = j * n_x;
    F(node) = u_left;
    K.row(node).setZero();
    K(node, node) = 1;
  }

  // 右边界 (x = lx)
  for (int j = 0; j < n_y; ++j) {
    int node = j * n_x + (n_x - 1);
    F(node) = u_right;
    K.row(node).setZero();
    K(node, node) = 1;
  }

  // 下边界 (y = 0)
  for (int i = 1; i < n_x - 1; ++i) {
    int node = i;
    F(node) = u_bottom;
    K.row(node).setZero();
    K(node, node) = 1;
  }

  // 上边界 (y = ly)
  for (int i = 1; i < n_x - 1; ++i) {
    int node = (n_y - 1) * n_x + i;
    F(node) = u_top;
    K.row(node).setZero();
    K(node, node) = 1;
  }
}

double analytical_solution(double x, double y, double lx = 1, double ly = 1,
                           double u_hat = 5) {
  double solution = 0;
  for (int n = 1; n < 100; n += 2) { // 仅取奇数项
    double term = (std::sin(n * M_PI * x / lx) * std::sinh(n * M_PI * y / lx) /
                   (n * std::sinh(n * M_PI * ly / lx)));
    solution += term;
  }
  return (4 * u_hat / M_PI) * solution;
}

void save_to_vtk(const std::string &filename, const MatrixXd &data,
                 const std::vector<double> &x, const std::vector<double> &y) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return;
  }

  // Write VTK header
  file << "# vtk DataFile Version 3.0\n";
  file << "Poisson equation solution\n";
  file << "ASCII\n";
  file << "DATASET STRUCTURED_GRID\n";

  int nx = x.size();
  int ny = y.size();
  file << "DIMENSIONS " << nx << " " << ny << " 1\n";
  file << "POINTS " << (nx * ny) << " float\n";

  // Write coordinates
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      file << x[i] << " " << y[j] << " 0\n";
    }
  }

  // Write data
  file << "POINT_DATA " << (nx * ny) << "\n";
  file << "SCALARS solution float 1\n";
  file << "LOOKUP_TABLE default\n";

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      file << data(j, i) << "\n";
    }
  }

  file.close();
}

VectorXd compute_residual(const MatrixXd &K, const VectorXd &U,
                          const VectorXd &F) {
  return K * U - F;
}

// 实现 FunctionParser 类的成员函数
FunctionParser::FunctionParser(const std::string &expr,
                               const std::vector<std::string> &vars) {
  var_names = vars;
  var_values.resize(vars.size(), 0.0);

  for (size_t i = 0; i < vars.size(); ++i) {
    parser.DefineVar(vars[i], &var_values[i]);
  }
  parser.SetExpr(expr);
}

double FunctionParser::evaluate(const std::vector<double> &values) {
  if (values.size() != var_values.size()) {
    throw std::runtime_error("Invalid number of arguments");
  }
  var_values = values;
  return parser.Eval();
}

// 实现 Config 类的静态成员函数
Config Config::from_json(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open config file: " + filename);
  }

  json j;
  try {
    file >> j;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("JSON parse error: " + std::string(e.what()));
  }

  check_required_fields(j);

  Config config;
  try {
    config.lx = j["lx"];
    config.ly = j["ly"];
    config.Nx = j["Nx"];
    config.Ny = j["Ny"];
    config.mesh = j["mesh"];
    config.guess_expr = j["guess"];
    config.source_expr = j["source"];
    config.source_derivatives_expr = j["source_derivatives"];
    config.max_iter = j["max_iter"];
    config.rel_tol = j["rel_tol"];
    config.abs_tol = j["abs_tol"];
    config.u_left = j["u_left"];
    config.u_right = j["u_right"];
    config.u_top = j["u_top"];
    config.u_bottom = j["u_bottom"];
    config.output_path = j["output_path"];
  } catch (const json::type_error &e) {
    throw std::runtime_error("Type error in config file: " +
                             std::string(e.what()));
  }

  // 验证参数值
  config.validate();

  return config;
}

// 计算非线性刚度矩阵
Matrix4d nonlinear_element_stiffness(double x0, double x1, double y0, double y1,
                                     const Vector4d &Ue,
                                     FunctionParser &source_deriv_func) {
  Matrix4d K = Matrix4d::Zero();
  std::vector<double (*)(double, double)> dN_ds = {dN1_ds, dN2_ds, dN3_ds,
                                                   dN4_ds};
  std::vector<double (*)(double, double)> dN_dt = {dN1_dt, dN2_dt, dN3_dt,
                                                   dN4_dt};
  std::vector<double (*)(double, double)> N = {N1, N2, N3, N4};

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      auto integrand = [&](double s, double t) -> double {
        Matrix2d J;
        J << 0.5 * (x1 - x0), 0, 0, 0.5 * (y1 - y0);
        double detJ = J.determinant();
        Matrix2d invJ = J.inverse();

        Vector2d dNi(dN_ds[i](s, t), dN_dt[i](s, t));
        Vector2d dNj(dN_ds[j](s, t), dN_dt[j](s, t));

        Vector2d gradNi = invJ * dNi;
        Vector2d gradNj = invJ * dNj;

        double u = Ue.dot(Vector4d(N1(s, t), N2(s, t), N3(s, t), N4(s, t)));
        // double source_deriv = -exp(-u);
        double source_deriv = source_deriv_func.evaluate({u});

        return (gradNi.dot(gradNj) + source_deriv * N[i](s, t) * N[j](s, t)) *
               detJ;
      };
      K(i, j) = gauss_integrate_2d(integrand, 3, -1, 1, -1, 1);
    }
  }
  return K;
}

// 计算非线性荷载向量
Vector4d nonlinear_element_force(FunctionParser &source_func, double x0,
                                 double x1, double y0, double y1,
                                 const Vector4d &Ue) {
  Vector4d F = Vector4d::Zero();
  std::vector<double (*)(double, double)> N = {N1, N2, N3, N4};

  for (int i = 0; i < 4; ++i) {
    auto integrand = [&](double s, double t) {
      Matrix2d J;
      J << 0.5 * (x1 - x0), 0, 0, 0.5 * (y1 - y0);
      double detJ = J.determinant();

      double u = Ue.dot(Vector4d(N1(s, t), N2(s, t), N3(s, t), N4(s, t)));

      double source_value = source_func.evaluate({u});
      // double source_value = exp(-u);

      return N[i](s, t) * source_value * detJ;
    };
    F(i) = gauss_integrate_2d(integrand, 3, -1, 1, -1, 1);
  }
  return F;
}

// 组装全局非线性矩阵
void assemble_nonlinear_global_matrices(int n_e_x, int n_e_y,
                                        const std::vector<double> &x,
                                        const std::vector<double> &y,
                                        FunctionParser &source_func,
                                        FunctionParser &source_deriv_func,
                                        const VectorXd &U, MatrixXd &K,
                                        VectorXd &F) {
  int n_x = n_e_x + 1;
  int n_n = n_x * (n_e_y + 1);
  K = MatrixXd::Zero(n_n, n_n);
  F = VectorXd::Zero(n_n);

  for (int i = 0; i < n_e_x; ++i) {
    for (int j = 0; j < n_e_y; ++j) {
      double x0 = x[i], x1 = x[i + 1];
      double y0 = y[j], y1 = y[j + 1];

      std::vector<int> nodes = {
          j * n_x + i,             // 左下
          j * n_x + (i + 1),       // 右下
          (j + 1) * n_x + (i + 1), // 右上
          (j + 1) * n_x + i        // 左上
      };

      Vector4d Ue;
      for (int a = 0; a < 4; ++a) {
        Ue(a) = U(nodes[a]);
      }

      Matrix4d Ke =
          nonlinear_element_stiffness(x0, x1, y0, y1, Ue, source_deriv_func);
      Vector4d Fe = nonlinear_element_force(source_func, x0, x1, y0, y1, Ue);

      for (int a = 0; a < 4; ++a) {
        F(nodes[a]) += Fe(a);
        for (int b = 0; b < 4; ++b) {
          K(nodes[a], nodes[b]) += Ke(a, b);
        }
      }
    }
  }
}

// 牛顿迭代求解非线性问题
VectorXd newton_solve_nonlinear(int n_e_x, int n_e_y,
                                const std::vector<double> &x,
                                const std::vector<double> &y,
                                FunctionParser &source_func,
                                FunctionParser &source_deriv_func, VectorXd &U0,
                                double u_left, double u_right, double u_top,
                                double u_bottom, double rel_tol, double abs_tol,
                                int max_iter) {
  VectorXd U = U0;
  int iter = 0;
  double residual_norm;
  double rel_error;

  MatrixXd K;
  VectorXd F;
  std::cout << "--------------------------------------------------\n";
  std::cout << "Step\tAbs. Error\t\tRel. Error\n";
  std::cout << "--------------------------------------------------\n";

  do {

    assemble_nonlinear_global_matrices(n_e_x, n_e_y, x, y, source_func,
                                       source_deriv_func, U, K, F);

    apply_boundary_conditions(K, F, n_e_x, n_e_y, u_left, u_right, u_top,
                              u_bottom);

    VectorXd F_un = compute_residual(K, U, F);

    // MatrixXd K_inv = K.inverse();
    // VectorXd dU = K_inv * (-F_un);
    VectorXd dU = K.lu().solve(-F_un);

    // 更新解 u_{n+1} = u_n + du
    VectorXd U_new = U + dU;

    // 计算相对误差
    rel_error = dU.norm() / (U.norm() + 1e-10);

    // 计算残差范数（绝对误差）
    residual_norm = F_un.norm();

    // 更新解
    U = U_new;

    // 打印当前迭代信息
    if (iter == 0) {
      std::cout << iter << "\t" << std::scientific << residual_norm
                << "\t\t-\n";
    } else {
      std::cout << iter << "\t" << std::scientific << residual_norm << "\t\t"
                << rel_error << "\n";
    }

    ++iter;

    if (iter >= max_iter) {
      std::cout << "Warning: Maximum iterations reached without convergence\n";
      break;
    }
  } while (residual_norm > abs_tol || rel_error > rel_tol);

  std::cout << "--------------------------------------------------\n";
  std::cout << "Newton method converged in " << iter << " iterations\n";
  std::cout << "Final absolute error: " << residual_norm << "\n";
  std::cout << "Final relative error: " << rel_error << "\n";
  return U;
}

// 计算数值解与解析解的误差并保存所有结果
void save_solutions_and_error(const std::string &numerical_vtk,
                              const std::string &analytical_vtk,
                              const std::string &error_vtk,
                              const VectorXd &numerical_sol,
                              const std::vector<double> &x,
                              const std::vector<double> &y, double lx,
                              double ly) {
  int nx = x.size();
  int ny = y.size();

  // 将数值解重组为矩阵形式
  MatrixXd numerical_matrix(ny, nx);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      numerical_matrix(j, i) = numerical_sol[j * nx + i];
    }
  }

  // 计算解析解
  MatrixXd analytical_matrix(ny, nx);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      analytical_matrix(j, i) = analytical_solution(x[i], y[j], lx, ly);
    }
  }

  // 计算误差
  MatrixXd error_matrix = (numerical_matrix - analytical_matrix).cwiseAbs();

  // 计算最大误差和平均误差
  double max_error = error_matrix.maxCoeff();
  double mean_error = error_matrix.mean();

  std::cout << "Error Analysis:\n";
  std::cout << "Maximum error: " << max_error << "\n";
  std::cout << "Mean error: " << mean_error << "\n";

  // 保存所有结果到VTK文件
  save_to_vtk(numerical_vtk, numerical_matrix, x, y);
  save_to_vtk(analytical_vtk, analytical_matrix, x, y);
  save_to_vtk(error_vtk, error_matrix, x, y);
}

// 实现参数验证函数
void Config::validate() const {
  // 检查几何参数
  if (lx <= 0 || ly <= 0) {
    throw std::invalid_argument("Domain dimensions (lx, ly) must be positive");
  }

  // 检查网格参数
  if (Nx <= 0 || Ny <= 0) {
    throw std::invalid_argument("Grid divisions (Nx, Ny) must be positive");
  }

  // 检查网格类型
  if (mesh != "rectangle") {
    throw std::invalid_argument(
        "Currently only 'rectangle' mesh type is supported");
  }

  // 检查表达式是否为空
  if (guess_expr.empty()) {
    throw std::invalid_argument("Initial guess expression cannot be empty");
  }
  if (source_expr.empty()) {
    throw std::invalid_argument("Source function expression cannot be empty");
  }
  if (source_derivatives_expr.empty()) {
    throw std::invalid_argument(
        "Source derivatives expression cannot be empty");
  }

  // 检查迭代参数
  if (max_iter <= 0) {
    throw std::invalid_argument("Maximum iterations must be positive");
  }
  if (rel_tol <= 0 || abs_tol <= 0) {
    throw std::invalid_argument("Tolerance values must be positive");
  }

  // 检查输出路径
  if (output_path.empty()) {
    throw std::invalid_argument("Output path cannot be empty");
  }
}

// 检查JSON中的必需字段
void Config::check_required_fields(const json &j) {
  std::vector<std::string> required_fields = {
      "lx",       "ly",      "Nx",       "Ny",
      "mesh",     "guess",   "source",   "source_derivatives",
      "max_iter", "rel_tol", "abs_tol",  "u_left",
      "u_right",  "u_top",   "u_bottom", "output_path"};

  for (const auto &field : required_fields) {
    if (!j.contains(field)) {
      throw std::invalid_argument("Missing required field in config file: " +
                                  field);
    }
  }

  // 检查字段类型
  if (!j["lx"].is_number() || !j["ly"].is_number()) {
    throw std::invalid_argument("Domain dimensions must be numbers");
  }
  if (!j["Nx"].is_number_integer() || !j["Ny"].is_number_integer()) {
    throw std::invalid_argument("Grid divisions must be integers");
  }
  if (!j["mesh"].is_string()) {
    throw std::invalid_argument("Mesh type must be a string");
  }
  if (!j["guess"].is_string() || !j["source"].is_string() ||
      !j["source_derivatives"].is_string()) {
    throw std::invalid_argument("Expressions must be strings");
  }
  if (!j["max_iter"].is_number_integer()) {
    throw std::invalid_argument("Maximum iterations must be an integer");
  }
  if (!j["rel_tol"].is_number() || !j["abs_tol"].is_number()) {
    throw std::invalid_argument("Tolerance values must be numbers");
  }
  if (!j["output_path"].is_string()) {
    throw std::invalid_argument("Output path must be a string");
  }
}
