#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Dense>
#include <functional>
#include <muParser.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace Eigen;
using json = nlohmann::json;

// 函数解析器类
class FunctionParser {
private:
  mu::Parser parser;
  std::vector<double> var_values;
  std::vector<std::string> var_names;

public:
  FunctionParser(const std::string &expr, const std::vector<std::string> &vars);
  double evaluate(const std::vector<double> &values);
};

// 配置类
struct Config {
  double lx, ly;
  int Nx, Ny;
  std::string mesh;
  std::string guess_expr;
  std::string source_expr;
  std::string source_derivatives_expr;
  double rel_tol;
  double abs_tol;
  double u_left, u_right, u_top, u_bottom;
  int max_iter;
  std::string output_path;

  static Config from_json(const std::string &filename);

  static void check_required_fields(const json &j);
  void validate() const;
};

// 组装和边界条件函数声明
double gauss_integrate_2d(std::function<double(double, double)> f, int n,
                         double xa, double xb, double ya, double yb);
                         
void assemble_nonlinear_global_matrices(int n_e_x, int n_e_y,
                                        const std::vector<double> &x,
                                        const std::vector<double> &y,
                                        FunctionParser &source_func,
                                        FunctionParser &source_deriv_func,
                                        const VectorXd &U, MatrixXd &K,
                                        VectorXd &F, const std::string& mesh_type);

void apply_boundary_conditions(MatrixXd &K, VectorXd &F, int n_e_x, int n_e_y,
                               double u_left, double u_right, double u_top,
                               double u_bottom);

// 非线性求解相关函数声明
VectorXd newton_solve_nonlinear(int n_e_x, int n_e_y,
                                const std::vector<double> &x,
                                const std::vector<double> &y,
                                FunctionParser &source_func,
                                FunctionParser &source_deriv_func, VectorXd &U0,
                                double u_left, double u_right, double u_top,
                                double u_bottom, double rel_tol, double abs_tol,
                                int max_iter, const std::string& mesh_type);

// 输出函数声明
void save_to_vtk(const std::string &filename, const MatrixXd &data,
                 const std::vector<double> &x, const std::vector<double> &y);

void save_solutions_and_error(const std::string &numerical_vtk,
                              const std::string &analytical_vtk,
                              const std::string &error_vtk,
                              const VectorXd &numerical_sol,
                              const std::vector<double> &x,
                              const std::vector<double> &y, double lx,
                              double ly);

#endif // SOLVER_H
