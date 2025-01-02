#include "../hdr/solver.h"
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input.json>" << std::endl;
    return 1;
  }

  try {
    // 读取并验证配置
    Config config = Config::from_json(argv[1]);

    // 创建函数解析器
    try {
      FunctionParser guess_func(config.guess_expr, {"x", "y"});
      FunctionParser source_func(config.source_expr, {"u"});
      FunctionParser source_deriv_func(config.source_derivatives_expr, {"u"});

      // 设置网格
      VectorXd x_vec = VectorXd::LinSpaced(config.Nx + 1, 0, config.lx);
      VectorXd y_vec = VectorXd::LinSpaced(config.Ny + 1, 0, config.ly);
      std::vector<double> x(x_vec.data(), x_vec.data() + x_vec.size());
      std::vector<double> y(y_vec.data(), y_vec.data() + y_vec.size());

      // 设置初始猜测解
      int n_x = config.Nx + 1;
      int n_y = config.Ny + 1;
      VectorXd U0 = VectorXd::Zero(n_x * n_y);

      for (int j = 0; j < n_y; ++j) {
        for (int i = 0; i < n_x; ++i) {
          int node = j * n_x + i;
          U0(node) = guess_func.evaluate({x[i], y[j]});
        }
      }

      VectorXd U = newton_solve_nonlinear(
          config.Nx, config.Ny, x, y, source_func, source_deriv_func, U0,
          config.u_left, config.u_right, config.u_top, config.u_bottom,
          config.rel_tol, config.abs_tol, config.max_iter);
      std::string numerical_vtk = "numerical_solution.vtk";
      std::string analytical_vtk = "analytical_solution.vtk";
      std::string error_vtk = "error.vtk";

      // 保存所有结果并计算误差
      // save_solutions_and_error(numerical_vtk, analytical_vtk, error_vtk, U, x, y,
      //                          config.lx, config.ly);
      MatrixXd res(n_y, n_x);
      for (int j = 0; j < n_y; ++j) {
        for (int i = 0; i < n_x; ++i) {
          int node = j * n_x + i;
          res(j, i) = U(node);
        }
      }

      // 保存结果
      save_to_vtk(config.output_path, res, x, y);

      std::cout << "Solution has been saved to: " << config.output_path
                << std::endl;

    } catch (const mu::Parser::exception_type &e) {
      std::cerr << "Error in function expression: " << e.GetMsg() << std::endl;
      return 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
