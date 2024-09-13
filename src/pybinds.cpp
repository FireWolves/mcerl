#include "env.hpp"
#include "grid_map.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;
namespace pybind11
{
namespace detail
{
/**
 * @brief 实现 cv::Point 和 tuple(x,y) 之间的转换。 copied from github
 *
 */
template <> struct type_caster<cv::Point>
{
  //! 定义该类型名为 tuple_xy, 并声明类型为 cv::Point 的局部变量 value。
  PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

  //! 步骤1：从 Python 转换到 C++。
  //! 将 Python tuple 对象转换为 C++ cv::Point 类型, 转换失败则返回 false。
  //! 其中参数2表示是否隐式类型转换。
  bool load(handle obj, bool)
  {
    // 确保传参是 tuple 类型
    if (!py::isinstance<py::tuple>(obj))
    {
      std::logic_error give_me_a_name("Point(x,y) should be a tuple!");
      return false;
    }

    // 从 handle 提取 tuple 对象，确保其长度是2。
    py::tuple pt = reinterpret_borrow<py::tuple>(obj);
    if (pt.size() != 2)
    {
      std::logic_error give_me_a_name("Point(x,y) tuple should be size of 2");
      return false;
    }

    //! 将长度为2的 tuple 转换为 cv::Point。
    value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
    return true;
  }

  //! 步骤2： 从 C++ 转换到 Python。
  //! 将 C++ cv::Mat 对象转换到 tuple，参数2和参数3常忽略。
  static handle cast(const cv::Point &pt, return_value_policy, handle) { return py::make_tuple(pt.x, pt.y).release(); }
};

} // namespace detail
} // namespace pybind11
/**
 * @brief Construct environment to pybind11 module object
 *
 */
PYBIND11_MODULE(_core, env)
{
  py::class_<Env::Environment>(env, "Environment")
      .def(py::init<int, int, int, int, int, int, int, int, float>())
      .def("reset", &Env::Environment::reset)
      .def("step", &Env::Environment::step)
      .def("done", &Env::Environment::done)
      .def("env_map", &Env::Environment::env_map)
      .def("global_map", &Env::Environment::global_map)
      .def("agent_map", &Env::Environment::agent_map)
      .def("test_map_update", &Env::Environment::test_map_update)
      .def("test_frontier_detection", &Env::Environment::test_frontier_detection)
      .def("test_a_star", &Env::Environment::test_a_star)
      .def("test_xy_coord", &Env::Environment::test_xy_coord)
      .def("test_xy_cv_mat", &Env::Environment::test_xy_cv_mat);

  py::class_<Env::GridMap>(env, "GridMap", py::buffer_protocol())
      .def_buffer([](Env::GridMap &m) -> py::buffer_info {
        return py::buffer_info(m.data(),                                 /* Pointer to buffer */
                               sizeof(uint8_t),                          /* Size of one scalar */
                               py::format_descriptor<uint8_t>::format(), /* Python struct-style format descriptor */
                               2,                                        /* Number of dimensions */
                               {m.rows(), m.cols()},                     /* Buffer dimensions */
                               {sizeof(uint8_t) * m.cols(),              /* Strides (in bytes) for each index */
                                sizeof(uint8_t)});
      })
      .def(py::init([](py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<uint8_t>::format())
          throw std::runtime_error("Incompatible format: expected a uint8_t array!");
        if (info.ndim != 2)
          throw std::runtime_error("Incompatible buffer dimension!");
        return Env::GridMap((uint8_t *)info.ptr, info.shape[1], info.shape[0]);
      }));

  py::class_<Env::Observation>(env, "Observation")
      .def(py::init<>())
      .def_readonly("frontier_points", &Env::Observation::frontier_points)
      .def_readonly("agent_poses", &Env::Observation::agent_poses)
      .def_readonly("agent_targets", &Env::Observation::agent_targets);

  py::class_<Env::FrontierPoint>(env, "FrontierPoint")
      .def(py::init<>())
      .def_readonly("pos", &Env::FrontierPoint::pos)
      .def_readonly("unexplored_pixels", &Env::FrontierPoint::unexplored_pixels)
      .def_readonly("distance", &Env::FrontierPoint::distance);

  py::class_<Env::Info>(env, "info")
      .def(py::init<>())
      .def_readonly("agent_id", &Env::Info::agent_id)
      .def_readonly("step_cnt", &Env::Info::step_cnt)
      .def_readonly("agent_step_cnt", &Env::Info::agent_step_cnt)
      .def_readonly("delta_time", &Env::Info::delta_time)
      .def_readonly("agent_exploration_rate", &Env::Info::agent_exploration_rate)
      .def_readonly("global_exploration_rate", &Env::Info::global_exploration_rate)
      .def_readonly("agent_explored_pixels", &Env::Info::agent_explored_pixels);

  py::class_<Env::Reward>(env, "Reward")
      .def(py::init<>())
      .def_readonly("exploration_reward", &Env::Reward::exploration_reward)
      .def_readonly("time_step_reward", &Env::Reward::time_step_reward);

#ifdef VERSION_INFO
  env.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}