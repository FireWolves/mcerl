#include "algorithms.hpp"
#include <boost/container_hash/hash.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <spdlog/spdlog.h>
#include <unordered_set>
namespace Alg
{
using namespace Env;
struct Node
{
  Coord pos;
  const Node *parent = nullptr;
  int f = 0, g = 0, h = 0;
  friend std::size_t hash_value(Node const &node)
  {
    std::size_t seed = 0;

    boost::hash_combine(seed, node.pos.x);
    boost::hash_combine(seed, node.pos.y);

    return seed;
  }
  bool operator==(const Node &rhs) const { return pos == rhs.pos; }
  bool operator<(const Node &rhs) const { return f < rhs.f; }
};
inline int sgn(const int &x)
{
  if (x > 0)
    return 1;
  if (x < 0)
    return -1;
  return 0;
};
cv::Point expand_point(cv::Point point, int rows, int cols, cv::Point origin, int expand_pixel)
{
  auto res = point + cv::Point{sgn((point.x - origin.x)), sgn((point.y - origin.y))} * expand_pixel;
  res.x = std::clamp(res.x, 0, cols - 1);
  res.y = std::clamp(res.y, 0, rows - 1);
  return res;
}
int calculate_surrounding_unexplored_pixels(Env::GridMap *exploration_map, Coord pos, int range)
{
  int rows = exploration_map->rows(), cols = exploration_map->cols();
  auto &&grid_mat = cv::Mat(rows, cols, CV_8UC1, exploration_map->data());

  int unexplored_pixels = 0;
  for (int col = pos.x - range; col <= pos.x + range; col++)
  {
    for (int row = pos.y - range; row <= pos.y + range; row++)
    {
      if (col < 0 || col >= cols || row < 0 || row >= rows)
        continue;
      if (grid_mat.at<uint8_t>(row, col) > 50 && grid_mat.at<uint8_t>(row, col) < 150)
        unexplored_pixels++;
    }
  }
  return unexplored_pixels;
}
void map_update(std::shared_ptr<GridMap> env_map, GridMap *exploration_map, Coord pos, int sensor_range, int num_rays,
                int expand_pixels)
{
  // we need to expand the pixels to avoid the ray cast error
  sensor_range = std::max(0, sensor_range - expand_pixels);

  std::default_random_engine random_engine;
  std::uniform_real_distribution<float> random_offset;

  auto &&mat = cv::Mat(env_map->rows(), env_map->cols(), CV_8UC1, env_map->data());
  auto &&mat_update = cv::Mat(exploration_map->rows(), exploration_map->cols(), CV_8UC1, exploration_map->data());

  /*** 计算需要进行光追的像素点 ***/
  std::vector<cv::Point> circle_end_points;
  for (int i = 0; i < num_rays; ++i)
  {
    auto angle = i * 2 * M_PI / num_rays + random_offset(random_engine);
    cv::Point ray_end_point =
        pos + cv::Point{static_cast<int>(cos(angle) * sensor_range), static_cast<int>(sin(angle) * sensor_range)};

    // 如果和上一个栅格相同则不予重复记录
    if (!circle_end_points.empty() &&
        (ray_end_point == circle_end_points.back() || ray_end_point == circle_end_points.front()))
      continue;
    circle_end_points.push_back(ray_end_point);
  }
  /*** 计算光追碰撞 ***/
  std::vector<cv::Point> polygon(circle_end_points);
  for (int i = 0; i < circle_end_points.size(); i++)
  {
    cv::LineIterator it(mat, pos, circle_end_points[i], 8);
    for (int j = 0; j < it.count; j++, ++it)
    {
      auto &&current_point = mat.at<uint8_t>(it.pos());
      polygon[i] = expand_point(it.pos(), mat.rows, mat.cols, pos, expand_pixels);
      if (current_point == OCCUPIED)
      {
        break;
      }
    }
  }
  /*** 计算多边形的最小外接矩形 ***/
  auto polygon_bbx = cv::boundingRect(polygon);
  /*** 计算多边形在最小外接矩形中的位置 ***/
  std::vector<cv::Point> polygon_in_roi(polygon.size());
  for (size_t i = 0; i < polygon.size(); ++i)
    polygon_in_roi[i] = polygon.at(i) - polygon_bbx.tl();
  /*** 计算多边形的ROI和需要填充的部分 ***/
  cv::Mat roi_mask(polygon_bbx.size(), CV_8UC1, cv::Scalar(0));
  cv::fillPoly(roi_mask, std::vector<std::vector<cv::Point>>{polygon_in_roi}, 255);
  cv::copyTo(mat(polygon_bbx), mat_update(polygon_bbx), roi_mask);
}

std::vector<FrontierPoint> frontier_detection(Env::GridMap *exploration_map, int min_pixels, int max_pixels,
                                              int sensor_range)
{
  std::vector<FrontierPoint> frontiers;
  /***********************************
   * 分离已探索空间和障碍物空间
   ***********************************/

  int rows = exploration_map->rows(), cols = exploration_map->cols();
  auto &&grid_mat = cv::Mat(rows, cols, CV_8UC1, exploration_map->data());

  auto &&explored = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
  auto &&obstacles = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(0));

  for (int col = 0; col < cols; col++)
  {
    for (int row = 0; row < rows; row++)
    {
      auto &raw_value = grid_mat.at<uint8_t>(row, col);

      // 分离已探索空间; 已探索区域为高亮
      if (raw_value > 150)
        explored.at<uint8_t>(row, col) = 0;

      // 分离障碍物空间; 障碍物区域为高亮
      if (raw_value < 100)
        obstacles.at<uint8_t>(row, col) = 255;
    }
  }

  /***********************************
   * 识别未探索区域并提取区域初始的边界
   ***********************************/

  // 获取各个区域的轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(explored, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  for (int i = 0; i < contours.size(); ++i)
  {
    auto &&contour = contours[i];
    // 按照可达性和最大像素数区分边界
    std::vector<cv::Point> frontier_points;
    cv::Point sum;
    for (auto itr = contour.begin(); itr != contour.end(); itr++)
    {
      auto &point = *itr;

      // 如果是障碍物或者边界点数已达标
      if (obstacles.at<uint8_t>(point) != 0 || frontier_points.size() >= max_pixels)
        goto SPLIT_FRONTIER;

      // 如果是图像边框, 则截断当前边界
      if (point.x == 0 || point.x + 1 == cols || point.y == 0 || point.y + 1 == rows)
        goto SPLIT_FRONTIER;

      // 如果是最后一个像素点
      if (itr == --contour.end())
        goto SPLIT_FRONTIER;

      sum += point;
      frontier_points.emplace_back(point);
      continue;

    SPLIT_FRONTIER:
      // 忽略像素点数目很少的边界
      if (frontier_points.size() >= min_pixels)
      {
        // // 记录新边界
        // Frontier frontier;
        auto center = cv::Point{static_cast<int>(sum.x / frontier_points.size()),
                                static_cast<int>(sum.y / frontier_points.size())};
        auto head = cv::Point{frontier_points.front().x, frontier_points.front().y};
        auto tail = cv::Point{frontier_points.back().x, frontier_points.back().y};
        auto unexplored_pixels = calculate_surrounding_unexplored_pixels(exploration_map, center, sensor_range);
        frontiers.push_back({center, unexplored_pixels});
      }

      frontier_points.clear();
      sum.x = sum.y = 0;
    }
  }
  return frontiers;
}
std::vector<Coord> DIRECTIONS = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {-1, -1}, {1, 1}, {-1, 1}, {1, -1}};

Coord get_surrogate_target(Coord target, int range, GridMap *exploration_map)
{
  // check surrounding pixels in range
  for (int x_offs = 0; x_offs < range; ++x_offs)
  {
    for (int y_offs = 0; y_offs < range; ++y_offs)
    {
      for (int i = 0; i < 8; i++)
      {
        auto &&next_pos = target + Coord{DIRECTIONS[i].x * x_offs, DIRECTIONS[i].y * y_offs};
        if (next_pos.x < 0 || next_pos.x >= exploration_map->cols() || next_pos.y < 0 ||
            next_pos.y >= exploration_map->rows())
          continue;
        if ((*exploration_map)(next_pos.x, next_pos.y) == FREE)
        {
          target = next_pos;
          return target;
        }
      }
    }
  }
  return target;
}

Path direct_path(Coord start, Coord end)
{
  Path path;
  int dx = end.x - start.x;
  int dy = end.y - start.y;
  int steps = std::max(std::abs(dx), std::abs(dy));
  int ortho_steps = std::min(std::abs(dx), std::abs(dy));
  for (int i = 0; i < ortho_steps; i++)
  {
    path.push_back({start.x + i * sgn(dx), start.y + i * sgn(dy)});
  }
  for (int i = ortho_steps; i < steps; i++)
  {
    if (std::abs(dx) > std::abs(dy))
      path.push_back({start.x + i * sgn(dx), start.y + ortho_steps * sgn(dy)});
    else
      path.push_back({start.x + ortho_steps * sgn(dx), start.y + i * sgn(dy)});
  }
  path.push_back(end);
  return path;
}

Path a_star(GridMap *exploration_map, Coord start, Coord end, int tolerance_range)
{
  end = get_surrogate_target(end, tolerance_range, exploration_map);
  spdlog::debug("start: ({}, {}), end: ({}, {})", start.x, start.y, end.x, end.y);
  if ((*exploration_map)(end.x, end.y) != FREE)
  {
    spdlog::warn("Target is not reachable");
    return {};
  }
  if (cv::norm(start - end) < tolerance_range)
  {
    spdlog::debug("direct path. ");
    auto path = direct_path(start, end);
    path.insert(path.begin(), start);
    path.push_back(end);
    return path;
  }

  // open_set: 未访问的节点 close_set: 已访问的节点
  std::unordered_set<Node, boost::hash<Node>> open_set, close_set;

  // 初始化起始节点
  Node start_node{start, nullptr};
  open_set.emplace(start_node);
  // 开始搜索
  while (!open_set.empty())
  {
    // 选择f值最小的节点
    auto current_node_itr = std::min_element(open_set.begin(), open_set.end());
    // 将当前节点加入已访问节点
    Node current_node = *current_node_itr;
    auto [itr, added] = close_set.insert(current_node);
    if (!added)
    {
      continue;
    }
    open_set.erase(current_node_itr);
    auto current_node_ptr = &(*itr);
    // 如果当前节点是目标节点, 则返回路径
    if (auto node_ptr = current_node_ptr; cv::norm(current_node_ptr->pos - end) < tolerance_range)
    {
      spdlog::trace("Path found, reconstructing path");
      Path path;
      while (node_ptr->parent != nullptr)
      {
        path.push_back(node_ptr->pos);
        node_ptr = node_ptr->parent;
      }
      spdlog::trace("reversing path");

      std::reverse(path.begin(), path.end());
      std::stringstream ss;
      for (auto &&p : path)
      {
        ss << "(" << p.x << "," << p.y << ") ";
      }
      spdlog::trace("path: {}", ss.str());
      if (path.back() != end)
      {
        spdlog::trace("adding end direct path");
        auto direct = direct_path(current_node_ptr->pos, end);
        std::stringstream s;
        for (auto &&p : direct)
        {
          s << "(" << p.x << "," << p.y << ") ";
        }
        spdlog::trace("direct path: {}", s.str());
        path.insert(path.end(), direct.begin() + 1, direct.end());
      }
      std::stringstream s;
      for (auto &&p : path)
      {
        s << "(" << p.x << "," << p.y << ") ";
      }
      spdlog::trace("Path reconstructed. final path: {}", s.str());
      return path;
    }
    // 获取当前节点的子节点
    for (int i = 0; i < 8; i++)
    {
      // 计算子节点的位置
      auto &&next_pos = current_node_ptr->pos + DIRECTIONS[i];
      // 如果子节点越界或者是障碍物, 则跳过
      if (next_pos.x < 0 || next_pos.x >= exploration_map->cols() || next_pos.y < 0 ||
          next_pos.y >= exploration_map->rows())
        continue;
      if ((*exploration_map)(next_pos.x, next_pos.y) != FREE)
        continue;
      Node child = {next_pos, current_node_ptr};
      // 如果子节点已经访问过, 则跳过
      if (close_set.count(child) != 0)
      {
        continue;
      }
      child.g = current_node_ptr->g + ((i < 4) ? 10 : 14);
      child.h = std::sqrt((std::pow(child.pos.x - end.x, 2) + std::pow(child.pos.y - end.y, 2))) * 10;
      child.f = child.g + child.h;
      // 如果子节点已经在open_set中, 则更新cost 和 parent
      auto node_in_open_set_itr = std::find(open_set.begin(), open_set.end(), child);
      if (node_in_open_set_itr != open_set.end())
      {
        if (child.g < node_in_open_set_itr->g)
        {
          // update cost and parent
          open_set.erase(node_in_open_set_itr);
          open_set.insert(child);
        }
      }
      else
      { // 如果子节点不在open_set中, 则加入open_set
        open_set.insert(child);
      }
    }
  }
  spdlog::warn("No path found");
  return {};
}
void map_merge(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *agent_map)
{
  // map_update\ global_map |  0  | 127 | 255
  //                    0   |  0  | <0> | <0>
  //                    127 |  0  | 127 | 255
  //                    255 |  0  |<255>| 255
  if (!global_map)
  {
    throw std::invalid_argument("Null pointer passed to map_merge: global_map");
  }
  if (!agent_map)
  {
    throw std::invalid_argument("Null pointer passed to map_merge: agent_map");
  }
  auto &&mat_global = cv::Mat(global_map->rows(), global_map->cols(), CV_8UC1, global_map->data());
  auto &&mat_update = cv::Mat(agent_map->rows(), agent_map->cols(), CV_8UC1, agent_map->data());
  // 检查矩阵尺寸是否匹配
  if (mat_global.size() != mat_update.size())
  {
    throw std::invalid_argument("Mismatched map sizes in map_merge");
  }
  mat_global.setTo(OCCUPIED, mat_update == OCCUPIED);
  mat_global.setTo(FREE, (mat_update == FREE) & (mat_global != OCCUPIED));
}
int calculate_valid_explored_pixels(std::shared_ptr<Env::GridMap> global_map, Env::GridMap *agent_map)
{
  spdlog::trace("calculate_new_explored_pixels");
  auto &&mat_global = cv::Mat(global_map->rows(), global_map->cols(), CV_8UC1, global_map->data());
  auto &&mat_agent = cv::Mat(agent_map->rows(), agent_map->cols(), CV_8UC1, agent_map->data());
  //  current map is known, and global_map is unknown
  return cv::countNonZero((mat_global == UNKNOWN) & (mat_agent != UNKNOWN));
}

int count_pixels(Env::GridMap *map, uint8_t value, bool exclude_value)
{
  auto &&mat = cv::Mat(map->rows(), map->cols(), CV_8UC1, map->data());
  if (exclude_value)
  {
    return cv::countNonZero(mat != value);
  }
  return cv::countNonZero(mat == value);
}

float exploration_rate(std::shared_ptr<Env::GridMap> env_map, Env::GridMap *exploration_map)
{
  return static_cast<float>(count_pixels(exploration_map, FREE, false)) /
         static_cast<float>(count_pixels(env_map.get(), FREE, false));
}

bool is_frontier_valid(std::shared_ptr<Env::GridMap> global_map, Env::FrontierPoint &frontier_point, int radius,
                       int threshold, bool check_reachability, Env::GridMap *exploration_map, Coord agent_pos)
{
  auto unexplored_pixels = calculate_surrounding_unexplored_pixels(global_map.get(), frontier_point.pos, radius);
  if (unexplored_pixels < threshold)
    return false;
  if (check_reachability)
  {
    if (!exploration_map)
    {
      return false;
    }
    if (a_star(exploration_map, agent_pos, frontier_point.pos).empty())
    {
      return false;
    }
  }
  return true;
}

} // namespace Alg