/**
 * @file grid_map.hpp
 * @author zhaoth
 * @brief GridMap data structure
 * @version 0.1
 * @date 2024-08-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <algorithm>
#include <cstdint>
namespace Env
{
/**
 * @brief grid map data structure
 * @note Note that OccupancyGrid data starts on lower left corner in row major
 * (if seen as an image), width / X / col is from left to right, height / Y /
 * row is from bottom to top
 */

struct GridMap
{
  GridMap(int width, int height) : width_(width), height_(height) { data_ = new uint8_t[width * height]; }
  GridMap(uint8_t *data, int width, int height) : width_(width), height_(height)
  {
    this->data_ = new uint8_t[width * height];
    std::copy(data, data + width * height, this->data_);
  }
  GridMap(const GridMap &other) : width_(other.width_), height_(other.height_)
  {
    this->data_ = new uint8_t[width_ * height_];
    std::copy(other.data_, other.data_ + width_ * height_, this->data_);
  }
  GridMap(int width, int height, uint8_t value)
  {
    this->width_ = width;
    this->height_ = height;
    this->data_ = new uint8_t[width * height];
    std::fill_n(this->data_, width * height, value);
  }

  int size() const { return width_ * height_; }
  uint8_t *data() { return data_; }
  int rows() const { return height_; }
  int cols() const { return width_; }
  int width() const { return width_; }
  int height() const { return height_; }
  uint8_t &operator()(int x, int y) { return data_[y * width_ + x]; }

  ~GridMap() { delete[] data_; }

  uint8_t *data_;
  int width_;
  int height_;
};

} // namespace Env
