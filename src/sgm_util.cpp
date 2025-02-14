﻿#include <algorithm>
#include <cassert>
#include <vector>
#include <queue>
#include <cstring>
#include "../include/sgm_util.h"

void sgm_util::census_transform_5x5(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height)
{
	if (source == nullptr || census == nullptr || width <= 5 || height <= 5) {
		return;
	}

	// 逐像素计算census值
	for (int32_t i = 2; i < height - 2; i++) {
		for (int32_t j = 2; j < width - 2; j++) {
			
			// 中心像素值
			const uint8_t gray_center = source[i * width + j];
			
			// 遍历大小为5x5的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
			uint32_t census_val = 0u;
			for (int32_t r = -2; r <= 2; r++) {
				for (int32_t c = -2; c <= 2; c++) {
					census_val <<= 1;
					const uint8_t gray = source[(i + r) * width + j + c];
					if (gray < gray_center) {
						census_val += 1;
					}
				}
			}

			// 中心像素的census值
			census[i * width + j] = census_val;		
		}
	}
}

void sgm_util::census_transform_9x7(const uint8_t* source, uint64_t* census, const int32_t& width, const int32_t& height)
{
	if (source == nullptr || census == nullptr || width <= 9 || height <= 7) {
		return;
	}

	// 逐像素计算census值
	for (int32_t i = 4; i < height - 4; i++) {
		for (int32_t j = 3; j < width - 3; j++) {

			// 中心像素值
			const uint8_t gray_center = source[i * width + j];

			// 遍历大小为5x5的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
			uint64_t census_val = 0u;
			for (int32_t r = -4; r <= 4; r++) {
				for (int32_t c = -3; c <= 3; c++) {
					census_val <<= 1;
					const uint8_t gray = source[(i + r) * width + j + c];
					if (gray < gray_center) {
						census_val += 1;
					}
				}
			}

			// 中心像素的census值
			census[i * width + j] = census_val;
		}
	}
}

uint8_t sgm_util::Hamming32(const uint32_t& x, const uint32_t& y)
{
	uint32_t dist = 0, val = x ^ y;

	// Count the number of set bits
	while (val) {
		++dist;
		val &= val - 1;
	}

	return static_cast<uint8_t>(dist);
}

uint8_t sgm_util::Hamming64(const uint64_t& x, const uint64_t& y)
{
	uint64_t dist = 0, val = x ^ y;

	// Count the number of set bits
	while (val) {
		++dist;
		val &= val - 1;
	}

	return static_cast<uint8_t>(dist);
}


void sgm_util::CostAggregateLeftRight(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
	const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{
	assert(width > 0 && height > 0 && max_disparity > min_disparity);

	// 视差范围
	const int32_t disp_range = max_disparity - min_disparity;

	// P1,P2
	const auto& P1 = p1;
	const auto& P2_Init = p2_init;

	// 正向(左->右) ：is_forward = true ; direction = 1
	// 反向(右->左) ：is_forward = false; direction = -1;
	const int32_t direction = is_forward ? 1 : -1;

	// 聚合
	for (int32_t i = 0u; i < height; i++) {
		// 路径头为每一行的首(尾,dir=-1)列像素
		auto cost_init_row = (is_forward) ? (cost_init + i * width * disp_range) : (cost_init + i * width * disp_range + (width - 1) * disp_range);
		auto cost_aggr_row = (is_forward) ? (cost_aggr + i * width * disp_range) : (cost_aggr + i * width * disp_range + (width - 1) * disp_range);
		auto img_row = (is_forward) ? (img_data + i * width) : (img_data + i * width + width - 1);

		// 路径上当前灰度值和上一个灰度值
		uint8_t gray = *img_row;
		uint8_t gray_last = *img_row;

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8_t));
		cost_init_row += direction * disp_range;
		cost_aggr_row += direction * disp_range;
		img_row += direction;

		// 路径上上个像素的最小代价值
		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (int32_t j = 0; j < width - 1; j++) {
			gray = *img_row;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++){
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_row[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));
				
				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);
				
				cost_aggr_row[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_row, disp_range * sizeof(uint8_t));

			// 下一个像素
			cost_init_row += direction * disp_range;
			cost_aggr_row += direction * disp_range;
			img_row += direction;
			
			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

void sgm_util::CostAggregateUpDown(const uint8_t* img_data, const int32_t& width, const int32_t& height,
	const int32_t& min_disparity, const int32_t& max_disparity, const int32_t& p1, const int32_t& p2_init,
	const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{
	assert(width > 0 && height > 0 && max_disparity > min_disparity);

	// 视差范围
	const int32_t disp_range = max_disparity - min_disparity;

	// P1,P2
	const auto& P1 = p1;
	const auto& P2_Init = p2_init;

	// 正向(上->下) ：is_forward = true ; direction = 1
	// 反向(下->上) ：is_forward = false; direction = -1;
	const int32_t direction = is_forward ? 1 : -1;

	// 聚合
	for (int32_t j = 0; j < width; j++) {
		// 路径头为每一列的首(尾,dir=-1)行像素
		auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
		auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

		// 路径上当前灰度值和上一个灰度值
		uint8_t gray = *img_col;
		uint8_t gray_last = *img_col;

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));
		cost_init_col += direction * width * disp_range;
		cost_aggr_col += direction * width * disp_range;
		img_col += direction * width;

		// 路径上上个像素的最小代价值
		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (int32_t i = 0; i < height - 1; i ++) {
			gray = *img_col;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_col[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

			// 下一个像素
			cost_init_col += direction * width * disp_range;
			cost_aggr_col += direction * width * disp_range;
			img_col += direction * width;

			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

void sgm_util::CostAggregateDagonal_1(const uint8_t* img_data, const int32_t& width, const int32_t& height,
	const int32_t& min_disparity, const int32_t& max_disparity, const int32_t& p1, const int32_t& p2_init,
	const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{
	assert(width > 1 && height > 1 && max_disparity > min_disparity);

	// 视差范围
	const int32_t disp_range = max_disparity - min_disparity;

	// P1,P2
	const auto& P1 = p1;
	const auto& P2_Init = p2_init;

	// 正向(左上->右下) ：is_forward = true ; direction = 1
	// 反向(右下->左上) ：is_forward = false; direction = -1;
	const int32_t direction = is_forward ? 1 : -1;

	// 聚合

	// 存储当前的行列号，判断是否到达影像边界
	int32_t current_row = 0;
	int32_t current_col = 0;

	for (int32_t j = 0; j < width; j++) {
		// 路径头为每一列的首(尾,dir=-1)行像素
		auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
		auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

		// 路径上当前灰度值和上一个灰度值
		uint8_t gray = *img_col;
		uint8_t gray_last = *img_col;

		// 对角线路径上的下一个像素，中间间隔width+1个像素
		// 这里要多一个边界处理
		// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
		current_row = is_forward ? 0 : height - 1;
		current_col = j;
		if (is_forward && current_col == width - 1 && current_row < height - 1) {
			// 左上->右下，碰右边界
			cost_init_col = cost_init + (current_row + direction) * width * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
			img_col = img_data + (current_row + direction) * width;
            current_col = 0;
		}
		else if (!is_forward && current_col == 0 && current_row > 0) {
			// 右下->左上，碰左边界
			cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			img_col = img_data + (current_row + direction) * width + (width - 1);
            current_col = width - 1;
		}
		else {
			cost_init_col += direction * (width + 1) * disp_range;
			cost_aggr_col += direction * (width + 1) * disp_range;
			img_col += direction * (width + 1);
		}

		// 路径上上个像素的最小代价值
		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		// 自方向上第2个像素开始按顺序聚合
		for (int32_t i = 0; i < height - 1; i ++) {
			gray = *img_col;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_col[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

			// 当前像素的行列号
			current_row += direction;
			current_col += direction;
			
			// 下一个像素,这里要多一个边界处理
			// 这里要多一个边界处理
			// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
			if (is_forward && current_col == width - 1 && current_row < height - 1) {
				// 左上->右下，碰右边界
				cost_init_col = cost_init + (current_row + direction) * width * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
				img_col = img_data + (current_row + direction) * width;
                current_col = 0;
			}
			else if (!is_forward && current_col == 0 && current_row > 0) {
				// 右下->左上，碰左边界
				cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				img_col = img_data + (current_row + direction) * width + (width - 1);
                current_col = width - 1;
			}
			else {
				cost_init_col += direction * (width + 1) * disp_range;
				cost_aggr_col += direction * (width + 1) * disp_range;
				img_col += direction * (width + 1);
			}

			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

void sgm_util::CostAggregateDagonal_2(const uint8_t* img_data, const int32_t& width, const int32_t& height,
	const int32_t& min_disparity, const int32_t& max_disparity, const int32_t& p1, const int32_t& p2_init,
	const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward)
{
	assert(width > 1 && height > 1 && max_disparity > min_disparity);

	// 视差范围
	const int32_t disp_range = max_disparity - min_disparity;

	// P1,P2
	const auto& P1 = p1;
	const auto& P2_Init = p2_init;

	// 正向(右上->左下) ：is_forward = true ; direction = 1
	// 反向(左下->右上) ：is_forward = false; direction = -1;
	const int32_t direction = is_forward ? 1 : -1;

	// 聚合

	// 存储当前的行列号，判断是否到达影像边界
	int32_t current_row = 0;
	int32_t current_col = 0;

	for (int32_t j = 0; j < width; j++) {
		// 路径头为每一列的首(尾,dir=-1)行像素
		auto cost_init_col = (is_forward) ? (cost_init + j * disp_range) : (cost_init + (height - 1) * width * disp_range + j * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_aggr + j * disp_range) : (cost_aggr + (height - 1) * width * disp_range + j * disp_range);
		auto img_col = (is_forward) ? (img_data + j) : (img_data + (height - 1) * width + j);

		// 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
		std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
		memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

		// 路径上当前灰度值和上一个灰度值
		uint8_t gray = *img_col;
		uint8_t gray_last = *img_col;

		// 对角线路径上的下一个像素，中间间隔width-1个像素
		// 这里要多一个边界处理
		// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
		current_row = is_forward ? 0 : height - 1;
		current_col = j;
		if (is_forward && current_col == 0 && current_row < height - 1) {
			// 右上->左下，碰左边界
			cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
			img_col = img_data + (current_row + direction) * width + (width - 1);
            current_col = width - 1;
		}
		else if (!is_forward && current_col == width - 1 && current_row > 0) {
			// 左下->右上，碰右边界
			cost_init_col = cost_init + (current_row + direction) * width * disp_range ;
			cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
			img_col = img_data + (current_row + direction) * width;
            current_col = 0;
		}
		else {
			cost_init_col += direction * (width - 1) * disp_range;
			cost_aggr_col += direction * (width - 1) * disp_range;
			img_col += direction * (width - 1);
		}

		// 路径上上个像素的最小代价值
		uint8_t mincost_last_path = UINT8_MAX;
		for (auto cost : cost_last_path) {
			mincost_last_path = std::min(mincost_last_path, cost);
		}

		// 自路径上第2个像素开始按顺序聚合
		for (int32_t i = 0; i < height - 1; i++) {
			gray = *img_col;
			uint8_t min_cost = UINT8_MAX;
			for (int32_t d = 0; d < disp_range; d++) {
				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const uint8_t  cost = cost_init_col[d];
				const uint16_t l1 = cost_last_path[d + 1];
				const uint16_t l2 = cost_last_path[d] + P1;
				const uint16_t l3 = cost_last_path[d + 2] + P1;
				const uint16_t l4 = mincost_last_path + std::max(P1, P2_Init / (abs(gray - gray_last) + 1));

				const uint8_t cost_s = cost + static_cast<uint8_t>(std::min(std::min(l1, l2), std::min(l3, l4)) - mincost_last_path);

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 重置上个像素的最小代价值和代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

			// 当前像素的行列号
			current_row += direction;
			current_col -= direction;

			// 下一个像素,这里要多一个边界处理
			// 这里要多一个边界处理
			// 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
			if (is_forward && current_col == 0 && current_row < height - 1) {
				// 右上->左下，碰左边界
				cost_init_col = cost_init + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range + (width - 1) * disp_range;
				img_col = img_data + (current_row + direction) * width + (width - 1);
                current_col = width - 1;
			}
			else if (!is_forward && current_col == width - 1 && current_row > 0) {
				// 左下->右上，碰右边界
				cost_init_col = cost_init + (current_row + direction) * width * disp_range;
				cost_aggr_col = cost_aggr + (current_row + direction) * width * disp_range;
				img_col = img_data + (current_row + direction) * width;
                current_col = 0;
			}
			else {
				cost_init_col += direction * (width - 1) * disp_range;
				cost_aggr_col += direction * (width - 1) * disp_range;
				img_col += direction * (width - 1);
			}

			// 像素值重新赋值
			gray_last = gray;
		}
	}
}

void sgm_util::MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height,
	const int32_t wnd_size)
{
	const int32_t radius = wnd_size / 2;
	const int32_t size = wnd_size * wnd_size;

	// 存储局部窗口内的数据
	std::vector<float> wnd_data;
	wnd_data.reserve(size);

	for (int32_t i = 0; i < height; i++) {
		for (int32_t j = 0; j < width; j++) {
			wnd_data.clear();

			// 获取局部窗口数据
			for (int32_t r = -radius; r <= radius; r++) {
				for (int32_t c = -radius; c <= radius; c++) {
					const int32_t row = i + r;
					const int32_t col = j + c;
					if (row >= 0 && row < height && col >= 0 && col < width) {
						wnd_data.push_back(in[row * width + col]);
					}
				}
			}

			// 排序
			std::sort(wnd_data.begin(), wnd_data.end());
			// 取中值
			out[i * width + j] = wnd_data[wnd_data.size() / 2];
		}
	}
}

void sgm_util::RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame, const uint32_t& min_speckle_aera, const float& invalid_val)
{
	assert(width > 0 && height > 0);
	if (width < 0 || height < 0) {
		return;
	}

	// 定义标记像素是否访问的数组
	std::vector<bool> visited(uint32_t(width*height),false);
	for(int32_t i=0;i<height;i++) {
		for(int32_t j=0;j<width;j++) {
			if (visited[i * width + j] || disparity_map[i*width+j] == invalid_val) {
				// 跳过已访问的像素及无效像素
				continue;
			}
			// 广度优先遍历，区域跟踪
			// 把连通域面积小于阈值的区域视差全设为无效值
			std::vector<std::pair<int32_t, int32_t>> vec;
			vec.emplace_back(i, j);
			visited[i * width + j] = true;
			uint32_t cur = 0;
			uint32_t next = 0;
			do {
				// 广度优先遍历区域跟踪	
				next = vec.size();
				for (uint32_t k = cur; k < next; k++) {
					const auto& pixel = vec[k];
					const int32_t row = pixel.first;
					const int32_t col = pixel.second;
					const auto& disp_base = disparity_map[row * width + col];
					// 8邻域遍历
					for(int r=-1;r<=1;r++) {
						for(int c=-1;c<=1;c++) {
							if(r==0&&c==0) {
								continue;
							}
							int rowr = row + r;
							int colc = col + c;
							if (rowr >= 0 && rowr < height && colc >= 0 && colc < width) {
								if(!visited[rowr * width + colc] &&
									(disparity_map[rowr * width + colc] != invalid_val) &&
									abs(disparity_map[rowr * width + colc] - disp_base) <= diff_insame) {
									vec.emplace_back(rowr, colc);
									visited[rowr * width + colc] = true;
								}
							}
						}
					}
				}
				cur = next;
			} while (next < vec.size());

			// 把连通域面积小于阈值的区域视差全设为无效值
			if(vec.size() < min_speckle_aera) {
				for(auto& pix:vec) {
					disparity_map[pix.first * width + pix.second] = invalid_val;
				}
			}
		}
	}
}
