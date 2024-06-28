#pragma once
#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif
#include <stdint.h>

namespace sgm_util
{
	//······ census工具集
	// census变换

	/**
	 * \brief census变换
	 * \param source	输入，影像数据
	 * \param census	输出，census值数组
	 * \param width		输入，影像宽
	 * \param height	输入，影像高
	 */
	void census_transform_5x5(const uint8_t* source, uint32_t* census, const int32_t& width, const int32_t& height);
	void census_transform_9x7(const uint8_t* source, uint64_t* census, const int32_t& width, const int32_t& height);
	// Hamming距离
	uint8_t Hamming32(const uint32_t& x, const uint32_t& y);
	uint8_t Hamming64(const uint64_t& x, const uint64_t& y);

	/**
	 * \brief 左右路径聚合 → ←
	 * \param img_data			输入，影像数据
	 * \param width				输入，影像宽
	 * \param height			输入，影像高
	 * \param min_disparity		输入，最小视差
	 * \param max_disparity		输入，最大视差
	 * \param p1				输入，惩罚项P1
	 * \param p2_init			输入，惩罚项P2_Init
	 * \param cost_init			输入，初始代价数据
	 * \param cost_aggr			输出，路径聚合代价数据
	 * \param is_forward		输入，是否为正方向（正方向为从左到右，反方向为从右到左）
	 */
	void CostAggregateLeftRight(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1,const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	/**
	 * \brief 上下路径聚合 ↓ ↑
	 * \param img_data			输入，影像数据
	 * \param width				输入，影像宽
	 * \param height			输入，影像高
	 * \param min_disparity		输入，最小视差
	 * \param max_disparity		输入，最大视差
	 * \param p1				输入，惩罚项P1
	 * \param p2_init			输入，惩罚项P2_Init
	 * \param cost_init			输入，初始代价数据
	 * \param cost_aggr			输出，路径聚合代价数据
	 * \param is_forward		输入，是否为正方向（正方向为从上到下，反方向为从下到上）
	 */
	void CostAggregateUpDown(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	/**
	 * \brief 对角线1路径聚合（左上<->右下）↘ ↖
	 * \param img_data			输入，影像数据
	 * \param width				输入，影像宽
	 * \param height			输入，影像高
	 * \param min_disparity		输入，最小视差
	 * \param max_disparity		输入，最大视差
	 * \param p1				输入，惩罚项P1
	 * \param p2_init			输入，惩罚项P2_Init
	 * \param cost_init			输入，初始代价数据
	 * \param cost_aggr			输出，路径聚合代价数据
	 * \param is_forward		输入，是否为正方向（正方向为从左上到右下，反方向为从右下到左上）
	 */
	void CostAggregateDagonal_1(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	/**
	 * \brief 对角线2路径聚合（右上<->左下）↙ ↗
	 * \param img_data			输入，影像数据
	 * \param width				输入，影像宽
	 * \param height			输入，影像高
	 * \param min_disparity		输入，最小视差
	 * \param max_disparity		输入，最大视差
	 * \param p1				输入，惩罚项P1
	 * \param p2_init			输入，惩罚项P2_Init
	 * \param cost_init			输入，初始代价数据
	 * \param cost_aggr			输出，路径聚合代价数据
	 * \param is_forward		输入，是否为正方向（正方向为从上到下，反方向为从下到上）
	 */
	void CostAggregateDagonal_2(const uint8_t* img_data, const int32_t& width, const int32_t& height, const int32_t& min_disparity, const int32_t& max_disparity,
		const int32_t& p1, const int32_t& p2_init, const uint8_t* cost_init, uint8_t* cost_aggr, bool is_forward = true);

	
	/**
	 * \brief 中值滤波
	 * \param in				输入，源数据 
	 * \param out				输出，目标数据
	 * \param width				输入，宽度
	 * \param height			输入，高度
	 * \param wnd_size			输入，窗口宽度
	 */
	void MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height, const int32_t wnd_size);


	/**
	 * \brief 剔除小连通区
	 * \param disparity_map		输入，视差图 
	 * \param width				输入，宽度
	 * \param height			输入，高度
	 * \param diff_insame		输入，同一连通区内的局部像素差异
	 * \param min_speckle_aera	输入，最小连通区面积
	 * \param invalid_val		输入，无效值
	 */
	void RemoveSpeckles(float* disparity_map, const int32_t& width, const int32_t& height, const int32_t& diff_insame,const uint32_t& min_speckle_aera, const float& invalid_val);
}