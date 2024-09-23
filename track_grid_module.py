# ==================================================================== #
# Copyright (C) 2023 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
# ==================================================================== #

import cv2
import numpy as np


class Track_Grid_Module():

	def __init__(self, w, h, grid_step=20) -> None:

		self.no_h_grid = int(np.ceil(h / grid_step))
		self.no_w_grid = int(np.ceil(w / grid_step))
		self.grid_step = grid_step

		self.grid_track = np.zeros((self.no_w_grid, self.no_h_grid))

		self.grid_coor_xy = np.zeros((self.no_w_grid, self.no_h_grid, 2))

		for idx_grid_w in range(0, self.no_w_grid):
			for idx_grid_h in range(0, self.no_h_grid):
				cx = int(idx_grid_w * self.grid_step)
				cy = int(idx_grid_h * self.grid_step)
				self.grid_coor_xy[idx_grid_w][idx_grid_h] = [cx, cy]

		self.grid_track_counter = np.zeros((self.no_w_grid, self.no_h_grid))
		self.grid_track_counter_noreset = np.zeros((self.no_w_grid, self.no_h_grid))
		self.grid_counter_max = 10

		self.grid_track_storage = np.zeros((self.no_w_grid, self.no_h_grid, 10, 2))

	def grid_visualize(self, bg_img):

		# print(self.grid_track.shape, self.no_w_grid, self.no_h_grid)
		for idx_grid_w in range(0, self.no_w_grid):
			for idx_grid_h in range(0, self.no_h_grid):
				cx = int(self.grid_coor_xy[idx_grid_w][idx_grid_h][0])
				cy = int(self.grid_coor_xy[idx_grid_w][idx_grid_h][1])

				if (self.grid_track[idx_grid_w, idx_grid_h] == 1):
					cv2.circle(bg_img, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
				else:
					cv2.circle(bg_img, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)

		# self.grid_track = np.zeros((self.no_w_grid, self.no_h_grid))

		return bg_img

	def cvt_xy_coor_to_grid_coor(self, coor):

		return int(np.floor(coor / self.grid_step))

	def cvt_xy_coor_to_grid_coor_centroid(self, centroid):
		return [self.cvt_xy_coor_to_grid_coor(centroid[0]), self.cvt_xy_coor_to_grid_coor(centroid[1])]

	def update_per_object(self, tracked_object):

		obj_past_detections = tracked_object.past_detections

		for idx in range(len(obj_past_detections) - 1, 0, -1):
			curr_centroid = obj_past_detections[idx].points[0]
			# next_centroid = obj_past_detections[idx + 1].points[0]

			curr_grid = self.cvt_xy_coor_to_grid_coor_centroid(curr_centroid)
			self.grid_track[curr_grid[0]][curr_grid[1]] = 1
			cur_counter = int(self.grid_track_counter[curr_grid[0]][curr_grid[1]])
			self.grid_track_storage[curr_grid[0]][curr_grid[1]][cur_counter] = curr_centroid

			self.grid_track_counter[curr_grid[0]][curr_grid[1]] += 1
			self.grid_track_counter_noreset[curr_grid[0]][curr_grid[1]] += 1

			# Check grid counter
			if (self.grid_track_counter[curr_grid[0]][curr_grid[1]] == self.grid_counter_max):
				new_coor = np.average(self.grid_track_storage[curr_grid[0]][curr_grid[1]], axis=0, keepdims=True)
				self.grid_coor_xy[curr_grid[0]][curr_grid[1]] = new_coor

				self.grid_track_storage[curr_grid[0]][curr_grid[1]][0] = new_coor
				self.grid_track_counter[curr_grid[0]][curr_grid[1]] = 1

			# curr_grid = cvt_xy_coor_to_grid_coor_centroid(curr_centroid)

			break

		return

	def update_per_object_yolo(self, history_moving_active_tracks):

		for obj_past_detections in history_moving_active_tracks:

			for idx in range(len(obj_past_detections) - 1, 0, -1):
				curr_centroid = [
					(obj_past_detections[idx][0] + obj_past_detections[idx][2]) / 2,
					(obj_past_detections[idx][1] + obj_past_detections[idx][3]) / 2
				]
				# next_centroid = obj_past_detections[idx + 1].points[0]

				curr_grid = self.cvt_xy_coor_to_grid_coor_centroid(curr_centroid)
				curr_grid_x = curr_grid[0]
				curr_grid_y = curr_grid[1]

				self.grid_track[curr_grid_x][curr_grid_y] = 1
				cur_counter = int(self.grid_track_counter[curr_grid_x][curr_grid_y])
				self.grid_track_storage[curr_grid_x][curr_grid_y][cur_counter] = curr_centroid

				self.grid_track_counter[curr_grid_x][curr_grid_y] += 1
				self.grid_track_counter_noreset[curr_grid_x][curr_grid_y] += 1

				# Check grid counter
				if (self.grid_track_counter[curr_grid_x][curr_grid_y] == self.grid_counter_max):
					new_coor = np.average(self.grid_track_storage[curr_grid_x][curr_grid_y], axis=0, keepdims=True)
					self.grid_coor_xy[curr_grid_x][curr_grid_y] = new_coor

					self.grid_track_storage[curr_grid_x][curr_grid_y][0] = new_coor
					self.grid_track_counter[curr_grid_x][curr_grid_y] = 1

				# curr_grid = cvt_xy_coor_to_grid_coor_centroid(curr_centroid)

				break

		return

	def list_of_active_grid(self):
		active_list = []

		for idx_grid_w in range(0, self.no_w_grid):
			for idx_grid_h in range(0, self.no_h_grid):
				cx = int(self.grid_coor_xy[idx_grid_w][idx_grid_h][0])
				cy = int(self.grid_coor_xy[idx_grid_w][idx_grid_h][1])

				if (self.grid_track[idx_grid_w, idx_grid_h] == 1):
					active_list.append([cx, cy])

		return np.array(active_list)

	def list_of_selected_active_grid(self, number_of_grid=5):

		indices = (-self.grid_track_counter_noreset).argpartition(number_of_grid, axis=None)[:number_of_grid]
		x_g, y_g = np.unravel_index(indices, self.grid_track_counter_noreset.shape)

		active_list = self.list_of_active_grid()
		selected_list = []
		for ix, iy in zip(x_g, y_g):
			cx = int(self.grid_coor_xy[ix][iy][0])
			cy = int(self.grid_coor_xy[ix][iy][1])

			if (self.grid_track[ix, iy] == 1):
				selected_list.append([cx, cy])

		return np.array(selected_list)

	def display_selected_point(self, selected_list, img):

		for g_point in selected_list:
			cv2.circle(img, (g_point[0], g_point[1]), 3, (255, 255, 0), 3, cv2.LINE_AA)

		return img

	def generate_grid_RoI(self, grid_active_points, bg_img):
		hulls = cv2.convexHull(grid_active_points)

		myROI = np.zeros((hulls.shape[0], 2), dtype=np.int32)

		for idx_hull in range(0, len(hulls)):
			myROI[idx_hull] = (hulls[idx_hull][0][0], hulls[idx_hull][0][1])

		mask = np.zeros((bg_img.shape[0], bg_img.shape[1]))
		mask = cv2.fillPoly(mask, [np.array(myROI)], 1)

		return mask


