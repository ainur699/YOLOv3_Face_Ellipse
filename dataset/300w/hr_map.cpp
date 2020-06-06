#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include "Estimate3d.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"

#define db(i, j) at<double>(i, j)
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#define PI 3.14159265f
static const int num_points_2d = 66;
static const int num_points_3d = 92 + 2;


//-----------------------------------------------------------------------------
cv::Point3d TopPoint(cv::Mat &shape)
{
	cv::Point3d res;

	const cv::Point3d v1(0.5 * (shape.db(19, 0) + shape.db(24, 0)), 0.5 * (shape.db(19 + num_points_2d, 0) + shape.db(24 + num_points_2d, 0)), 0.5 * (shape.db(19 + 2 * num_points_2d, 0) + shape.db(24 + 2 * num_points_2d, 0)));
	const cv::Point3d v2(shape.db(33, 0), shape.db(33 + num_points_2d, 0), shape.db(33 + 2 * num_points_2d, 0));
	const double      df = cv::norm(v1 - v2);
	const double	  dz = 0.5 * (shape.db(39 + 2 * num_points_2d, 0) + shape.db(42 + 2 * num_points_2d, 0)) - shape.db(33 + 2 * num_points_2d, 0);
	res.x = v1.x;
	res.y = v1.y - df;
	res.z = v1.z + 2 * dz;

	return res;
}


//-----------------------------------------------------------------------------
void Euler2Rot(cv::Mat& R, double pitch, double yaw, double roll)
{
	R.create(3, 3, CV_64F);

	const double sina = sin(pitch), sinb = sin(yaw), sinc = sin(roll);
	const double cosa = cos(pitch), cosb = cos(yaw), cosc = cos(roll);

	R.db(0, 0) = cosb * cosc;
	R.db(0, 1) = -cosb * sinc;
	R.db(0, 2) = sinb;
	R.db(1, 0) = cosa * sinc + sina * sinb * cosc;
	R.db(1, 1) = cosa * cosc - sina * sinb * sinc;
	R.db(1, 2) = -sina * cosb;

	R.db(2, 0) = R.db(0, 1) * R.db(1, 2) - R.db(0, 2) * R.db(1, 1);
	R.db(2, 1) = R.db(0, 2) * R.db(1, 0) - R.db(0, 0) * R.db(1, 2);
	R.db(2, 2) = R.db(0, 0) * R.db(1, 1) - R.db(0, 1) * R.db(1, 0);
}


//-----------------------------------------------------------------------------
cv::Point2f GetPoints(const cv::Point3d& shape, const cv::Mat& global)
{
	cv::Mat R;
	Euler2Rot(R, global.db(1, 0), global.db(2, 0), global.db(3, 0));

	cv::Point2f vertex;
	const double a = global.db(0, 0);
	const double x = global.db(4, 0);
	const double y = global.db(5, 0);

	vertex.x = static_cast<float>(a * (R.db(0, 0) * shape.x + R.db(0, 1) * shape.y + R.db(0, 2) * shape.z) + x);
	vertex.y = static_cast<float>(a * (R.db(1, 0) * shape.x + R.db(1, 1) * shape.y + R.db(1, 2) * shape.z) + y);

	return vertex;
}


//-----------------------------------------------------------------------------
std::vector<cv::Point2f> GetPoints(const std::vector<cv::Point3d>& shape, const cv::Mat& global)
{
	cv::Mat R;
	Euler2Rot(R, global.db(1, 0), global.db(2, 0), global.db(3, 0));

	std::vector<cv::Point2f> vertex(shape.size());
	const double a = global.db(0, 0);
	const double x = global.db(4, 0);
	const double y = global.db(5, 0);

	for (size_t i = 0; i < shape.size(); i++)
	{
		vertex[i].x = static_cast<float>(a * (R.db(0, 0) * shape[i].x + R.db(0, 1) * shape[i].y + R.db(0, 2) * shape[i].z) + x);
		vertex[i].y = static_cast<float>(a * (R.db(1, 0) * shape[i].x + R.db(1, 1) * shape[i].y + R.db(1, 2) * shape[i].z) + y);
	}

	return vertex;
}


//-----------------------------------------------------------------------------
cv::Mat addExtraPoints(cv::Mat& shape)
{
	///
	cv::Mat shape_dst(3 * num_points_3d, 1, CV_64F);
	for (int i = 0; i < num_points_2d; i++)
	{
		shape_dst.db(i, 0) = shape.db(i, 0);
		shape_dst.db(i + num_points_3d, 0) = shape.db(i + num_points_2d, 0);
		shape_dst.db(i + 2 * num_points_3d, 0) = shape.db(i + 2 * num_points_2d, 0);
	}


	/// forehead
#if 0
	const cv::Point3d v1(shape.db(19, 0), shape.db(19 + num_points_2d, 0), 0);
	const cv::Point3d v2(shape.db(8, 0), shape.db(8 + num_points_2d, 0), 0);
	const double      cf = cv::norm(v1 - v2);
	const double      cb = cf / 2;
	shape_dst.db(68, 0) = shape.db(27, 0);
	shape_dst.db(68 + num_points_3d, 0) = shape.db(19, 0) - 0.8 * cb;
	shape_dst.db(68 + 2 * num_points_3d, 0) = shape.db(27, 0) + 0.1 * cb;
#else 
	const cv::Point3d v1(0.5 * (shape.db(19, 0) + shape.db(24, 0)), 0.5 * (shape.db(19 + num_points_2d, 0) + shape.db(24 + num_points_2d, 0)), 0.5 * (shape.db(19 + 2 * num_points_2d, 0) + shape.db(24 + 2 * num_points_2d, 0)));
	const cv::Point3d v2(shape.db(33, 0), shape.db(33 + num_points_2d, 0), shape.db(33 + 2 * num_points_2d, 0));
	const double      df = cv::norm(v1 - v2);
	const double	  dz = 0.5 * (shape.db(39 + 2 * num_points_2d, 0) + shape.db(42 + 2 * num_points_2d, 0)) - shape.db(33 + 2 * num_points_2d, 0);
	shape_dst.db(68, 0) = v1.x;
	shape_dst.db(68 + num_points_3d, 0) = v1.y - df;
	shape_dst.db(68 + 2 * num_points_3d, 0) = v1.z + 2 * dz;
#endif

	const cv::Point3d initial_left(shape.db(0, 0), shape.db(0 + num_points_2d, 0), shape.db(0 + 2 * num_points_2d, 0));
	const cv::Point3d initial_right(shape.db(16, 0), shape.db(16 + num_points_2d, 0), shape.db(16 + 2 * num_points_2d, 0));
	const cv::Point3d range_left(shape_dst.db(68, 0) - shape.db(0, 0), shape_dst.db(68 + num_points_3d, 0) - shape.db(num_points_2d, 0), shape_dst.db(68 + 2 * num_points_3d, 0) - shape.db(2 * num_points_2d, 0));
	const cv::Point3d range_right(shape_dst.db(68, 0) - shape.db(16, 0), shape_dst.db(68 + num_points_3d, 0) - shape.db(16 + num_points_2d, 0), shape_dst.db(68 + 2 * num_points_3d, 0) - shape.db(16 + 2 * num_points_2d, 0));
	const double      factors_x[2] = { 0.5, 0.8 };
	const double      factors_y[2] = { 1.6, 1.3 };
	const double      factors_z[2] = { -0.5, 0.0 };
	const double      k = 3.0;
	for (int i = num_points_2d, j = 70; i < 68; i++, j--)
	{
		shape_dst.db(i, 0) = initial_left.x + ((i - num_points_2d + 1) / k) * range_left.x * factors_x[i - num_points_2d];
		shape_dst.db(i + num_points_3d, 0) = initial_left.y + ((i - num_points_2d + 1) / k) * range_left.y * factors_y[i - num_points_2d];
		shape_dst.db(i + 2 * num_points_3d, 0) = shape_dst.db(68 + 2 * num_points_3d, 0) + range_left.z * factors_z[i - num_points_2d];
		shape_dst.db(j, 0) = initial_right.x + ((i - num_points_2d + 1) / k) * range_right.x * factors_x[i - num_points_2d];
		shape_dst.db(j + num_points_3d, 0) = initial_right.y + ((i - num_points_2d + 1) / k) * range_right.y * factors_y[i - num_points_2d];
		shape_dst.db(j + 2 * num_points_3d, 0) = shape_dst.db(68 + 2 * num_points_3d, 0) + range_left.z * factors_z[i - num_points_2d];
	}


	/// left eye
	double dist = 0.25 * hypot(shape.db(36, 0) - shape.db(39, 0), shape.db(36 + num_points_2d, 0) - shape.db(39 + num_points_2d, 0));
	int    eye = 36, eyebrow = 17, eyelash = 71;
	for (int i = 0; i < 4; i++, eye++, eyebrow++, eyelash++)
	{
		if (i == 2) eyebrow++;

		const double mod = hypot(shape.db(eye, 0) - shape.db(eyebrow, 0), shape.db(eye + num_points_2d, 0) - shape.db(eyebrow + num_points_2d, 0));
		shape_dst.db(eyelash, 0) = shape.db(eye, 0) + (shape.db(eyebrow, 0) - shape.db(eye, 0)) / mod * dist;
		shape_dst.db(eyelash + num_points_3d, 0) = shape.db(eye + num_points_2d, 0) + (shape.db(eyebrow + num_points_2d, 0) - shape.db(eye + num_points_2d, 0)) / mod * dist;
		shape_dst.db(eyelash + 2 * num_points_3d, 0) = shape.db(eye + 2 * num_points_2d, 0);
	}


	/// right eye
	dist = 0.25 * hypot(shape.db(42, 0) - shape.db(45, 0), shape.db(42 + num_points_2d, 0) - shape.db(45 + num_points_2d, 0));
	eye = 42, eyebrow = 22, eyelash = 75;
	for (int i = 0; i < 4; i++, eye++, eyebrow++, eyelash++)
	{
		if (i == 2) eyebrow++;

		const double mod = hypot(shape.db(eye, 0) - shape.db(eyebrow, 0), shape.db(eye + num_points_2d, 0) - shape.db(eyebrow + num_points_2d, 0));
		shape_dst.db(eyelash, 0) = shape.db(eye, 0) + (shape.db(eyebrow, 0) - shape.db(eye, 0)) / mod * dist;
		shape_dst.db(eyelash + num_points_3d, 0) = shape.db(eye + num_points_2d, 0) + (shape.db(eyebrow + num_points_2d, 0) - shape.db(eye + num_points_2d, 0)) / mod * dist;
		shape_dst.db(eyelash + 2 * num_points_3d, 0) = shape.db(eye + 2 * num_points_2d, 0);
	}


	/// nose
	const double k_1 = 0.2;
	const double k_2 = 1.5;
	shape_dst.db(79, 0) = shape.db(31, 0) + k_1 * (shape.db(32, 0) - shape.db(31, 0));  shape_dst.db(79 + num_points_3d, 0) = (shape.db(28 + num_points_2d, 0) + shape.db(29 + num_points_2d, 0)) / 2;  shape_dst.db(79 + 2 * num_points_3d, 0) = shape.db(31 + 2 * num_points_2d, 0);
	shape_dst.db(80, 0) = shape.db(31, 0);                                              shape_dst.db(80 + num_points_3d, 0) = (shape.db(29 + num_points_2d, 0) + shape.db(30 + num_points_2d, 0)) / 2;  shape_dst.db(80 + 2 * num_points_3d, 0) = shape.db(31 + 2 * num_points_2d, 0);
	shape_dst.db(81, 0) = shape.db(31, 0) - k_2 * (shape.db(32, 0) - shape.db(31, 0));  shape_dst.db(81 + num_points_3d, 0) = shape.db(31 + num_points_2d, 0);                                         shape_dst.db(81 + 2 * num_points_3d, 0) = shape.db(31 + 2 * num_points_2d, 0);
	shape_dst.db(82, 0) = shape.db(35, 0) + k_2 * (shape.db(35, 0) - shape.db(34, 0));  shape_dst.db(82 + num_points_3d, 0) = shape.db(35 + num_points_2d, 0);                                         shape_dst.db(82 + 2 * num_points_3d, 0) = shape.db(35 + 2 * num_points_2d, 0);
	shape_dst.db(83, 0) = shape.db(35, 0);                                              shape_dst.db(83 + num_points_3d, 0) = (shape.db(29 + num_points_2d, 0) + shape.db(30 + num_points_2d, 0)) / 2;  shape_dst.db(83 + 2 * num_points_3d, 0) = shape.db(35 + 2 * num_points_2d, 0);
	shape_dst.db(84, 0) = shape.db(35, 0) - k_1 * (shape.db(35, 0) - shape.db(34, 0));  shape_dst.db(84 + num_points_3d, 0) = (shape.db(28 + num_points_2d, 0) + shape.db(29 + num_points_2d, 0)) / 2;  shape_dst.db(84 + 2 * num_points_3d, 0) = shape.db(35 + 2 * num_points_2d, 0);

	/// left and right nose
	shape_dst.db(85, 0) = shape.db(31, 0) - (shape.db(32, 0) - shape.db(31, 0)) * 0.65;	shape_dst.db(85 + num_points_3d, 0) = (shape.db(30 + num_points_2d, 0) * 0.65 + shape.db(33 + num_points_2d, 0) * 0.35);	shape_dst.db(85 + 2 * num_points_3d, 0) = shape.db(31 + 2 * num_points_2d, 0);
	shape_dst.db(86, 0) = shape.db(35, 0) + (shape.db(35, 0) - shape.db(34, 0)) * 0.65;	shape_dst.db(86 + num_points_3d, 0) = (shape.db(30 + num_points_2d, 0) * 0.65 + shape.db(33 + num_points_2d, 0) * 0.35);		shape_dst.db(86 + 2 * num_points_3d, 0) = shape.db(35 + 2 * num_points_2d, 0);

	/// head top
	int ray_point = 27; //27, 28, 29, 30 or 33
	double af_half = 2 * (shape.db(33 + num_points_2d, 0) - shape.db(27 + num_points_2d, 0));
	double coeff[] = { 0.70, 0.95, 1, 0.95, 0.70 };

	for (int forehead_idx = 66, head_idx = 87; head_idx <= 91; forehead_idx++, head_idx++)
	{
		cv::Point2f vect = cv::Point2f(shape_dst.db(forehead_idx, 0) - shape_dst.db(ray_point, 0), shape_dst.db(forehead_idx + num_points_3d, 0) - shape_dst.db(ray_point + num_points_3d, 0));
		vect = vect / cv::norm(vect) * af_half;
		shape_dst.db(head_idx, 0) = shape_dst.db(ray_point, 0) + vect.x * coeff[forehead_idx - 66];
		shape_dst.db(head_idx + num_points_3d, 0) = shape_dst.db(ray_point + num_points_3d, 0) + vect.y * coeff[forehead_idx - 66];
		shape_dst.db(head_idx + 2 * num_points_3d, 0) = shape_dst.db(forehead_idx + 2 * num_points_3d, 0);
	}


	/// центры масс глаз (симуляция зрачка)
	{
		int    posX, posY, posZ;
		double sX, sY, sZ, cnt;

		/// левый
		posX = 92, posY = posX + num_points_3d, posZ = posY + num_points_3d;
		sX = 0.0, sY = 0.0, sZ = 0.0, cnt = 0.0;
		for (int X = 36; X <= 41; X++, cnt += 1.0) { sX += shape_dst.db(X, 0); sY += shape_dst.db(X + num_points_3d, 0); sZ += shape_dst.db(X + num_points_3d * 2, 0); }
		shape_dst.db(posX, 0) = sX /= cnt; shape_dst.db(posY, 0) = sY /= cnt; shape_dst.db(posZ, 0) = sZ /= cnt;

		/// правый
		posX = 93, posY = posX + num_points_3d, posZ = posY + num_points_3d;
		sX = 0.0, sY = 0.0, sZ = 0.0, cnt = 0.0;
		for (int X = 42; X <= 47; X++, cnt += 1.0) { sX += shape_dst.db(X, 0); sY += shape_dst.db(X + num_points_3d, 0); sZ += shape_dst.db(X + num_points_3d * 2, 0); }
		shape_dst.db(posX, 0) = sX /= cnt; shape_dst.db(posY, 0) = sY /= cnt; shape_dst.db(posZ, 0) = sZ /= cnt;
	}


	/// все
	return shape_dst;
}


int main_300w_ver1()
{
	dlib::frontal_face_detector dlib_detector = dlib::get_frontal_face_detector();

	Estimate3D shape_estimate;
	bool res = shape_estimate.Init("fr_stage-3.bin");
	if (!res) {
		std::cout << "init error" << std::endl;
		std::system("pause");
		return -1;
	}

	std::string img_dir("D:/Github/HRNet-Facial-Landmark-Detection/data/300w/");
	std::ifstream ifile(img_dir + "face_landmarks_300w_train.csv");

	std::string line;
	std::getline(ifile, line);

	int image_id = 0;

	while (std::getline(ifile, line)) {
		size_t pos = line.find_first_of(",");
		std::string name = line.substr(0, pos);

		pos = line.find_first_of(",", pos + 1);
		pos = line.find_first_of(",", pos + 1);
		pos = line.find_first_of(",", pos + 1);

		size_t prev_pos;
		std::vector<cv::Point2f> pts(68);

		for (int i = 0; i < 68; i++) {
			prev_pos = pos + 1;
			pos = line.find_first_of(",", prev_pos);
			pts[i].x = std::stof(line.substr(prev_pos, pos - prev_pos));

			prev_pos = pos + 1;
			pos = line.find_first_of(",", prev_pos);
			pts[i].y = std::stof(line.substr(prev_pos, pos - prev_pos));
		}

		//---------------------------------------------------------------------------
		cv::Mat image = cv::imread(img_dir + "images/" + name);

		cv::Mat img_gray;
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
		std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlib::cv_image<uchar>(img_gray));
		if (faces_dlib.size() > 1) continue;

		cv::Mat landmarks(2 * num_points_2d, 1, CV_64F);
		for (int i = 0, j = 0; i < 68; i++)
		{
			if (i == 60 || i == 64) continue;

			landmarks.at<double>(j, 0) = pts[i].x;
			landmarks.at<double>(j + num_points_2d, 0) = pts[i].y;
			j++;
		}

		cv::Mat global, local, shape3d;
		shape_estimate.estimate(landmarks, global, local, shape3d);

		cv::Point3d forehead_3d = TopPoint(shape3d);
		cv::Point2f forehead_2d = GetPoints(forehead_3d, global);

		cv::Point2f d = (pts[45] - pts[36]) / cv::norm(pts[45] - pts[36]);

		float a_r = d.x;
		float b_r = d.y;
		float c_r = -d.x * pts[16].x - d.y * pts[16].y;

		float a_b = d.y;
		float b_b = -d.x;
		float c_b = d.x * pts[8].y - d.y * pts[8].x;

		float a_l = d.x;
		float b_l = d.y;
		float c_l = -d.x * pts[0].x - d.y * pts[0].y;

		float a_t = d.y;
		float b_t = -d.x;
		float c_t = d.x * forehead_2d.y - d.y * forehead_2d.x;

		cv::Point2f tl, tr, br;
		tl.x = (-c_l * b_t + c_t * b_l) / (a_l * b_t - a_t * b_l);
		tl.y = (-a_l * c_t + a_t * c_l) / (a_l * b_t - a_t * b_l);

		tr.x = (-c_r * b_t + c_t * b_r) / (a_r * b_t - a_t * b_r);
		tr.y = (-a_r * c_t + a_t * c_r) / (a_r * b_t - a_t * b_r);

		br.x = (-c_r * b_b + c_b * b_r) / (a_r * b_b - a_b * b_r);
		br.y = (-a_r * c_b + a_b * c_r) / (a_r * b_b - a_b * b_r);

		cv::Point center = 0.5f * (tl + br);
		cv::Size2f size(cv::norm(tr - br), cv::norm(tl - tr));
		float angle = 180.f / PI * std::atan2(d.y, d.x);
		angle = angle >= 0 ? angle - 90.f : 90 + angle;
		cv::RotatedRect bbox(center, size, angle);

		// out
		cv::circle(image, pts[0], 4, cv::Scalar(0, 255, 0), -1);
		cv::circle(image, pts[16], 4, cv::Scalar(0, 255, 0), -1);
		cv::circle(image, pts[8], 4, cv::Scalar(0, 255, 0), -1);
		cv::circle(image, forehead_2d, 4, cv::Scalar(0, 255, 0), -1);

		cv::ellipse(image, bbox, cv::Scalar(255, 0, 0), 2);

		std::cout << image_id << std::endl;
		cv::imwrite("C:/Users/zirga/Desktop/test_images/" + std::to_string(image_id++) + ".png", image);
		if (image_id > 200) break;
	}


	return 0;
}


int main_300w_ver2()
{
	std::ofstream ofile("D:/Github/pppFaceDetection/dataset/300w/300w_valid.txt");

	dlib::frontal_face_detector dlib_detector = dlib::get_frontal_face_detector();

	Estimate3D shape_estimate;
	bool res = shape_estimate.Init("fr_stage-3.bin");
	if (!res) {
		std::cout << "init error" << std::endl;
		std::system("pause");
		return -1;
	}

	std::string img_dir("D:/Github/HRNet-Facial-Landmark-Detection/data/300w/");
	std::ifstream ifile(img_dir + "face_landmarks_300w_test.csv");

	std::string line;
	std::getline(ifile, line);

	int image_id = 0;

	while (std::getline(ifile, line)) {
		size_t pos = line.find_first_of(",");
		std::string name = line.substr(0, pos);

		pos = line.find_first_of(",", pos + 1);
		pos = line.find_first_of(",", pos + 1);
		pos = line.find_first_of(",", pos + 1);

		size_t prev_pos;
		std::vector<cv::Point2f> pts(68);

		for (int i = 0; i < 68; i++) {
			prev_pos = pos + 1;
			pos = line.find_first_of(",", prev_pos);
			pts[i].x = std::stof(line.substr(prev_pos, pos - prev_pos));

			prev_pos = pos + 1;
			pos = line.find_first_of(",", prev_pos);
			pts[i].y = std::stof(line.substr(prev_pos, pos - prev_pos));
		}

		//---------------------------------------------------------------------------
		cv::Mat image = cv::imread(img_dir + "images/" + name);

		cv::Mat img_gray;
		cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
		std::vector<dlib::rectangle> faces_dlib = dlib_detector(dlib::cv_image<uchar>(img_gray));
		if (faces_dlib.size() > 1) continue;

		cv::Mat landmarks(2 * num_points_2d, 1, CV_64F);
		for (int i = 0, j = 0; i < 68; i++)
		{
			if (i == 60 || i == 64) continue;

			landmarks.at<double>(j, 0) = pts[i].x;
			landmarks.at<double>(j + num_points_2d, 0) = pts[i].y;
			j++;
		}

		cv::Mat global, local, shape3d;
		shape_estimate.estimate(landmarks, global, local, shape3d);

		cv::Mat shape3d_extra = addExtraPoints(shape3d);
		int shape_size = shape3d_extra.rows / 3;
		std::vector<cv::Point3d> shape(shape_size);
		for (int i = 0; i < shape_size; i++) shape[i] = cv::Point3d(shape3d_extra.db(i, 0), shape3d_extra.db(i + shape_size, 0), shape3d_extra.db(i + 2 * shape_size, 0));

		std::vector<cv::Point2f> l_extra = GetPoints(shape, global);
		std::vector<cv::Point2f> contour;
		for (int i = 0; i < 17; i++)   contour.push_back(l_extra[i]);
		for (int i = 70; i >= 66; i--) contour.push_back(l_extra[i]);

		cv::RotatedRect bbox = cv::fitEllipse(contour);

		cv::Point2f verts[4];
		bbox.points(verts);
		
		std::vector<float> y_vals(4), x_vals(4);
		for (int i = 0; i < 4; i++) {
			x_vals[i] = verts[i].x;
			y_vals[i] = verts[i].y;
		}

		cv::Point2f tl, tr, br;

		std::vector<size_t> y_ind = sort_indexes(y_vals);
		if (verts[y_ind[3]].x < verts[y_ind[2]].x) {
			tl = verts[y_ind[3]];
			tr = verts[y_ind[2]];
		}
		else {
			tl = verts[y_ind[2]];
			tr = verts[y_ind[3]];
		}

		if (verts[y_ind[0]].x > verts[y_ind[1]].x) {
			br = verts[y_ind[0]];
		}
		else {
			br = verts[y_ind[1]];
		}

		cv::Point2f center = 0.5f * (tl + br);
		cv::Size2f size(cv::norm(tl - tr), cv::norm(tr - br));
		cv::Point2f d = tr - tl;
		float angle = std::atan2(d.y, d.x);
		cv::RotatedRect bbox_norm(center, size, angle);
#if 0
		for (size_t i = 0; i < contour.size(); i++) {
			cv::circle(image, contour[i], 2, cv::Scalar(0, 255, 0), -1);
		}

		bbox_norm.angle *= 180.f / PI;
		cv::ellipse(image, bbox_norm, cv::Scalar(255, 0, 0), 2);

		// out
		std::cout << image_id << std::endl;
		cv::imwrite("C:/Users/zirga/Desktop/test_images/" + std::to_string(image_id++) + ".png", image);
		if (image_id > 200) break;
#else
		ofile << name << std::endl;
		ofile << 1 << std::endl;
		ofile << bbox_norm.center.x << " " << bbox_norm.center.y << " " << bbox_norm.size.width << " " << bbox_norm.size.height << " " << bbox_norm.angle << std::endl;

		std::cout << image_id++ << std::endl;
#endif
	}

	return 0;
}
