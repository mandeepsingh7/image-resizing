#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

//using namespace cv;
//using namespace std;
//using namespace std::chrono;

// Enumeration of the interpolation method
enum InterpolationFlags {
	INTER_NEAREST_CUSTOM = 0,
	INTER_LINEAR_CUSTOM = 1,
	INTER_CUBIC_CUSTOM = 2,
};

// Function to check consistency between 2 images 
// This function returns the percentage of pixels for which the difference in the intensity values of 2 images is less than a given tolerance tol.
// If tol == 0, then it will return the percentage of pixels with exact match between 2 images.
float isConsistent(cv::Mat& img_1, cv::Mat& img_2, int tol) {
	if (img_1.size() != img_2.size()) {
		return false;
	}
	cv::Mat diff;
	cv::absdiff(img_1, img_2, diff);

	int count = 0;

	for (int y = 0; y < img_1.rows; y++) {
		for (int x = 0; x < img_1.cols; x++) {
			cv::Vec3b pixels = diff.at<cv::Vec3b>(y, x);
			for (int i = 0; i < 3; i++) {
				if (static_cast<int>(pixels[i]) > tol) {
					count++;
				}
			}
		}
	}
	float consistency = static_cast<float>(count) / (img_1.cols * img_1.rows * 3);
	return (100 - (consistency * 100));
}

// Function for BiLinear Interpolation 
cv::Vec3b BiLinear(cv::Mat& src, float x, float y) {
	x = std::max(0.0f, x);
	y = std::max(0.0f, y);

	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);
	int x2 = std::min(x1 + 1, src.cols - 1);
	int y2 = std::min(y1 + 1, src.rows - 1);

	cv::Vec3b intensity_1 = src.at<cv::Vec3b>(y1, x1);
	cv::Vec3b intensity_2 = src.at<cv::Vec3b>(y1, x2);
	cv::Vec3b intensity_3 = src.at<cv::Vec3b>(y2, x1);
	cv::Vec3b intensity_4 = src.at<cv::Vec3b>(y2, x2);

	float xAlpha = x - x1;
	float yAlpha = y - y1; 

	cv::Vec3b intensity_new;

	for (int i = 0; i < 3; i++) {
		intensity_new[i] = intensity_1[i] * (1.0f - xAlpha) * (1.0f - yAlpha) +
					intensity_2[i] * xAlpha * (1.0f - yAlpha) +
					intensity_3[i] * (1.0f - xAlpha) * yAlpha +
					intensity_4[i] * xAlpha * yAlpha;
	}

	return intensity_new;
}

// Function for Cubic Interpolation in 1D
float Spline_Interpolate(float i1, float i2, float i3, float i4, float alpha) {
	float alpha2 = alpha * alpha;
	float alpha3 = alpha2 * alpha;
	
	const float c1 = (-i1 + 3.0f * i2 - 3.0f * i3 + i4) / 6.0f;
	const float c2 = (i1 - 2.0f * i2 + i3) / 2.0f;
	const float c3 = (-2.0f * i1 - 3.0f * i2 + 6.0f * i3 - i4) / 6.0f;

	return c1 * alpha3 + c2 * alpha2 + c3 * alpha + i2;
}

// Function for BiCubic Interpolation
cv::Vec3b BiCubic(cv::Mat& src, float x, float y) {
	
	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);

	int x0 = std::max(x1 - 1, 0);
	int x2 = std::min(x1 + 1, src.cols - 1);
	int x3 = std::min(x1 + 2, src.cols - 1);
	int y0 = std::max(y1 - 1, 0);
	int y2 = std::min(y1 + 1, src.rows - 1);
	int y3 = std::min(y1 + 2, src.rows - 1);

	//cout << "(x_src, y_src) = (" << x << "," << y << ")." << endl;
	//cout << "(x0, x1, x2, x3) = (" << x0 << ", " << x1 << ", " << x2 << ", " << x3 << ")." << endl;
	//cout << "(y0, y1, y2, y3) = (" << y0 << ", " << y1 << ", " << y2 << ", " << y3 << ")." << endl;

	float alphaX = x - x1;
	float alphaY = y - y1; 
	cv::Vec3b intensity_new;
	for (int i = 0; i < 3; i++) {
		float intensity_0 = Spline_Interpolate(
			src.at<cv::Vec3b>(y0, x0)[i],
			src.at<cv::Vec3b>(y0, x1)[i],
			src.at<cv::Vec3b>(y0, x2)[i],
			src.at<cv::Vec3b>(y0, x2)[i],
			alphaX
		);

		float intensity_1 = Spline_Interpolate(
			src.at<cv::Vec3b>(y1, x0)[i],
			src.at<cv::Vec3b>(y1, x1)[i],
			src.at<cv::Vec3b>(y1, x2)[i],
			src.at<cv::Vec3b>(y1, x2)[i],
			alphaX
		);

		float intensity_2 = Spline_Interpolate(
			src.at<cv::Vec3b>(y2, x0)[i],
			src.at<cv::Vec3b>(y2, x1)[i],
			src.at<cv::Vec3b>(y2, x2)[i],
			src.at<cv::Vec3b>(y2, x2)[i],
			alphaX
		);

		float intensity_3 = Spline_Interpolate(
			src.at<cv::Vec3b>(y3, x0)[i],
			src.at<cv::Vec3b>(y3, x1)[i],
			src.at<cv::Vec3b>(y3, x2)[i],
			src.at<cv::Vec3b>(y3, x2)[i],
			alphaX
		);
		//cout << "Color Channel = " << i << endl;
		//cout << "(i0, i1, i2, i3) = (" << intensity_0 << ", " << intensity_1 << ", " << intensity_2 << ", " << intensity_3 << ")." << endl;
		


		float final_intensity = Spline_Interpolate(intensity_0, intensity_1, intensity_2, intensity_3, alphaY);
		intensity_new[i] = static_cast<uchar>(std::max(0.0f, std::min(255.0f, final_intensity)));

		//cout << "Final Spline = " << final_intensity << endl;;
		//cout << "Final Intensity = " << static_cast<int>(intensity_new[i]) << endl;;
		//cout << "---------------------------------------------------" << endl;
	}
	return intensity_new;
}

// Custom Resize Function with different Interpolation Methods 
void custom_resize(cv::Mat& src, cv::Mat& dst, cv::Size dsize, double fx = 0.0, double fy = 0.0, int interpolation = INTER_NEAREST_CUSTOM) {

	if (dsize == cv::Size()) {
		dsize = cv::Size(cv::saturate_cast<int>(fx * src.cols), cv::saturate_cast<int>(fy * src.rows));
	}

	if (dsize != dst.size()) {
		dst.create(dsize, src.type());
	}

	float scaling_factor_x = ((static_cast<float>(src.cols)) / (dsize.width));
	float scaling_factor_y = ((static_cast<float>(src.rows)) / (dsize.height));

	std::vector<float> x_src_components(dsize.width);
	std::vector<float> y_src_components(dsize.height);

	for (int x = 0; x < dsize.width; x++) {
		x_src_components[x] = (x + 0.5f) * scaling_factor_x - 0.5f;
	}

	for (int y = 0; y < dsize.height; y++) {
		y_src_components[y] = (y + 0.5f) * scaling_factor_y - 0.5f;
	}

	for (int y = 0; y < dsize.height; y++) {
		float y_src = y_src_components[y];
		//float y_src = (y + 0.5f) * scaling_factor_y - 0.5f;
		int y_src_int = static_cast<int>(std::floor(0.49999f + y_src));
		for (int x = 0; x < dsize.width; x++) {
			float x_src = x_src_components[x];
			//float x_src = (x+0.5f) * scaling_factor_x - 0.5f;
			//cout << "For (x, y) = (" << x << ", " << y << "), (x_src, y_src) = (" << x_src << ", " << y_src << "), Int Value = (" << static_cast<int>(floor(0.49999 + x_src)) << "," << static_cast<int>(floor(0.49999 + y_src)) << ")" << endl;
			//cout << "-------------------------------------------" << endl; 


			if (interpolation == INTER_NEAREST_CUSTOM) {
				dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y_src_int, static_cast<int>(floor(0.49999 + x_src)));
			}
			else if (interpolation == INTER_LINEAR_CUSTOM) {
				dst.at<cv::Vec3b>(y, x) = BiLinear(src, x_src, y_src);
			}
			else if (interpolation == INTER_CUBIC_CUSTOM) {
				dst.at<cv::Vec3b>(y, x) = BiCubic(src, x_src, y_src);
			}
		}
	}
}


int main() {
	std::string image_path = cv::samples::findFile("G178_2 -1080.BMP");
	cv::Mat img = cv::imread(image_path);
	std::cout << "Size of Original Image = " << img.size() << std::endl;
	int height = img.rows;
	int width = img.cols;
	cv::Size new_size(width / 2, height / 2);
	std::cout << "Size of Resized Image = " << new_size << std::endl;
	std::cout << std::endl;
	//-----------------------------------
	// Step 1
	//-----------------------------------

	cv::Mat resize_nearest, resize_linear, resize_cubic;
	cv::resize(img, resize_nearest, new_size, 0, 0, cv::INTER_NEAREST);
	cv::imwrite("Resize_Nearest_OpenCV.bmp", resize_nearest);
	cv::resize(img, resize_linear, new_size, 0, 0, cv::INTER_LINEAR);
	cv::imwrite("Resize_Linear_OpenCV.bmp", resize_linear);
	cv::resize(img, resize_cubic, new_size, 0, 0, cv::INTER_CUBIC);
	cv::imwrite("Resize_Cubic_OpenCV.bmp", resize_cubic);
	//resize(img, resize_cubic_new, Size(), 0.5, 0.5, INTER_CUBIC);

	std::cout << "-------------------------------------------------------------------------------------------" << std::endl;

	//-----------------------------------
	// Step 2 
	//-----------------------------------
	int iteration_count = 100;
	std::vector<float> built_in_times;

	// Nearest Neighbours Interpolation
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		cv::resize(img, resize_nearest, new_size, 0, 0, cv::INTER_NEAREST);
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	built_in_times.push_back(duration);
	std::cout << "Time taken for " << iteration_count << " iterations using built-in resize function (INTER_NEAREST) = " << duration << " ms." << std::endl;

	// BiLinear Interpolation
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		cv::resize(img, resize_linear, new_size, 0, 0, cv::INTER_LINEAR);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	built_in_times.push_back(duration);
	std::cout << "Time taken for " << iteration_count << " iterations using built-in resize function (INTER_LINEAR) = " << duration << " ms." << std::endl;

	// BiCubic Interpolation
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		cv::resize(img, resize_cubic, new_size, 0, 0, cv::INTER_CUBIC);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	built_in_times.push_back(duration);
	std::cout << "Time taken for " << iteration_count << " iterations using built-in resize function (INTER_CUBIC) = " << duration << " ms." << std::endl;
	std::cout << "-------------------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	//-----------------------------------
	// Step 3
	//-----------------------------------

	cv::Mat resize_nearest_custom, resize_linear_custom, resize_cubic_custom;
	custom_resize(img, resize_nearest_custom, new_size, 0, 0, INTER_NEAREST_CUSTOM);
	cv::imwrite("Resize_Nearest_Custom.bmp", resize_nearest_custom);
	custom_resize(img, resize_linear_custom, new_size, 0, 0, INTER_LINEAR_CUSTOM);
	cv::imwrite("Resize_Linear_Custom.bmp", resize_linear_custom);
	custom_resize(img, resize_cubic_custom, new_size, 0, 0, INTER_CUBIC_CUSTOM);
	cv::imwrite("Resize_Cubic_Custom.bmp", resize_cubic_custom);
	std::cout << "Consistency of custom resize function (INTER_NEAREST) (tolerance = 0) = " << isConsistent(resize_nearest, resize_nearest_custom, 0) << "%." << std::endl;
	std::cout << "-----------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl; 
	std::cout << "----------------------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << "Tolerance is the maximum difference between pixel intensity of the image generated from custom and built-in functions." << std::endl;
	std::cout << "----------------------------------------------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl ;
	std::cout << "Consistency of custom resize function (INTER_LINEAR):" << std::endl;
	std::cout << "-----------------------------------------------------" << std::endl;
	for (int i = 0; i < 6; i++) {
		std::cout << "Consistency (tolerance = " << i << ") = " << std::fixed << std::setprecision(2) << isConsistent(resize_linear, resize_linear_custom, i) << "%." << std::endl;
	}
	std::cout << std::endl;
	std::cout << "Consistency of custom resize function (INTER_CUBIC):" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;
	for (int i = 0; i < 31; i = i + 5) {
		std::cout << "Consistency (tolerance = " << i << ") = " << std::fixed << std::setprecision(2) << isConsistent(resize_cubic, resize_cubic_custom, i) << "%." << std::endl;
	}
	std::cout << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
	std::vector<float> custom_times;

	// Nearest Neighbours Interpolation with Custom Function 
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		custom_resize(img, resize_nearest_custom, new_size, 0, 0, INTER_NEAREST_CUSTOM);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	custom_times.push_back(duration);
	std::cout << "Time taken for " << iteration_count << " iterations using custom resize function (INTER_NEAREST_CUSTOM) = " << duration << " ms." << std::endl;

	// BiLinear Interpolation with Custom Function 
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		custom_resize(img, resize_linear_custom, new_size, 0, 0, INTER_LINEAR_CUSTOM);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	custom_times.push_back(duration);
	std::cout << "Time taken for " << iteration_count << " iterations using custom resize function (INTER_LINEAR_CUSTOM) = " << duration << " ms." << std::endl;

	// BiCubic Interpolation with Custom Function 
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iteration_count; i++) {
		custom_resize(img, resize_cubic_custom, new_size, 0, 0, INTER_CUBIC_CUSTOM);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	custom_times.push_back(duration);
	std::cout << "Time taken for " << iteration_count << " iterations using custom resize function (INTER_CUBIC_CUSTOM) = " << duration << " ms." << std::endl;
	std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl; 

	std::vector<std::string> interpolation_methods = { "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC" };

	std::cout << "Comparing time taken by built-in and custom function for " << iteration_count << " iterations:" << std::endl;

	std::cout << std::setw(20) << "--------------------" << std::setw(20) << "------------------" << std::setw(20) << "----------------" << std::endl;
	std::cout << std::setw(20) << "Interpolation Method" << std::setw(20) << "Built-In Times(ms)" << std::setw(20) << "Custom Times(ms)" << std::endl;
	std::cout << std::setw(20) << "--------------------" << std::setw(20) << "------------------" << std::setw(20) << "----------------" << std::endl;
	for (int i = 0; i < 3; i++) {
		std::cout << std::setw(20) << interpolation_methods[i] << std::setw(20) << built_in_times[i] << std::setw(20) << custom_times[i] << std::endl;
	}
	return 0;
}
