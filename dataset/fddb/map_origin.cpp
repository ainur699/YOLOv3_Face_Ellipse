int map()
{
	std::ofstream ofile("D:/Github/pppFaceDetection/dataset/fddb/FDDB.txt");

	std::string image_dir("D:/Datasets/FDDB/originalPics/");
	std::string laber_dir("D:/Datasets/FDDB/FDDB-folds/");
	std::vector<std::string> folders = {
		"FDDB-fold-01-ellipseList.txt",
		"FDDB-fold-02-ellipseList.txt",
		"FDDB-fold-03-ellipseList.txt",
		"FDDB-fold-04-ellipseList.txt",
		"FDDB-fold-05-ellipseList.txt",
		"FDDB-fold-06-ellipseList.txt",
		"FDDB-fold-07-ellipseList.txt",
		"FDDB-fold-08-ellipseList.txt",
		"FDDB-fold-09-ellipseList.txt",
		"FDDB-fold-10-ellipseList.txt"
	};

	
	for (size_t j = 0; j < folders.size(); j++) {
		std::ifstream ifile(laber_dir + folders[j]);

		std::vector<std::string> lines;
		std::string line;
		while (std::getline(ifile, line)) lines.push_back(line);

		int id = 0;
		while (id < lines.size()) {
			std::string image_name = lines[id] + ".jpg";
			int ell_num = std::stoi(lines[id + 1]);
			id += 2;

			std::vector<cv::RotatedRect> ellipses(ell_num);
			for (int i = 0; i < ell_num; i++) {
				std::string ell_str = lines[id + i];

				size_t pos, prev_pos = 0;
				float p[5];
				for (int j = 0; j < 5; j++) {
					pos = ell_str.find_first_of(" ", prev_pos);
					p[j] = std::stof(ell_str.substr(prev_pos, pos - prev_pos));
					prev_pos = pos + 1;
				}

				ellipses[i].size = cv::Size2f(2 * p[1], 2 * p[0]);
				ellipses[i].angle = p[2] > 0 ? p[2] - 0.5f * PI : 0.5f * PI + p[2];
				ellipses[i].center = cv::Point2f(p[3], p[4]);
			}

			id += ell_num;

			///
#if 0
			cv::Mat img = cv::imread(image_dir + image_name);
			for (size_t i = 0; i < ellipses.size(); i++) {
				ellipses[i].angle *= 180.f / PI;
				cv::ellipse(img, ellipses[i], cv::Scalar(0, 255, 0));
			}

			cv::imshow("test", img);
			cv::waitKey();
#else
			ofile << image_name << std::endl;
			ofile << ellipses.size() << std::endl;
			for (size_t i = 0; i < ellipses.size(); i++) {
				ofile << ellipses[i].center.x << " " << ellipses[i].center.y << " " << ellipses[i].size.width << " " << ellipses[i].size.height << " " << ellipses[i].angle << std::endl;
			}
#endif
		}
	}

	return 0;
}