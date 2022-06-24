#include "pc_pre_processor.h"

#define _SEQ 1

void callback_pointcloud_lidar(const sensor_msgs::PointCloud2ConstPtr& msg_pc)
{
	// cout << "call back lidar" << endl;
	// ROS_INFO("LiDAR Processing Start");

	ros::Time time = msg_pc->header.stamp;
	// ROS_INFO("sec = %d", time.sec);
    	// ROS_INFO("nsec = %d", time.nsec);
	string time_in_string = to_string_with_sec_nsec(time.sec, time.nsec);

	if(!is_img_data_inserted)
		return;

	// ----------------------- [ Buffer ] --------------------------------
	pcl::PointCloud<OusterPointType>::Ptr p_pc_lidar(new PointCloud<OusterPointType>);
	pcl::fromROSMsg(*msg_pc, *p_pc_lidar);

	std::ostringstream file_name;
	file_name << "point_cloud/seq_" << to_string(_SEQ) << "/pc_" << time_in_string << ".pcd";
	pcl::io::savePCDFileASCII(file_name.str(), *p_pc_lidar);

	cv::Mat mat_img_frontal = p_cv_img->image.clone();

	std::stringstream ss_img;
	ss_img << "frontal_img/seq_" << to_string(_SEQ) << "/frontal_img_" << time_in_string << ".jpg";
	imwrite(ss_img.str(), mat_img_frontal);

	// ----------------------- [ Msg ] --------------------------------
	lidar_msgs::PillarTensorTraining msg_training_data;
	msg_training_data.time_in_string = time_in_string;

	// ----------------------- [ ROI Filtering ] --------------------------------
	if(is_x)
	{
		pcl::PassThrough<OusterPointType> pass;
		pass.setInputCloud(p_pc_lidar);
		pass.setFilterFieldName("x");
		pass.setFilterLimits(x_min, x_max);
		pass.filter(*p_pc_lidar);
	}
	
	if(is_y)
	{
		pcl::PassThrough<OusterPointType> pass;
		pass.setInputCloud(p_pc_lidar);
		pass.setFilterFieldName("y");
		pass.setFilterLimits(y_min, y_max);
		pass.filter(*p_pc_lidar);
	}

	if(is_z)
	{
		pcl::PassThrough<OusterPointType> pass;
		pass.setInputCloud(p_pc_lidar);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(z_min, z_max);
		pass.filter(*p_pc_lidar);
	}

	if(is_i)
	{
		pcl::PassThrough<OusterPointType> pass;
		pass.setInputCloud(p_pc_lidar);
		pass.setFilterFieldName("intensity");
		pass.setFilterLimits(i_min, i_max);
		pass.filter(*p_pc_lidar);
	}

	if(is_r)
	{
		pcl::PointCloud<OusterPointType>::Ptr p_pc_lidar_temp(new PointCloud<OusterPointType>);

		uint16_t cast_r_min = uint16_t(r_min);
		uint16_t cast_r_max = uint16_t(r_max);
		uint16_t cast_r = uint16_t(0);

		for(int i=0; i<p_pc_lidar->points.size(); i++)
		{
			OusterPointType point_ouster = p_pc_lidar->points[i];
			cast_r = point_ouster.reflectivity;
			if((cast_r >= cast_r_min) && (cast_r <= cast_r_max))
			{
				OusterPointType point_temp;
				point_temp.x = point_ouster.x;
				point_temp.y = point_ouster.y;
				point_temp.z = point_ouster.z;
				point_temp.intensity = point_ouster.intensity;
				point_temp.reflectivity = point_ouster.reflectivity;
				p_pc_lidar_temp->push_back(point_temp);
			}
		}

		p_pc_lidar->clear();
		*p_pc_lidar += *p_pc_lidar_temp;
		p_pc_lidar_temp->clear();
	}

	// ----------------------- [ Make Pillar Tensor ] --------------------------------
	float max_intensity_data = 128.f;

	int n_pc_num = p_pc_lidar->points.size();

	// Hash map
	std::unordered_map<std::pair<uint32_t, uint32_t>, std::vector<PillarPoint>, IntPairHash> map;

	// Pillar tensor
	for(int i=0; i<n_pc_num; i++)
	{
		OusterPointType point_ouster = p_pc_lidar->points[i];
		
		// Tensor Coordinate
		uint32_t x_idx = static_cast<uint32_t>(std::floor((point_ouster.x - x_min)/x_step)); // 0 ~ 47
		uint32_t y_idx = static_cast<uint32_t>(std::floor((point_ouster.y - y_min)/y_step)); // 0 ~ 47

		if(point_ouster.intensity >= max_intensity_data)
			point_ouster.intensity = max_intensity_data;

		PillarPoint p = {
			point_ouster.x,
			point_ouster.y,
			point_ouster.z,
			point_ouster.intensity/max_intensity_data,	// 1로 normalize
			0.0f,
			0.0f,
			0.0f,
		};

		map[{x_idx, y_idx}].emplace_back(p);
	}

	float*** pppTensor = new float**[n_max_pillars];
	
	for(int i=0; i<n_max_pillars; i++)
	{
		pppTensor[i] = new float*[n_max_points_per_pillar];
		
		for(int j=0; j<n_max_points_per_pillar; j++)
		{
			pppTensor[i][j] = new float[n_in_features];

			// 0.0f로 초기화하는게 매우 매우 매우 중요하다!!!!!!!!!!!!!!!!!!!!!!
			for(int k=0; k<n_in_features; k++)
			{
				pppTensor[i][j][k] = 0.0f;
			}
		}
	}

	int** ppIndices = new int*[n_max_pillars];

	for(int i=0; i<n_max_pillars; i++)
	{
		ppIndices[i] = new int[3];

		for(int j=0; j<2; j++)
		{
			ppIndices[i][j] = 0;
		}
	}

	int pillarId = 0;
	for(auto& pair: map)
	{
		// if (pillarId >= n_max_pillars)
		// {
		// 	break;
		// }

		float x_mean = 0.0f;
		float y_mean = 0.0f;
		float z_mean = 0.0f;

		int n_point_in_pillar_size = 0;
		for (const auto& p: pair.second)
		{
			x_mean += p.x;
			y_mean += p.y;
			z_mean += p.z;

			n_point_in_pillar_size++;
		}

		x_mean /= pair.second.size();
		y_mean /= pair.second.size();
		z_mean /= pair.second.size();

		// based on mean
		for (auto& p: pair.second)
		{
			p.xc = p.x - x_mean;
			p.yc = p.y - y_mean;
			p.zc = p.z - z_mean;
		}

		int n_pillar_x_idx = static_cast<int>(std::floor((x_mean - x_min) / x_step));
		int n_pillar_y_idx = static_cast<int>(std::floor((y_mean - y_min) / y_step));
		float z_mid = (z_max - z_min) * 0.5f;
		
		// m_indices
		ppIndices[pillarId][1] = n_pillar_x_idx;
		ppIndices[pillarId][2] = n_pillar_y_idx;

		if(is_sort_with_z)
			sort(pair.second.begin(), pair.second.end(), isSortWithZ);

		if(is_sort_with_intensity)
			sort(pair.second.begin(), pair.second.end(), isSortWithIntensity);
		
		if(_DEBUG)
		{
			int i =0;
			cout << "=======================================================" << endl;
			cout << "x = " << n_pillar_x_idx << ", y = " << n_pillar_y_idx << ", number of points = " << pair.second.size() << endl;
			for(const auto& p: pair.second)
			{
				cout << "" << i << "th : i = " << p.intensity << ", z = " << p.z << " / ";
				i++;
			}
			cout << endl << "================end====================================" << endl << endl;
		}
		

		int pointId = 0;

		for(const auto& p: pair.second)
		{
			if(pointId >= n_max_points_per_pillar)
			{
				break;
			}

			// m_tensor
			pppTensor[pillarId][pointId][0] = p.x - (n_pillar_x_idx*x_step + x_min);
			pppTensor[pillarId][pointId][1] = p.y - (n_pillar_y_idx*y_step + y_min);
			pppTensor[pillarId][pointId][2] = p.z - z_mid;
			pppTensor[pillarId][pointId][3] = p.intensity;
			pppTensor[pillarId][pointId][4] = p.xc;
			pppTensor[pillarId][pointId][5] = p.yc;
			pppTensor[pillarId][pointId][6] = p.zc;

			pointId++;
		}
		
		pillarId++;
	}

	// Pillar Tensor Msg
	msg_training_data.data_f.clear();
	for(int i=0; i<n_max_pillars; i++)
	{
		for(int j=0; j<n_max_points_per_pillar; j++)
		{
			for(int k=0; k<n_in_features; k++)
			{
				msg_training_data.data_f.push_back(pppTensor[i][j][k]);
			}
		}
	}

	for(int i=0; i<n_max_pillars; i++)
	{
		for(int j=0; j<n_max_points_per_pillar; j++)
		{
			delete[] pppTensor[i][j];
		}
		delete[] pppTensor[i];
	}
	delete pppTensor;

	msg_training_data.layout_f.dim.push_back(std_msgs::MultiArrayDimension());
	msg_training_data.layout_f.dim.push_back(std_msgs::MultiArrayDimension());
	msg_training_data.layout_f.dim.push_back(std_msgs::MultiArrayDimension());

	msg_training_data.layout_f.dim[0].size = n_max_pillars;
	msg_training_data.layout_f.dim[1].size = n_max_points_per_pillar;
	msg_training_data.layout_f.dim[2].size = n_in_features;

	msg_training_data.layout_f.dim[0].stride = n_max_pillars*n_max_points_per_pillar*n_in_features;
	msg_training_data.layout_f.dim[1].stride = n_max_points_per_pillar*n_in_features;
	msg_training_data.layout_f.dim[2].stride = n_in_features;

	msg_training_data.data_n.clear();

	for(int i=0; i<n_max_pillars; i++)
	{
		msg_training_data.data_n.push_back(ppIndices[i][0]);
		msg_training_data.data_n.push_back(ppIndices[i][1]);
		msg_training_data.data_n.push_back(ppIndices[i][2]);
	}

	msg_training_data.layout_n.dim.push_back(std_msgs::MultiArrayDimension());
	msg_training_data.layout_n.dim.push_back(std_msgs::MultiArrayDimension());

	msg_training_data.layout_n.dim[0].size = n_max_pillars;
	msg_training_data.layout_n.dim[1].size = 3;

	msg_training_data.layout_n.dim[0].stride = n_max_pillars*3;
	msg_training_data.layout_n.dim[1].stride = 3;
	
	// ----------------------- [ Make BEV Image ] -------------------------------- 
	// Image Create
	int img_width = 576;
	int img_height = 1152;
	float pixel_step = 0.04;
	float max_intensity_input = 128.f;

	float** ppBEV = new float*[img_height];
	
	for(int i=0; i<img_height; i++)
	{
		ppBEV[i] = new float[img_width];
		for(int j=0; j<img_width; j++)
		{
			ppBEV[i][j] = 0.f;
		}
	}

	for(int i=0; i<n_pc_num; i++)
	{
		OusterPointType point_ouster = p_pc_lidar->points[i];

		// Intensity Filtering 0~max_intensity -> 0~1, max_intensity~ -> 1
		float intensity = point_ouster.intensity;
		if(intensity >= max_intensity_input)
			intensity = max_intensity_input;

		// Tensor Coordinate in pixel grid
		int x_idx = static_cast<int>(std::floor((point_ouster.x - x_min)/pixel_step)); // 0 ~ 1151 : 1152 in Y_img
		int y_idx = static_cast<int>(std::floor((point_ouster.y - y_min)/pixel_step)); // 0 ~ 575 : 576 in X_img

		// Image Coordinate
		int x_img_idx = img_width - y_idx - 1;
		int y_img_idx = img_height - x_idx - 1;

		ppBEV[y_img_idx][x_img_idx] = point_ouster.intensity;
	}

	msg_training_data.bev_f.clear();

	for(int i=0; i<img_height; i++)
	{
		for(int j=0; j<img_width; j++)
		{
			msg_training_data.bev_f.push_back(ppBEV[i][j]);
		}
	}

	for(int i=0; i<img_height; i++)
	{
		delete[] ppBEV[i];
	}
	delete[] ppBEV;

	msg_training_data.layout_bev_f.dim.push_back(std_msgs::MultiArrayDimension());
	msg_training_data.layout_bev_f.dim.push_back(std_msgs::MultiArrayDimension());

	msg_training_data.layout_bev_f.dim[0].size = img_height;
	msg_training_data.layout_bev_f.dim[1].size = img_width;

	msg_training_data.layout_bev_f.dim[0].stride = img_height*img_width;
	msg_training_data.layout_bev_f.dim[1].stride = img_width;

	// ----------------------- [ Labeling용 BEV 이미지 ] --------------------------------
	cv::Mat matBEV;

	// Size(Width, Height), 8bit Unsigned Channel3
	matBEV.create(cv::Size(img_width, img_height), CV_8UC3);
	matBEV = cv::Scalar(0); // 0으로 초기화 (검은색 이미지)

	if(_DEBUG)
	{
		// cv::imshow("img checker", matBEV);
		// cv::waitKey(0);
	}

	float max_intensity = 100.f; // 100
	float red_intensity = max_intensity*0.1;
	float green_intensity = max_intensity*0.3;
	int red_radius = 2;
	int green_radius = 1;

	// gray
	for(int i=0; i<n_pc_num; i++)
	{
		OusterPointType point_ouster = p_pc_lidar->points[i];

		// Intensity Filtering 0~max_intensity -> 0~1, max_intensity~ -> 1
		float intensity = point_ouster.intensity;
		if(intensity >= max_intensity)
			intensity = max_intensity;

		// Overlap in gray
		if(intensity >= red_intensity) // gray
		{
			continue;
		}

		// 256에 대해서 Normalise
		unsigned char uc_intensity = static_cast<unsigned char>(std::floor(256.f*intensity/max_intensity));
		unsigned char color[3] = {0, }; // BGR

		if(intensity < red_intensity) // gray
		{
			color[0] = uc_intensity; color[1] = uc_intensity; color[2] = uc_intensity;
		}
		else if(intensity >= red_intensity && intensity < green_intensity) // blue
		{
			color[1] = uc_intensity;
		}
		else // larger than 4/3 red
		{
			color[2] = uc_intensity;
		}

		// Tensor Coordinate in pixel grid
		int x_idx = static_cast<int>(std::floor((point_ouster.x - x_min)/pixel_step)); // 0 ~ 959 : 960 in Y
		int y_idx = static_cast<int>(std::floor((point_ouster.y - y_min)/pixel_step)); // 0 ~ 479 : 480 in X
		
		// Image Coordinate
		int x_img_idx = img_width - y_idx - 1;
		int y_img_idx = img_height - x_idx - 1;

		matBEV.at<cv::Vec3b>(y_img_idx, x_img_idx)[0] = color[0];
		matBEV.at<cv::Vec3b>(y_img_idx, x_img_idx)[1] = color[1];
		matBEV.at<cv::Vec3b>(y_img_idx, x_img_idx)[2] = color[2];
	}

	// red
	for(int i=0; i<n_pc_num; i++)
	{
		OusterPointType point_ouster = p_pc_lidar->points[i];

		// Intensity Filtering 0~max_intensity -> 0~1, max_intensity~ -> 1
		float intensity = point_ouster.intensity;
		if(intensity >= max_intensity)
			intensity = max_intensity;

		// Gray
		if(intensity < red_intensity || intensity >= green_intensity) // gray
		{
			continue;
		}

		// 256에 대해서 Normalise
		unsigned char uc_intensity = static_cast<unsigned char>(std::floor(256.f*intensity/max_intensity));
		unsigned char color[3] = {0, }; // BGR

		if(intensity < red_intensity) // gray
		{
			color[0] = uc_intensity; color[1] = uc_intensity; color[2] = uc_intensity;
		}
		else if(intensity >= red_intensity && intensity < green_intensity) // green
		{
			color[1] = uc_intensity;
		}
		else
		{
			color[2] = uc_intensity;
		}

		// Tensor Coordinate in pixel grid
		int x_idx = static_cast<int>(std::floor((point_ouster.x - x_min)/pixel_step)); // 0 ~ 959 : 960 in Y
		int y_idx = static_cast<int>(std::floor((point_ouster.y - y_min)/pixel_step)); // 0 ~ 479 : 480 in X
		
		// Image Coordinate
		int x_img_idx = img_width - y_idx - 1;
		int y_img_idx = img_height - x_idx - 1;

		cv::circle(matBEV, cv::Point(x_img_idx,y_img_idx), red_radius, cv::Scalar(0,0,uc_intensity), CV_FILLED);
	}

	// green
	for(int i=0; i<n_pc_num; i++)
	{
		OusterPointType point_ouster = p_pc_lidar->points[i];

		// Intensity Filtering 0~max_intensity -> 0~1, max_intensity~ -> 1
		float intensity = point_ouster.intensity;
		if(intensity >= max_intensity)
			intensity = max_intensity;

		// Gray
		if(intensity < green_intensity) // gray
		{
			continue;
		}

		// 256에 대해서 Normalise
		unsigned char uc_intensity = static_cast<unsigned char>(std::floor(256.f*intensity/max_intensity));
		unsigned char color[3] = {0, }; // BGR

		if(intensity < red_intensity) // gray
		{
			color[0] = uc_intensity; color[1] = uc_intensity; color[2] = uc_intensity;
		}
		else if(intensity >= red_intensity && intensity < green_intensity) // green
		{
			color[1] = uc_intensity;
		}
		else
		{
			color[2] = uc_intensity;
		}

		// Tensor Coordinate in pixel grid
		int x_idx = static_cast<int>(std::floor((point_ouster.x - x_min)/pixel_step)); // 0 ~ 959 : 960 in Y
		int y_idx = static_cast<int>(std::floor((point_ouster.y - y_min)/pixel_step)); // 0 ~ 479 : 480 in X
		
		// Image Coordinate
		int x_img_idx = img_width - y_idx - 1;
		int y_img_idx = img_height - x_idx - 1;

		cv::circle(matBEV, cv::Point(x_img_idx,y_img_idx), green_radius, cv::Scalar(0,uc_intensity,0), CV_FILLED);
	}

	// matBEV
	cv_bridge::CvImage cv_img_bev, cv_img_frontal;
	cv_img_bev.encoding = sensor_msgs::image_encodings::BGR8;
	cv_img_bev.image = matBEV;
	
	// matFrontal
	cv_img_frontal.encoding = sensor_msgs::image_encodings::BGR8;
	cv_img_frontal.image = mat_img_frontal;

	//
	sensor_msgs::Image msg_img_bev, msg_img_frontal;
	
	// toMSG
	msg_training_data.n_file_idx = n_file_idx;
	cv_img_frontal.toImageMsg(msg_img_bev);
	msg_training_data.img_frontal = msg_img_bev;
	cv_img_bev.toImageMsg(msg_img_frontal);
	msg_training_data.img_bev = msg_img_frontal;

	n_file_idx++;

	// ----------------------- [ Publish ] --------------------------------
	sensor_msgs::PointCloud2 msg_pc_out;
	pcl::toROSMsg(*p_pc_lidar, msg_pc_out);
	msg_pc_out.header.frame_id = "/os_lidar";
	pub_pc_filtered.publish(msg_pc_out);

	pub_tr_data.publish(msg_training_data);
	
	// ROS_INFO("LiDAR Processing End");
}

void callback_img_camera(const sensor_msgs::CompressedImageConstPtr& msg_img)
{
	try
	{
		p_cv_img = cv_bridge::toCvCopy(msg_img, sensor_msgs::image_encodings::BGR8);
	}
	catch(const std::exception& e)
	{
		ROS_ERROR("image callback error: %s", e.what());
		return;
	}

	if(!is_img_data_inserted)
		is_img_data_inserted = true;
}

void callback_params(const lidar_msgs::Params& msg_params)
{
	cout << "callback params" << endl;

	x_min = msg_params.x_min;
	x_max = msg_params.x_max;
	y_min = msg_params.y_min;
	y_max = msg_params.y_max;
	z_min = msg_params.z_min;
	z_max = msg_params.z_max;
	i_min = msg_params.i_min;
	i_max = msg_params.i_max;
	r_min = msg_params.r_min;
	r_max = msg_params.r_max;

	is_x = msg_params.is_x;
	is_y = msg_params.is_y;
	is_z = msg_params.is_z;
	is_i = msg_params.is_i;
	is_r = msg_params.is_r;
}

// void callback_timer(const ros::TimerEvent& timer_event)
// {

// }

int main(int argc, char** argv)
{
	// Init ROS
	ros::init(argc, argv, "point_pillars");
	ros::NodeHandle nh;

	// subs, publ
	ros::Subscriber sub_pc_lidar = nh.subscribe(PC_LIDAR, 1, callback_pointcloud_lidar); // 10 Hz
	ros::Subscriber sub_img_camera = nh.subscribe(IMG_CAM, 1, callback_img_camera); // 40 Hz
	ros::Subscriber sub_params = nh.subscribe("/lidar_params", 1, callback_params);
	pub_pc_filtered = nh.advertise<sensor_msgs::PointCloud2>("/pc_filtered", 1);
	pub_tr_data = nh.advertise<lidar_msgs::PillarTensorTraining>("/pillar_tensor_training", 1);

	// spin
	ros::spin();
}
