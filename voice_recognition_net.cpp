#include "voice_recognition_net.hpp"
#include "neural_network.hpp"
#include "gradient_descent.hpp"
#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <stdexcept>



VoiceRecognitionNet::VoiceRecognitionNet(const std::string & data){
	
	load_data(data);

	gd.train( [this] (auto && n, size_t epoch_i) {
		auto && res = n->feed_forward(test_data);
		size_t ok_cnt = 0;

		for(size_t i = 0; i < test_size; ++i){
			size_t sex = 2;
			double bst = -100000000;
			for(size_t j = 0; j < num_of_sexes; ++j){
				if(res(j, i) > bst){
					bst = res(j, i);
					sex = j;
				}
			}
			if(sex == test_labels[i]){
				++ok_cnt;
			}
		}

		std::cout << "After epoch #"<<epoch_i<<" I classified "<< ok_cnt<<" / "<< test_size << std::endl;

		return false;
	});

	
}


VoiceRecognitionNet::VoiceRecognitionNet(const std::string & saved_weights, const std::string & saved_normalization_parameters){
	gd.n.load(saved_weights);

	std::ifstream nin(saved_normalization_parameters);
	
	means.resize(property_cnt);
	stddevs.resize(property_cnt);

	for(size_t i = 0; i < property_cnt; ++i){
		if(!(nin >> means[i])) throw std::runtime_error{"Bad normalization parameters file."};
	}

	for(size_t i = 0; i < property_cnt; ++i){
		if(!(nin >> stddevs[i])) throw std::runtime_error{"Bad normalization parameters file."};
	}

	nin.close();
}

void VoiceRecognitionNet::save_weights(const std::string & weigths_file, const std::string & normalization_parameters_file){
	gd.n.save(weigths_file);

	std::ofstream nout(normalization_parameters_file);
	
	for(auto && m : means){
		nout << m << " ";
	}
	nout << std::endl;

	for(auto && s : stddevs){
		nout << s << " ";
	}
	nout << std::endl;

	nout.close();
}



std::pair<arma::mat, std::vector<size_t>> VoiceRecognitionNet::read_data(const std::string & f, size_t n){
	std::ifstream in(f);

	arma::mat retmat(property_cnt, n);
	std::vector<size_t> labels(n);

	for(size_t i = 0; i < n; ++i){

		auto && cl = retmat.col(i);
		double tmp;

		for(size_t j=0; j < property_cnt_in_data; ++j){
			in >> tmp;
			// I can not compute all properties in the data file, I can only compute first twelve of them
			// hence it is necessary to ignore the rest of each line.
			if(j < property_cnt)cl(j) = tmp;
		}
		in >> labels[i];

	}

	in.close();


	compute_normalization_parameters(retmat);
	normalize(retmat);

	return {std::move(retmat), std::move(labels)};
}

void VoiceRecognitionNet::load_data(const std::string & f){

	auto raw = read_data(f, training_size+test_size);

	// Training data first


	std::array<arma::mat, 2> data;
	data[0] = raw.first.submat(0, 0, property_cnt-1, training_size-1);
	data[1].set_size(num_of_sexes, training_size);

	for(size_t i = 0; i < training_size; ++i){
		arma::vec labels(num_of_sexes, arma::fill::zeros);
		labels[raw.second[i]]=1.0;

		data[1].col(i) = labels;
	}

	gd.set_training_data(std::move(data));


	// Then test data

	test_data = raw.first.submat(0, training_size, property_cnt-1, raw.first.n_cols-1);
	test_labels.resize(test_size);

	std::copy(raw.second.begin()+training_size, raw.second.end(), test_labels.begin());

}

// We need inputs to be roughly from the interval [0,1]
// z-score normalization seems to work better than [min,max] -> [0,1] normalization
void VoiceRecognitionNet::compute_normalization_parameters(arma::mat & m){
	means.clear(); stddevs.clear();
	means.resize(property_cnt); stddevs.resize(property_cnt);

	arma::mat mean_mat = arma::mean(m, 1);
	arma::mat stddev_mat = arma::stddev(m, 0, 1);
	for(size_t i = 0; i < property_cnt; ++i){
		means[i] = mean_mat(i,0);
		stddevs[i] = stddev_mat(i,0);
	}
}

void VoiceRecognitionNet::normalize(arma::mat & m){

	for(size_t i = 0; i < m.n_rows; ++i){

		double mean = means[i];
		double stddev = stddevs[i];

		m.row(i).transform( [mean, stddev] (auto x) {
			return 0.5 + (x-mean)/(2*stddev);
		});
	}
}


std::pair<double, double> VoiceRecognitionNet::identify_voice(const std::array<double, property_cnt> & data){
	std::vector<double> v(data.begin(), data.end());

	auto res = gd.n.feed_forward(std::vector<std::vector<double>>{v});

	return {res[0][0], res[0][1]};
}