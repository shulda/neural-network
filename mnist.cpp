#include "mnist.hpp"
#include "neural_network.hpp"
#include "gradient_descent.hpp"
#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>





MNIST::MNIST(const std::string & train_i, const std::string & train_l, const std::string & test_i, const std::string & test_l){
	
	load_training_data(train_i, train_l);
	load_test_data(test_i, test_l);

	gd.train( [this] (auto && n, size_t epoch_i) {
		auto && res = n->feed_forward(test_data);
		size_t ok_cnt = 0;

		for(size_t i = 0; i < test_size; ++i){
			uint8_t dig = 11;
			double bst = -100000000;
			for(uint8_t j = 0; j < num_of_digits; ++j){
				if(res(j, i) > bst){
					bst = res(j, i);
					dig = (uint8_t) j;
				}
			}

			if(dig == test_labels[i]){
				++ok_cnt;
			}
		}

		std::cout << "After epoch #"<<epoch_i<<" I classified "<< ok_cnt<<" / "<< test_size << std::endl;

		return false;
	});
}

// See http://yann.lecun.com/exdb/mnist/
std::vector< std::pair< std::array<double, MNIST::img_size>, uint8_t>> MNIST::read_data(const std::string & img_f,
						const std::string & labels_f, size_t n){
	std::ifstream img_in(img_f, std::ios::binary);
	std::ifstream labels_in(labels_f, std::ios::binary);

	img_in.seekg(16, std::ios::beg);
	labels_in.seekg(8, std::ios::beg);

	uint8_t buffer[img_size];
	uint8_t label;

	std::vector< std::pair< std::array<double, img_size>, uint8_t>> res(n);

	for(size_t i = 0; i < n; ++i){

		img_in.read((char*)buffer, img_size);
		labels_in.read((char*)&label, 1);

		res[i].second = label;
		for(size_t j = 0; j < img_size; ++j){
			res[i].first[j] = ((double)buffer[j])/255.0; 
		}

	}

	img_in.close();
	labels_in.close();

	return std::move(res);
}

void MNIST::load_training_data(const std::string & img_f, const std::string & labels_f){

	auto raw = read_data(img_f, labels_f, training_size);

	
	std::array<arma::mat, 2> data;
	data[0].set_size(img_size, training_size);
	data[1].set_size(num_of_digits, training_size);

	for(size_t i = 0; i < training_size; ++i){
		arma::vec imgs(&raw[i].first[0], img_size);
		arma::vec labels(num_of_digits, arma::fill::zeros);
		labels[(size_t)raw[i].second]=1.0;

		data[0].col(i) = imgs;
		data[1].col(i) = labels;
	}

	gd.set_training_data(std::move(data));
}

void MNIST::load_test_data(const std::string & img_f, const std::string & labels_f){

	auto raw = read_data(img_f, labels_f, test_size);

	test_data.set_size(img_size, test_size);
	test_labels.resize(test_size);

	for(size_t i = 0; i < test_size; ++i){
		arma::vec imgs(&raw[i].first[0], img_size);

		test_data.col(i) = imgs;
		test_labels[i] = raw[i].second;
	}
}
