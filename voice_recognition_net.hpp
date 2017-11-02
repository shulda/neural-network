#ifndef _VOICE_RECOGNITION_NET_HPP
#define _VOICE_RECOGNITION_NET_HPP

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


class VoiceRecognitionNet{
public:


	const static size_t property_cnt = 12; // Also input layer size of the neural network
	const static size_t num_of_sexes = 2; // Output layer size of the neural network


	VoiceRecognitionNet(const std::string & data);

	VoiceRecognitionNet(const std::string & saved_weights, const std::string & saved_normalization_parameters);

	void save_weights(const std::string & file, const std::string & normalization_parameters_file);

	// .first - how certain the network is that the data correspond to a male voice, .second dtto for female
	std::pair<double, double> identify_voice(const std::array<double, property_cnt> & data);

private:

	struct GradientDescentParams{
		struct CostFunction : nn::CrossEntropyCostFunction{};
		
		const static size_t epochs = 200;
		const static size_t batch_size = 10;

		constexpr static double learning_rate = 0.0005; // eta
		constexpr static double regularization_param = 0.003; // lambda
	};


	nn::GradientDescent<nn::Network<property_cnt, 10, num_of_sexes>, GradientDescentParams> gd;

	static const size_t training_size = 3000;
	static const size_t test_size = 168;

	// In the teaching data, there are 20 properties; due to its application,
	// this class needs to support teaching only from the first property_cnt properties.
	static const size_t property_cnt_in_data = 20; 

	arma::mat test_data;
	std::vector<size_t> test_labels;

	std::vector<double> means, stddevs; // for normalization


	std::pair<arma::mat, std::vector<size_t>> read_data(const std::string & f, size_t n);

	void load_data(const std::string & f);
	void compute_normalization_parameters(arma::mat & m);
	void normalize(arma::mat & m);

};



#endif