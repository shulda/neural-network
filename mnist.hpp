#ifndef _MNIST_HPP
#define _MNIST_HPP

#include "neural_network.hpp"
#include "gradient_descent.hpp"
#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>


//	MNIST m("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte",
//		"mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");


class MNIST{
public:

	const static size_t img_size = 28*28;
	const static size_t num_of_digits = 10;

private:

	struct GradientDescentParams{
		struct CostFunction : nn::CrossEntropyCostFunction{};

		const static size_t epochs = 15;
		const static size_t batch_size = 10;

		constexpr static double learning_rate = 0.3; // eta
		constexpr static double regularization_param = 0.1; // lambda
	};

	nn::GradientDescent<nn::Network<img_size, 120, num_of_digits>, GradientDescentParams> gd;


public:


	MNIST(const std::string & train_i, const std::string & train_l, const std::string & test_i, const std::string & test_l);


private:

	static const size_t training_size = 60000;
	static const size_t test_size = 10000;

	arma::mat test_data;
	std::vector<uint8_t> test_labels;


	// See http://yann.lecun.com/exdb/mnist/
	std::vector< std::pair< std::array<double, img_size>, uint8_t>> read_data(const std::string & img_f,
							const std::string & labels_f, size_t n);

	void load_training_data(const std::string & img_f, const std::string & labels_f);

	void load_test_data(const std::string & img_f, const std::string & labels_f);

};


#endif