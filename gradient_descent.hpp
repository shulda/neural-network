#ifndef _GRADIENT_DESCENT_HPP
#define _GRADIENT_DESCENT_HPP


//#define ARMA_NO_DEBUG // For speed

#include <armadillo>
#include <array>
#include <iostream>
#include <type_traits>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <utility>


#include "neural_network.hpp"


// [1] http://neuralnetworksanddeeplearning.com



namespace nn{


struct CrossEntropyCostFunction{
	// a - output, y - expected output
	inline static double f(arma::mat a, arma::mat y){
		return arma::accu(-((y % arma::trunc_log(a)) + ((1-y) % arma::trunc_log(1-a))));
	}
	// See [1]
	inline static arma::mat delta(arma::mat a, arma::mat y, arma::mat){
		return std::move(a-y);
	}
};

struct DefaultParams{

	struct CostFunction : CrossEntropyCostFunction {};

	const static size_t epochs = 30;
	const static size_t batch_size = 10;

	constexpr static double learning_rate = 0.5; // eta
	constexpr static double regularization_param = 0.1; // lambda

};


template<class Net, class Params>
class GradientDescent{

	// {input/output}_size * %_data_size * 2 (two for input/output)
	std::array<arma::mat, 2> training_data;

	size_t data_size; // training data size
	static const size_t input_size = Net::input_size;
	static const size_t output_size = Net::output_size;



public:
	Net n;
	
	GradientDescent() = default;

	GradientDescent(std::array<arma::mat, 2> tr_data):
					training_data(std::move(tr_data)) {
		data_size = (size_t)training_data[0].n_cols;
	}

	void set_training_data(std::array<arma::mat, 2> tr_data){
		training_data = std::move(tr_data);
		data_size = (size_t)training_data[0].n_cols;
	}

	// F after_epoch is a function which takes a pointer to the network and the number of the epoch
	// and returns true if the training should stop. Typically it will be used to
	// compute the success rate on test data after each epoch.
	template<typename F>
	void train(F after_epoch){

		for(size_t ep = 1; ep <= Params::epochs; ++ep){
			err = 0;

			for(size_t i = 0; i < data_size/Params::batch_size; ++i){
				process_mini_batch(i);
			}

			// std::cout << "Avg err: " << err/data_size << std::endl;

			if(after_epoch(&n, ep))break;
		}
	}


private:

	double err;

	// See [1]

	void process_mini_batch(size_t minibatch_i){
		size_t start = Params::batch_size*minibatch_i;
		arma::mat inp = training_data[0].cols(start, start+Params::batch_size-1);
		arma::mat outp = training_data[1].cols(start, start+Params::batch_size-1);

		n.feed_forward(inp);

		std::vector<arma::mat> nabla_b(Net::layers_n);
		std::vector<arma::cube> nabla_w(Net::layers_n);

		arma::mat delta = Params::CostFunction::delta(n.a[n.layers_n-1], outp, n.z[n.layers_n-1]);


		err+=Params::CostFunction::f(n.a[n.layers_n-1], outp);

		nabla_b[n.layers_n-1] = delta;
		nabla_w[n.layers_n-1] = arma::cube(n.sizes[n.layers_n-1], n.sizes[n.layers_n-2], delta.n_cols);

		for(size_t i = 0; i < delta.n_cols; ++i){
			nabla_w[n.layers_n-1].slice(i) = delta.col(i)*(n.a[n.layers_n-2].col(i).t());
		}

		// Yes, this runs only once for three-layer network
		for(size_t lay = 2; lay < n.layers_n; ++lay){
			arma::mat sp = n.a[n.layers_n-2];
			sp.transform([] (double x) { return sigmoid_prime(x); });

			delta = ((n.w[n.layers_n-lay].t()) * delta) % sp;
			nabla_b[n.layers_n-lay] = delta;

			nabla_w[n.layers_n-lay] = arma::cube(n.sizes[n.layers_n-lay], n.sizes[n.layers_n-lay-1], delta.n_cols);

			for(size_t i = 0; i < delta.n_cols; ++i){
				nabla_w[n.layers_n-lay].slice(i) = delta.col(i)*(n.a[n.layers_n-lay-1].col(i).t());
			}

		}

		std::vector<arma::vec> nabla_b_cum(nabla_b.size());
		std::transform(nabla_b.begin()+1, nabla_b.end(), nabla_b_cum.begin()+1, [] ( auto && mat ) { return arma::sum(mat, 1 ); });


		std::vector<arma::mat> nabla_w_cum(nabla_w.size()-1);
		std::transform(nabla_w.begin()+1, nabla_w.end(), nabla_w_cum.begin(), [] ( auto && cube ) { return arma::sum(cube, 2 ); });



		auto nabb = nabla_b_cum.begin()+1;
		for(auto b = n.b.begin()+1; b != n.b.end(); ++b, ++nabb){
			*b -= (Params::learning_rate/Params::batch_size)*(*nabb);
		}

		auto nabw = nabla_w_cum.begin();
		for(auto w = n.w.begin(); w != n.w.end(); ++w, ++nabw){
			*w = (1-Params::learning_rate*(Params::regularization_param/data_size))*(*w)
					-(Params::learning_rate/Params::batch_size)*(*nabw);
		}
	
	}

	// The derivative of sigmoid function (as a function of the result of the sigmoid function for efficiency reasons)
	static double sigmoid_prime(double sigm){
		return sigm*(1-sigm);
	}


};






};

#endif