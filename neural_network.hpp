#ifndef _NEURAL_NETWORK_HPP
#define _NEURAL_NETWORK_HPP

#include <armadillo>
#include <array>
#include <iostream>
#include <type_traits>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <utility>
#include <stdexcept>

// [1] http://neuralnetworksanddeeplearning.com


/**
* Usage: Network<input_layer_size, hidden_layer_size, output_layer_size> n([file_with_weights]);
* n.save(file);
* vector<vector<double> > n.feed_forward(input); // input is vector<vector<double> >, the inner vector
* 	corresponds to the input layer.
*/


namespace nn{

/**
* Could be made variadic and support multiple hidden layers, but
* the learning algorithm is efficient only for one-hidden-layer networks
* anyway, so I chose to keep it simple.
*/
template<size_t is, size_t hs, size_t os>
class Network{
	//Perhaps everything should be public in order to allow third-party learning algorithms?

	const static size_t layers_n = 3;

	const static size_t input_size = is,
		hidden_size = hs,
		output_size = os;

	std::array<size_t, 3> sizes;

	template<class N, class Param> friend class GradientDescent;


	/**
	* Weights, w[i] is the matrix of weights between i-th and (i+1)-th layer,
	* CAUTION!! w[i][j][k] = weight between the k-th neuron in i-th layer and j-th neuron in (i+1)-th layer
	*/
	std::vector<arma::mat> w;

	/**
	* Biases of the i-th layer
	* b[0] is undefined
	*/
	std::vector<arma::vec> b;


	/**
	* Activations of the i-th layer (a[0] corresponds to the input, a[a.size() -1] to the output)
	* arma::mat, because it enables us to process multiple inputs at once.
	*/
	std::vector<arma::mat> a;

	/**
	* Weighed input to i-th layer (see [1]), w[0] is undefined.
	*/
	std::vector<arma::mat> z;


public:
	Network() {
		init();

		fill_randomly();
	}

	Network(const std::string & file){
		init();

		fill_from_file(file);
	}


	void save(const std::string & file){
		save_to_file(file);
	}

	void load(const std::string & file){
		fill_from_file(file);
	}

	std::vector< std::vector<double> > feed_forward(const std::vector< std::vector<double> > & input){
		size_t testcases = input.size();
		if(testcases == 0) return std::vector< std::vector<double> > {};

		size_t size = input[0].size();

		for(auto && x : input){
			if(x.size() != size) throw std::invalid_argument{"The sizes of single inputs do not match."};
		}

		if(size != input_size) throw std::invalid_argument{"Wrong input size."};

		std::vector<double> tmp(testcases*size);

		auto it = tmp.begin();

		for(auto &&x : input){
			it = std::copy(x.begin(), x.end(), it);
		}

		// Copy data from memory
		arma::mat m(&tmp[0], size, testcases, false);

		auto && res = feed_forward(m);

		std::vector< std::vector<double>> ret(res.n_cols);

		for (size_t i = 0; i < res.n_cols; ++i) {
			ret[i] = arma::conv_to< std::vector<double> >::from(res.col(i));
		};

		return std::move(ret);
	}



	arma::mat & feed_forward(arma::mat input){
		if(input.n_rows != input_size) throw std::invalid_argument{"Wrong input size."};

		// When processing more queries at the same time, we need to "copy" the bias vector
		arma::rowvec biases_to_matrix(input.n_cols, arma::fill::ones);

		a[0] = input;

		for(size_t i = 0; i < layers_n-1; ++i){
			a[i+1] = z[i+1] = w[i]*a[i] + b[i+1]*biases_to_matrix;
			a[i+1].transform([] (double x) { return sigmoid(x); });
		}

		return a[layers_n-1];
	}


private:

	static double sigmoid(double x){
		return 1.0 / (1.0 + std::exp(-x));
	}

	// Sets all weights as 1 and biases as 0
	void testing_fill(){
		std::vector<size_t> sizes = { input_size, hidden_size, output_size };

		for(size_t i = 0; i < layers_n-1; ++i){
			w.push_back(arma::mat(sizes[i+1], sizes[i], arma::fill::ones));
		}
		for(size_t i = 0; i < layers_n; ++i){
			b.push_back(arma::vec(sizes[i], arma::fill::zeros));
		}
	}

	void fill_randomly(){
		std::default_random_engine generator;

		std::normal_distribution<double> bias_distribution(0.0, 1.0);

		for(size_t i = 0 ; i < layers_n; ++i){
			b.push_back(arma::vec(sizes[i]));
			b[i].imbue( [&generator, &bias_distribution] () { return bias_distribution(generator); });
		}

		for(size_t i = 0; i < layers_n-1; ++i){
			std::normal_distribution<double> w_distr(0.0, 1.0/std::sqrt((double)sizes[i]));

			w.push_back(arma::mat(sizes[i+1], sizes[i]));
			w[i].imbue( [&generator, &w_distr] () { return w_distr(generator); });
		}
	}


	void fill_from_file(const std::string & file){
		std::ifstream in(file);

		/* Check that the file is compatible with the network topology */
		size_t f_n, tmp;
		in >> f_n;
		if(f_n != layers_n) throw std::runtime_error{"Wrong topology."};

		for(size_t i = 0; i < layers_n; ++i){
			in >> tmp;
			if(tmp != sizes[i]) throw std::runtime_error{"Wrong topology."};
		}

		w.resize(layers_n-1);
		b.resize(layers_n);


		for(size_t i = 0; i < layers_n - 1; ++i){

			w[i].load(in, arma::arma_ascii);
			b[i+1].load(in, arma::arma_ascii);
		}

		in.close();
	}

	void save_to_file(const std::string & file){
		std::ofstream out(file);

		out << layers_n << std::endl;
		for(auto && l : sizes) out << l << " ";
		out << std::endl;

		for(size_t i = 0; i < layers_n - 1; ++i){
			w[i].save(out, arma::arma_ascii);
			b[i+1].save(out, arma::arma_ascii);
		}

		out.close();
	}

	void init(){
		sizes = { is, hs, os };

		a.resize(layers_n);
		z.resize(layers_n);
	}


};



};

#endif