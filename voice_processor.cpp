#include "voice_processor.hpp"
#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>



VoiceProcessor::VoiceProcessor(const std::string & file, double sample_length /* s */, size_t sample_rate /* Hz */):
	sample_length(sample_length), sample_rate(sample_rate) {
	
	if(sample_rate < 2*max_human_voice_frequency) throw std::invalid_argument{"Too low sample rate."};

	read_data(file);

	compute_moment_properties();
	compute_quantile_properties();
	compute_spectral_entropy();
	compute_centroid();
	compute_spectral_flattness();
	compute_mode();

	


}


void VoiceProcessor::read_data(const std::string & file){

	size_t len = (size_t)std::ceil(sample_length*sample_rate);

	std::vector<int16_t> buffer(len);

	std::ifstream in(file, std::ios::binary);

	in.read((char*)&buffer[0], len*sizeof(int16_t));

	in.close();


	arma::vec v(len);

	for(size_t i = 0; i < len; ++i){
		v(i) = (double)buffer[i];
	}


	arma::vec tmp = arma::abs(arma::fft(v));

	// Cut only the lower 0..max_human_voice_frequency Hz;
	ft_data = tmp.subvec(0, (size_t)std::ceil(max_human_voice_frequency*sample_length)); 
}



// Computes the n_th moments of of_what where the probability distribution
// is given by weights (just rescaling by denominator).
double moment(arma::vec of_what, arma::vec & weights, double n_th, double denominator){
	double numerator = arma::accu(weights % (of_what.transform([n_th] (auto x) { return std::pow(x, n_th); })));
	return numerator/denominator;
}

double moment(arma::vec of_what, arma::vec & weights, double n_th){
	return moment(std::move(of_what), weights, n_th, arma::accu(weights));
}

void VoiceProcessor::compute_moment_properties(){
	double weight_sum = arma::accu(ft_data);

	size_t n = ft_data.n_rows;

	arma::vec freqs(n);
	int imbue_cnt = 0;
	freqs.imbue( [&imbue_cnt] () { return imbue_cnt++; });
	// freqs = {0,1,2,3,...,n-1}


	// MEANFREQ
	double meanf = moment(freqs, ft_data, 1, weight_sum);

	properties[MEANFREQ] = to_khz(meanf);

	freqs.transform( [meanf] (auto x) { return x - meanf; }); // Normalize the frequencies for moment calculation

	// STANDARD DEVIATION
	double variance = moment(freqs, ft_data, 2, weight_sum);
	double stdev = sqrt(variance);
	properties[SD] = to_khz(stdev);

	// SKEW
	properties[SKEW] = moment(freqs, ft_data, 3, weight_sum)/(stdev*stdev*stdev);

	// KURTOSIS
	properties[KURT] = moment(freqs, ft_data, 4, weight_sum)/(variance*variance);


}

/**
* Returns the (lineary interpolated) index i such that the cummulative sum of data up to i is exactly proportion
* For example the median is get_proportion_i(data, sum*data)/2).
*/
double VoiceProcessor::get_proportion_i(arma::vec & data, double proportion){
	size_t i = 0;
	for(; i < data.n_rows && data(i) < proportion; ++i, proportion-=data(i));
	if(i == data.n_rows)return -1;
	return i + proportion/data(i);
}

void VoiceProcessor::compute_quantile_properties(){
	double weight_sum = arma::accu(ft_data);
	double q1 = get_proportion_i(ft_data, 0.25*weight_sum),
		median = get_proportion_i(ft_data, 0.5*weight_sum),
		q3 = get_proportion_i(ft_data, 0.75*weight_sum);
	double iqr = q3-q1;

	properties[Q25] = to_khz(q1);
	properties[MEDIAN] = to_khz(median);
	properties[Q75] = to_khz(q3);
	properties[IQR] = to_khz(iqr);
}

void VoiceProcessor::compute_spectral_entropy(){
	arma::vec p = (1/(double)ft_data.n_rows) * (ft_data % ft_data);
	double sum = arma::accu(p);
	properties[SPENT] = -arma::accu(p.transform( [sum] (auto x) {
			double y = x/sum; return y*std::log(y);
	}))/std::log((double)ft_data.n_rows);
}


void VoiceProcessor::compute_centroid(){
	auto rel_amplitudes = ft_data / arma::accu(ft_data);

	size_t n = ft_data.n_rows;

	arma::vec freqs(n);
	int imbue_cnt = 0;
	freqs.imbue( [&imbue_cnt] () { return imbue_cnt++; });

	properties[CENTROID] = to_khz(arma::accu(freqs % rel_amplitudes));
}

void VoiceProcessor::compute_spectral_flattness(){
	size_t n = ft_data.n_rows;

	arma::vec rel_amplitudes = ft_data / arma::accu(ft_data);
	double lg = arma::accu(arma::log(rel_amplitudes));
	properties[SFM] = n*std::exp(lg/n);

}

void VoiceProcessor::compute_mode(){
	properties[MODE] = to_khz(arma::index_max(ft_data));
}