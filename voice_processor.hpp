#ifndef _VOICE_PROCESSOR_HPP
#define _VOICE_PROCESSOR_HPP

#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>


class VoiceProcessor{
public:
	const static size_t sample_size = 16; // bit; each sample needs to have this many bits
	const static size_t property_cnt = 12;

	const static size_t max_human_voice_frequency = 280; // Hz; human voice frequency is at most 280 Hz

	// Indexes into the properties array
	const static size_t MEANFREQ=0, SD=1, MEDIAN=2, Q25=3, Q75=4, IQR=5, SKEW=6, KURT=7, SPENT=8, SFM=9, MODE=10, CENTROID=11;

	std::array<double, property_cnt> properties; // The "return" value, input for the neural network


	/** 
	* file should be a 16-bit LPCM raw audio file in the correct endianity,
	* encoded as signed integers with one channel.
	* sample_rate needs to be at least 2*max_human_voice_frequency (but preferably much more)
	*/
	VoiceProcessor(const std::string & file, double sample_length /* s */, size_t sample_rate /* Hz */);

private:
	arma::vec ft_data; // Fourier transform of the raw sound data
	double sample_length; // s
	size_t sample_rate; // Hz

	void read_data(const std::string & file);



	void compute_moment_properties();
	void compute_quantile_properties();
	void compute_spectral_entropy();
	void compute_centroid();
	void compute_spectral_flattness();
	void compute_mode();

	double get_proportion_i(arma::vec & data, double proportion);

	// Frequency "given by" the Fourier transform
	inline double to_khz(double freq){
		return (freq/sample_length)/1000;
	}

};



#endif