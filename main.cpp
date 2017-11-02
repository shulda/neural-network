#include "voice_processor.hpp"
#include "voice_recognition_net.hpp"
#include "mnist.hpp"



int main(){


	MNIST m("mnist/train-images.idx3-ubyte", "mnist/train-labels.idx1-ubyte",
		"mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte");

	
	// VoiceRecognitionNet m("voice_gender_data");

	// VoiceProcessor vpr("voice/voice.raw", 4, 44100);

	// auto res = m.identify_voice(vpr.properties);

	// std::cout << "IDENTIFIED as MALE with weight " << res.first << ", as FEMALE with weight " << res.second << std::endl;

}	