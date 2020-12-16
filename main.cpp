#include <stdio.h>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
//#include "hls_math.h"
#include "hw_lstm.hpp"
#include "hw_config.h"
#include "r_model_fw_bw.hpp"
//using namespace hls;
int main(void){
	ap_uint<5056>in;
	int raw_input[100]={5,  7,  3, 10,  8,  3, 13,  0, 16,  3,  5, 33,  8,  3,  4,  9,  0, 12,
	         6,  3, 18,  7,  5, 21,  0, 14, 26,  4, 15,  8,  3, 13, 20, 10,  9,  1,
	         3,  8, 20,  1,  3,  5,  1, 13,  3, 14,  7,  2,  6,  3, 16,  8,  7, 12,
	         6,  3,  1, 25, 12, 20,  0,  5, 19,  1,  3, 21, 10, 21,  3,  5, 33,  8,
	         3, 17,  0,  9,  9,  3,  0, 24,  0,  2,  8,  3, 17,  2, 10, 21,  0, 14,
	         3,  0, 16,  3,  8, 20,  1,  3, 21,  7};
	ap_uint<50>in_temp[100];
	for(int i=0;i<100;i++){
		in_temp[i]=pow(2,(float)(raw_input[i]));
        in=in >> 50;
        in(5055,5006)=in_temp[i];

	}
	in=in >> 56;

  ap_uint<64>in_buff[79];
  for(int i=0;i<79;i++){
	  in_buff[i]=in(63+i*64,i*64);

  }

  ap_uint<50>out[100];





	t_fixed_sum_fc out_buff[5000];//t_fixed_sum_fc out_buff[5000];
	t_fixed_sum_fc out_res[100][50];
	 topLevel_BLSTM_CTC(100,
			 632,
			 in_buff,

		out_buff);




   for(int a=0;a<100;a++){
	   for(int b=0;b<50;b++){
		   //out_res[a][b]=out_buff[a*50+b];
		   std::cout<<out_buff[a*50+b]<<",";
	   }
	   std::cout<<std::endl;
   }






	return 0;
}

