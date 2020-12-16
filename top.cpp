#include "hw_lstm.hpp"
#include "hw_config.h"
#include "r_model_fw_bw.hpp"
#include <stdint.h>
#include "stdint-gcc.h"
#include <hls_stream.h>

//#define AP_INT_MAX_W 6000
//#include <ap_int.h>
/*#ifdef __INT8_TYPE__
typedef __INT8_TYPE__ int8_t;
#endif
#ifdef __INT16_TYPE__
typedef __INT16_TYPE__ int16_t;
#endif
#ifdef __INT32_TYPE__
typedef __INT32_TYPE__ int32_t;
#endif
#ifdef __INT64_TYPE__
typedef __INT64_TYPE__ int64_t;
#endif
#ifdef __UINT8_TYPE__
typedef __UINT8_TYPE__ uint8_t;
#endif
#ifdef __UINT16_TYPE__
typedef __UINT16_TYPE__ uint16_t;
#endif
#ifdef __UINT32_TYPE__
typedef __UINT32_TYPE__ uint32_t;
#endif
#ifdef __UINT64_TYPE__
typedef __UINT64_TYPE__ uint64_t;
#endif*/
void DoCompute(ap_uint<16> numberColumns,
		ap_uint<32> numberBytesRead,
			   ap_uint<DATAWIDTH> *input_buffer,
			   t_fixed_sum_fc *output_buffer)

{
#pragma HLS ALLOCATION instances=DSP48 limit=160 core
	const unsigned int StreamPerColumn = (SIZE_OF_VECTOR*VECTORWIDTH) / DATAWIDTH + (((SIZE_OF_VECTOR*VECTORWIDTH) % DATAWIDTH)>0); // CEILING
	const unsigned int BitPadding = StreamPerColumn*DATAWIDTH - SIZE_OF_VECTOR*VECTORWIDTH;
	const unsigned int LastStreamBits = DATAWIDTH - BitPadding;
	#pragma HLS DATAFLOW

	hls::stream<ap_uint<DATAWIDTH> >output_stream_dma_input("output_stream_dma_input");
#pragma HLS STREAM variable=output_stream_dma_input depth=2

	hls::stream<ap_uint<SIZE_OF_VECTOR*VECTORWIDTH> > output_stream_columns("output_stream_columns");
#pragma HLS STREAM variable=output_stream_columns depth=2

	hls::stream<ap_uint<DATAWIDTH * StreamPerColumn> > stream_column_padded("stream_column_padded");
#pragma HLS STREAM variable=stream_column_padded depth=2

	hls::stream<ap_uint<OUTPUTACTIVATIONHIDDENLAYERWIDTH*NUMBER_OF_NEURONS/PE> > output_stream_hidden_layer("output_stream_hidden_layer");
#pragma HLS STREAM variable=output_stream_hidden_layer depth=2

	hls::stream<ap_uint<OUTPUTACTIVATIONHIDDENLAYERWIDTH * NUMBER_OF_NEURONS> > output_stream_input_streamer("output_stream_input_streamer");
#pragma HLS STREAM variable=output_stream_input_streamer depth=2

	hls::stream<t_fixed_sum_fc> output_stream_mac("output_stream_mac");
#pragma HLS STREAM variable=output_stream_mac depth=2

	hls::stream<t_fixed_sum_fc> output_stream_concatenator("output_stream_concatenator");
#pragma HLS STREAM variable=output_stream_concatenator depth=2

	hls::stream<maxx> output_stream_div_max_per_column("output_stream_div_max_per_column");
#pragma HLS STREAM variable=output_stream_div_max_per_column depth=2

	hls::stream<ap_uint<8> > output_stream_final_labeling("output_stream_final_labeling");
#pragma HLS STREAM variable=output_stream_final_labeling depth=2

	//Mem2Stream<DATAWIDTH>(input_buffer, output_stream_dma_input, numberBytesRead);

	// Converts data widths of streams into multiple or submultiples
	//StreamingDataWidthConverter_Batch<DATAWIDTH, StreamPerColumn * DATAWIDTH, StreamPerColumn>(output_stream_dma_input, stream_column_padded, numberColumns);

	// This cast will remove the padding from the MSBs..casts intput to output
	//StreamingCast< ap_uint<StreamPerColumn * DATAWIDTH>, ap_uint<SIZE_OF_VECTOR*VECTORWIDTH> >(stream_column_padded, output_stream_columns, numberColumns);
	ap_uint<5056>in_tf;
	for(int i=0;i<numberBytesRead/8;i++){
		in_tf=in_tf >> 64;
		in_tf(5055,5055-63)=input_buffer[i];

	}


	for(int i=0;i<numberColumns;i++){
    	output_stream_columns.write(in_tf(49+i*50,i*50));

    }
	HiddenLayer_noPH
	<PE, SIMD_INPUT, SIMD_RECURRENT, t_fixed_image, VECTORWIDTH,
	t_fixed_bgi, BIASWIDTH, t_fixed_wgi, WEIGHTWIDTH, t_fixed_sum_wgi, t_fixed_gix_sum,
	t_fixed_bgf, BIASWIDTH, t_fixed_wgf, WEIGHTWIDTH, t_fixed_sum_wgf, t_fixed_gfx_sum,
	t_fixed_bgo, BIASWIDTH,t_fixed_wgo, WEIGHTWIDTH, t_fixed_sum_wgo, t_fixed_gox_sum,
	t_fixed_bci, BIASWIDTH, t_fixed_wci, WEIGHTWIDTH, t_fixed_sum_wci, t_fixed_ci_gi_mul,
	t_fixed_recurrent, OUTPUTACTIVATIONHIDDENLAYERWIDTH,
	ap_uint<SIZE_OF_VECTOR_TYPEWIDTH>, SIZE_OF_VECTOR,
	ap_uint<NUMBER_OF_NEURONS_TYPEWIDTH>, NUMBER_OF_NEURONS,
	MAX_NUMBER_COLUMNS_TEST_SET,
	t_fixed_state,
	t_fixed_sigma_o, NUMBER_OF_LUT_ETRIES_SIGMOID_1, t_fixed_lut_sigmoid_limit, t_fixed_lut_sigmoid_recip_step,
	t_fixed_tanh_o, NUMBER_OF_LUT_ETRIES_TANH_1, t_fixed_lut_tanh_limit, t_fixed_lut_tanh_recip_step,t_norm_weights,t_norm_bias
	>
	(numberColumns, output_stream_columns, output_stream_hidden_layer,
	bgi,iih_norm_w,iih_norm_b,wgi_ih,ihh_norm_w,ihh_norm_b,wgi_hh,bgf,fih_norm_w,fih_norm_b,wgf_ih,fhh_norm_w,fhh_norm_b,wgf_hh,
	bgo,oih_norm_w,oih_norm_b,wgo_ih,ohh_norm_w,ohh_norm_b,wgo_hh,bci,cih_norm_w,cih_norm_b,wci_ih,chh_norm_w,chh_norm_b,wci_hh, lut_sigmoid_1, lut_tanh_1);

	StreamingDataWidthConverter_Batch<OUTPUTACTIVATIONHIDDENLAYERWIDTH*(NUMBER_OF_NEURONS/PE), OUTPUTACTIVATIONHIDDENLAYERWIDTH * NUMBER_OF_NEURONS, PE>(output_stream_hidden_layer, output_stream_input_streamer, numberColumns);


	OutputLayer
	<
	t_fixed_bfc, FCBIASWIDTH,
	t_fixed_wfc, FCWEIGHTWIDTH,
	t_fixed_recurrent, OUTPUTACTIVATIONHIDDENLAYERWIDTH,
	t_fixed_sum_fc, OUTPUTACTIVATIONOUTPUTLAYERWIDTH,
	ap_uint<NUMBER_OF_NEURONS_TYPEWIDTH>, NUMBER_OF_NEURONS,
	ap_uint<NUMBER_OF_CLASSES_TYPEWIDTH>, NUMBER_OF_CLASSES,
	ap_uint<MAX_NUMBER_COLUMNS_TEST_SET_TYPEWIDTH>
	>
	(bfc, wfc, numberColumns, output_stream_input_streamer, output_stream_mac);



	for(int a=0;a<numberColumns*SIZE_OF_VECTOR;a++){
	 // t_fixed_sum_fc temp;
	  //output_stream_mac.read(temp);
	  output_buffer[a]=output_stream_mac.read();
  }


}

//===================================================================================================================================================================================
// TOP LEVEL
//===================================================================================================================================================================================
void topLevel_BLSTM_CTC(ap_uint<16> numberColumns,
 	   //ap_uint<16> numberColumnsTwice,
	       ap_uint<32> numberBytesRead,

		   ap_uint<DATAWIDTH> *input_buffer,
		   t_fixed_sum_fc *output_buffer)//ap_uint<SIZE_OF_VECTOR*VECTORWIDTH> *input_buffer,
	//ap_uint<OUTPUTACTIVATIONHIDDENLAYERWIDTH * NUMBER_OF_NEURONS> *output_buffer)
		//	t_fixed_sum_fc *output_buffer)
{
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=numberColumns bundle=control
#pragma HLS INTERFACE s_axilite port=numberBytesRead bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=input_buffer bundle=hostmem depth=100
#pragma HLS INTERFACE s_axilite port=input_buffer bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=output_buffer bundle=hostmem depth=100
#pragma HLS INTERFACE s_axilite port=output_buffer bundle=control



#pragma HLS ARRAY_RESHAPE variable=bgi complete dim=1
//#pragma HLS ARRAY_RESHAPE variable=bgi_hh complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgf complete dim=1
//#pragma HLS ARRAY_RESHAPE variable=bgf_hh complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bgo complete dim=1
//#pragma HLS ARRAY_RESHAPE variable=bgo_hh complete dim=1
#pragma HLS ARRAY_RESHAPE variable=bci complete dim=1
//#pragma HLS ARRAY_RESHAPE variable=bci_hh complete dim=1

#pragma HLS ARRAY_RESHAPE variable=wfc complete dim=1

	DoCompute(numberColumns,numberBytesRead,input_buffer, output_buffer);

}
