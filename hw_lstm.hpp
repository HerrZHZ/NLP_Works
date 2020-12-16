#ifndef HW_LSTM_HPP
#define HW_LSTM_HPP
//#define AP_INT_MAX_W 6000
#include <stdio.h>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
//#include <cstdint>
//#include <cmath>
#include "stdint-gcc.h"
#include "hls_math.h"
using namespace hls;

#include "activations.hpp"

#define CASSERT_DATAFLOW(x) ;
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
//===========================================================================================================================================================================================
// TAKEN FROM FINN
//===========================================================================================================================================================================================
template<unsigned int DataWidth>
void Mem2Stream(ap_uint<DataWidth> * in, stream<ap_uint<DataWidth> > & out, const unsigned int numBytes)
{
	CASSERT_DATAFLOW(DataWidth % 8 == 0);
	const unsigned int numWords = numBytes / (DataWidth / 8);
	CASSERT_DATAFLOW(numWords != 0);
	for (unsigned int i = 0; i < numWords; i++)
	{
#pragma HLS PIPELINE II=1
		ap_uint<DataWidth> e = in[i];
		out.write(e);
	}
}

template<typename InT, typename OutT>
void StreamingCast(stream<InT> & in, stream<OutT> & out, unsigned int numReps)
{
  for(unsigned int i = 0; i < numReps; i++)
  {
#pragma HLS PIPELINE II=1
	out.write((OutT) in.read());
  }
}

template<unsigned int InWidth, unsigned int OutWidth, unsigned int NumInWords>
void StreamingDataWidthConverter_Batch(stream<ap_uint<InWidth> > & in, stream<ap_uint<OutWidth> > & out, const unsigned int numReps)
{
	if (InWidth > OutWidth)
	{
		// emit multiple output words per input word read
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		const unsigned int outPerIn = InWidth / OutWidth;
		const unsigned int totalIters = NumInWords * outPerIn * numReps;
		unsigned int o = 0;
		ap_uint<InWidth> ei = 0;
		for (unsigned int t = 0; t < totalIters; t++)
		{
	#pragma HLS PIPELINE II=1
			// read new input word if current out count is zero
			if (o == 0)
				ei = in.read();
			// pick output word from the rightmost position
			ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
			out.write(eo);
			// shift input to get new output word for next iteration
			ei = ei >> OutWidth;
			// increment written output count
			o++;
			// wraparound indices to recreate the nested loop structure
			if (o == outPerIn)
			{
				o = 0;
			}
		}
	}
	else if (InWidth == OutWidth)
	{
		// straight-through copy
		for (unsigned int i = 0; i < NumInWords * numReps; i++)
		{
	#pragma HLS PIPELINE II=1
			ap_uint<InWidth> e = in.read();
			out.write(e);
		}

	}
	else
	{ // InWidth < OutWidth
		// read multiple input words per output word emitted
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		//const unsigned int inPerOut = OutWidth / InWidth;
		const ap_uint<8> inPerOut = OutWidth / InWidth;
		const unsigned int totalIters = NumInWords * numReps;
		//unsigned int i = 0;
		ap_uint<8> i = 0;
		ap_uint<OutWidth> eo = 0;
		for (unsigned int t = 0; t < totalIters; t++)
		{
	#pragma HLS PIPELINE II=1
			// read input and shift into output buffer
			ap_uint<InWidth> ei = in.read();
			eo = eo >> InWidth;
			eo(OutWidth - 1, OutWidth - InWidth) = ei;
			// increment read input count
			i++;
			// wraparound logic to recreate nested loop functionality
			if (i == inPerOut)
			{
				i = 0;
				out.write(eo);
			}
		}
	}
}

//===================================================================================================================================================================================
// OUTPUT LAYER
//===================================================================================================================================================================================
template
<
typename Bias_fc_t,
unsigned int BiasWidth_fc,
typename Weight_fc_t,
unsigned int WeightWidth_fc,
typename OutputActivationHiddenLayer_t,
unsigned int OutputActivationHiddenLayerWidth,
typename OutputActivationOutputLayer_t,
unsigned int OutputActivationOutputLayerWidth,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t
>
void OutputLayer(const Bias_fc_t biases[NumberOutputUnits],
				 const Weight_fc_t weights[NumberHiddenUnits][NumberOutputUnits],
				 NumberColumns_t numberOfColumns,
		 	 	 hls::stream<ap_uint<OutputActivationHiddenLayerWidth * NumberHiddenUnits> > &input_stream,
				 hls::stream<OutputActivationOutputLayer_t> &output_stream)
{
	ap_uint<OutputActivationHiddenLayerWidth * NumberHiddenUnits> input_stream_temp;
	OutputActivationOutputLayer_t mul[NumberHiddenUnits];
	OutputActivationOutputLayer_t sum;

	#pragma HLS ARRAY_PARTITION variable=mul complete dim=1

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
			input_stream.read(input_stream_temp);

			for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
			{
			#pragma HLS PIPELINE II=1

				Bias_fc_t bias = biases[currentClass];//[count * NumberOutputUnits + currentClass];
				//Bias_fc_t bias = *reinterpret_cast<Bias_fc_t *>(&bias_temp);

				sum = (OutputActivationOutputLayer_t)bias;

				for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits; currentHiddenUnit++)
				{
				#pragma HLS UNROLL

					ap_int<OutputActivationHiddenLayerWidth> input_temp = input_stream_temp((currentHiddenUnit + 1) * OutputActivationHiddenLayerWidth - 1, currentHiddenUnit * OutputActivationHiddenLayerWidth);
					OutputActivationHiddenLayer_t input = *reinterpret_cast<OutputActivationHiddenLayer_t *>(&input_temp);
					Weight_fc_t weigth = weights[currentHiddenUnit][currentClass];//[count * NumberOutputUnits + currentClass];
					//Weight_fc_t weigth = *reinterpret_cast<Weight_fc_t *>(&weigth_temp);
					/*if(currentClass==0)
					 {std::cout<<currentHiddenUnit<<":"<<input<<std::endl;}*/
					mul[currentHiddenUnit] = input * weigth;
				}

				for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits; currentHiddenUnit++)
				{
				#pragma HLS UNROLL

					sum += mul[currentHiddenUnit];
				}

				output_stream.write(sum);

			}
//		}
	}
}
//===================================================================================================================================================================================
// MEMORY ROUTER
//===================================================================================================================================================================================
template
<
typename OutputActivationOutputLayer_t,
unsigned int OutputActivationOutputLayerWidth,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t,
unsigned int MaxNumberColumns
>
void Concatenator(NumberColumns_t numberOfColumns,
			       hls::stream<OutputActivationOutputLayer_t> & input_stream,
			       hls::stream<OutputActivationOutputLayer_t> & output_stream)
{
	static OutputActivationOutputLayer_t fw_bw_mem[MaxNumberColumns * NumberOutputUnits];
	OutputActivationOutputLayer_t input, mem, sum;
	//ap_uint<OutputActivationOutputLayerWidth> output;

	for(ap_int<16> currentColumn = numberOfColumns - 1; currentColumn >= 0; currentColumn--)
	{
		for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
		{
		#pragma HLS PIPELINE II=1 rewind

		#pragma HLS loop_flatten off

			input_stream.read(input);

			ap_uint<32> offset = currentColumn * NumberOutputUnits + currentClass;
			fw_bw_mem[offset] = input;
		}
	}

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
		{
		#pragma HLS PIPELINE II=1 rewind

		#pragma HLS loop_flatten off

			input_stream.read(input);

			ap_uint<32> offset = currentColumn * NumberOutputUnits + currentClass;
			mem = fw_bw_mem[offset];

			sum = mem + input;

			//output = *reinterpret_cast<ap_uint<OutputActivationOutputLayerWidth> *>(&sum);

			output_stream.write(sum);

		}
	}
}
//===================================================================================================================================================================================
// MAX PER COLUMN
//===================================================================================================================================================================================
template
<
typename Input_t,
unsigned int InputWidth,
typename Max_t,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t
>
void MaxPerColumn(NumberColumns_t numberOfColumns,
				  hls::stream<Input_t> & input_stream,//hls::stream<ap_uint<InputWidth>> & input_stream,
				  hls::stream<Max_t> & output_stream)
{
	Input_t input;
	//ap_uint<InputWidth> input_temp;
	Max_t max;
	max.value = 0.0;
	max.label = 0;

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for(NumberOutputUnits_t currentClass = 0; currentClass < NumberOutputUnits; currentClass++)
		{
		#pragma HLS PIPELINE II=1 rewind

		#pragma HLS loop_flatten off

			input_stream.read(input);

			//input = *reinterpret_cast<Input_t *>(&input_temp);

			if(input > max.value)
			{
				max.value = input;
				max.label = currentClass;
			}
		}

		output_stream.write(max);
		max.value = 0.0;

	}
}
//===================================================================================================================================================================================
// FINAL LABELING
//===================================================================================================================================================================================
template
<
typename Max_t,
typename NumberOutputUnits_t,
unsigned int NumberOutputUnits,
typename NumberColumns_t,
unsigned int MaxNumberColumns
>
void FinalLabeling(NumberColumns_t numberOfColumns,
			   	   hls::stream<Max_t> & input_stream,
				   hls::stream<NumberOutputUnits_t > & output_stream)
{

	static Max_t max_global[MaxNumberColumns];

	Max_t max_per_previous_column;
	max_per_previous_column.value = 0.0;
	max_per_previous_column.label = 0;

	NumberColumns_t center = (numberOfColumns >> 1);
	ap_int<16> pointer = 0;

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
	#pragma HLS PIPELINE II=1 rewind

		Max_t max_per_current_column;
		input_stream.read(max_per_current_column);

		if(max_per_current_column.label == 0)
			max_per_current_column.value = 0.0;

		max_global[center + pointer] = max_per_current_column;

		//if(currentColumn % 2 == 0)
		if(currentColumn(0,0) == 0)
		{
			pointer++;
			pointer = -pointer;
		}
		else
		{
			pointer = -pointer;
		}
	}

	for(NumberColumns_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
	#pragma HLS PIPELINE II=1 rewind

		Max_t max_per_current_column = max_global[currentColumn];

		if(max_per_current_column.value > max_per_previous_column.value)
		{
			max_per_previous_column = max_per_current_column;
		}
		else if(max_per_previous_column.label != 0 && max_per_current_column.label == 0)
		{
			output_stream.write(max_per_previous_column.label);
			max_per_previous_column = max_per_current_column;
		}
	}

	output_stream.write((NumberOutputUnits_t)NumberOutputUnits);
}
//===================================================================================================================================================================================
// DMA OUTPUT
//===================================================================================================================================================================================
template
<
unsigned int NumberOutputUnits,
unsigned int DataWidth,
unsigned int MaxSizeOutputString,
typename NumberOutputUnits_t
>
void Stream2Mem(hls::stream<NumberOutputUnits_t > & input_stream,
				ap_uint<DataWidth> * output_mem)
{

	NumberOutputUnits_t input = 0;
	NumberOutputUnits_t counter = 0;

	while(input != NumberOutputUnits)
	{
	#pragma HLS PIPELINE II=1

		input_stream.read(input);
		counter++;
		if(counter > MaxSizeOutputString - 1)
		{
			break;
		}
		else
		{
			output_mem[counter] = (ap_uint<DataWidth>)input;
		}
	}
	output_mem[0] = (ap_uint<DataWidth>)(counter - 1);
}
//===================================================================================================================================================================================
// DOT PRODUCT FUNCTIONS
//===================================================================================================================================================================================
template
<
unsigned int PE,
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,
unsigned int PixelWidth,
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename Bias_t,
unsigned int BiasWidth,
typename Weight_t,
unsigned int WeightWidth,
typename DotProductResult_t,
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits
>
void DotVectorToMatrix(//const Bias_t biases[PE][NumberHiddenUnits/PE],

				     const Weight_t weights_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][NumberHiddenUnits/PE],
				     const Weight_t weights_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],
				     ap_uint<ColumnHeight * PixelWidth> image_column,
				     ap_uint<OutputActivationWidth * NumberHiddenUnits> inputs,
				     NumberHiddenUnits_t currentHiddenUnit,
				     NumberHiddenUnits_t PE_count,
					 DotProductResult_t &sum_pix1,
					 DotProductResult_t &sum_neuron1)
{
	const unsigned int FoldingInput = ColumnHeight / SIMD_INPUT;
	const unsigned int FoldingRecurrent = NumberHiddenUnits / SIMD_RECURRENT;
	DotProductResult_t mul_pix[SIMD_INPUT][FoldingInput];
	DotProductResult_t mul_neuron[SIMD_RECURRENT][FoldingRecurrent];
	DotProductResult_t sum_pix = 0.0;
	DotProductResult_t sum_neuron = 0.0;
	DotProductResult_t sum = 0.0;
	//ap_int<BiasWidth> bias_temp = biases[PE_count][currentHiddenUnit];
	//Bias_t bias = biases[PE_count][currentHiddenUnit];
	//Bias_t bias = *reinterpret_cast<Bias_t *>(&bias_temp);

#pragma HLS ARRAY_PARTITION variable=mul_pix complete dim=1
#pragma HLS ARRAY_PARTITION variable=mul_neuron complete dim=1
#pragma HLS ARRAY_RESHAPE variable=weights_i complete dim=1
#pragma HLS ARRAY_RESHAPE variable=weights_h complete dim=1
#pragma HLS ARRAY_RESHAPE variable=weights_i complete dim=3
#pragma HLS ARRAY_RESHAPE variable=weights_h complete dim=3
	for(ColumnHeight_t j = 0; j < FoldingInput; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(ColumnHeight_t i = 0; i < SIMD_INPUT; i++)
		{
	#pragma HLS UNROLL
			unsigned int PixelInColumn = i*FoldingInput+j;
			ap_uint<PixelWidth> pixel_temp = image_column((PixelInColumn+1)*PixelWidth-1, PixelInColumn*PixelWidth);
			Pixel_t pixel = *reinterpret_cast<Pixel_t *>(&pixel_temp);


			Weight_t weigth = weights_i[i][j][PE_count][currentHiddenUnit];
			/* if(currentHiddenUnit==0&&PE_count==0){
			         //std::cout<<"FoldingInput:"<<j<<"SIMD_INPUT"<<i<<
			            }*/
			mul_pix[i][j] = pixel * weigth;
		}
	}
	for(NumberHiddenUnits_t j = 0; j < FoldingRecurrent; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(NumberHiddenUnits_t  i = 0; i < SIMD_RECURRENT; i++)
		{
	#pragma HLS UNROLL
			unsigned int ActivationInRecurrent = i*FoldingRecurrent+j;
			ap_int<OutputActivationWidth> input_temp = inputs((ActivationInRecurrent + 1) * OutputActivationWidth - 1, ActivationInRecurrent * OutputActivationWidth);
			OutputActivation_t input = *reinterpret_cast<OutputActivation_t *>(&input_temp);
			//std::cout<<"h_prev"<<i*FoldingRecurrent+j<<":"<<input<<",";

			Weight_t weigth = weights_h[i][j][PE_count][currentHiddenUnit];

			mul_neuron[i][j] = input * weigth;
		}
	}
	//std::cout<<std::endl;
	for(ColumnHeight_t j = 0; j < FoldingInput; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(ColumnHeight_t  i = 0; i < SIMD_INPUT; i++)
		{
	#pragma HLS UNROLL
			sum_pix += mul_pix[i][j];
		}
	}
	for(NumberHiddenUnits_t j = 0; j < FoldingRecurrent; j++) {
#pragma HLS PIPELINE II=1 rewind
		for(NumberHiddenUnits_t  i = 0; i < SIMD_RECURRENT; i++)
		{
	#pragma HLS UNROLL
			sum_neuron += mul_neuron[i][j];
		}
	}

	sum_pix1=sum_pix;
	sum_neuron1=sum_neuron;

	//sum = sum_pix + sum_neuron;//(DotProductResult_t)bias  + sum_pix + sum_neuron;

	//return sum;
}
//===================================================================================================================================================================================
// NEURON
//===================================================================================================================================================================================
template
<
unsigned int PE,					// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Bias_t_gi,					// Type of the bias for gate i
unsigned int BiasWidth_gi,			// number of bits of each bias (gate i)
typename Weight_t_gi,				// Type of the weights for gate i
unsigned int WeightWidth_gi,		// number of bits of each weight (gate i)
typename DotProductResult_t_gi, 	// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename Bias_t_gf,					// Type of the bias for gate f
unsigned int BiasWidth_gf,			// number of bits of each bias (gate f)
typename Weight_t_gf,				// Type of the weights for gate f
unsigned int WeightWidth_gf,		// number of bits of each weight (gate f)
typename DotProductResult_t_gf, 	// type of the result for MAC with weight of gate f
typename gfx_accumulator_t,
typename Bias_t_go,					// Type of the bias for gate o
unsigned int BiasWidth_go,			// number of bits of each bias (gate o)
typename Weight_t_go,				// Type of the weights for gate o
unsigned int WeightWidth_go,		// number of bits of each weight (gate o)
typename DotProductResult_t_go, 	// type of the result for MAC with weight of gate o
typename gox_accumulator_t,
typename Bias_t_ci,					// Type of the bias for gate ci
unsigned int BiasWidth_ci,			// number of bits of each bias (gate ci)
typename Weight_t_ci,				// Type of the weights for gate ci
unsigned int WeightWidth_ci,		// number of bits of each weight (gate ci)
typename DotProductResult_t_ci, 	// type of the result for MAC with weight of gate ci
typename gi_ci_accumulator_t,
typename Weight_wip_t,
unsigned int WeightWidth_wip,
typename Wip_mul_t,
typename Weight_wfp_t,
unsigned int WeightWidth_wfp,
typename Wfp_mul_t,
typename Weight_wop_t,
unsigned int WeightWidth_wop,
typename Wop_mul_t,
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
typename State_t,
typename Sigmoid_out_t,
unsigned int Lut_Entries_Sigmoid,
typename Sigmoid_limit_t,
typename Sigmoid_step_t,
typename Tanh_out_t,
unsigned int Lut_Entries_Tanh,
typename Tanh_limit_t,
typename Tanh_step_t
>
void LSTMCell(uint16_t currentColumn,
			NumberHiddenUnits_t currentHiddenUnit,
			NumberHiddenUnits_t PE_count,
			ap_uint<ColumnHeight  * PixelWidth> image,
			ap_uint<OutputActivationWidth * NumberHiddenUnits> h_prev,
			State_t c_prev,
			State_t & c_next,
			OutputActivation_t & h_next,
			const ap_uint<BiasWidth_gi> biases_gii[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_gi> biases_gih[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gi> weights_gi_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gi> weights_gi_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_gf> biases_gfi[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_gf> biases_gfh[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gf> weights_gf_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_gf> weights_gf_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_go> biases_goi[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_go> biases_goh[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_go> weights_go_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_go> weights_go_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_ci> biases_cii[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<BiasWidth_ci> biases_cih[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_ci> weights_ci_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_ci> weights_ci_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid],
			Tanh_out_t lut_tanh_1[Lut_Entries_Tanh],
			const ap_uint<WeightWidth_wip> peepholes_ip[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_wfp> peepholes_fp[PE][2 * NumberHiddenUnits/PE],
			const ap_uint<WeightWidth_wop> peepholes_op[PE][2 * NumberHiddenUnits/PE]
			)

{
	Wip_mul_t wip_state_mul;
	Wfp_mul_t wfp_state_mul;
	Wop_mul_t wop_state_mul;
	gi_ci_accumulator_t ci_gi_mul;
	gix_accumulator_t gix_sum;
	gfx_accumulator_t gfx_sum;
	gox_accumulator_t gox_sum;

	Sigmoid_out_t gi;
	Sigmoid_out_t gf;
	Sigmoid_out_t go;
	Tanh_out_t ci;
	Tanh_out_t tanhf_o;

	State_t tmp_c_prev, tmp_c_next;
	State_t gf_state_mul;

	DotProductResult_t_gi gix;
	DotProductResult_t_gf gfx;
	DotProductResult_t_go gox;
	DotProductResult_t_ci cix;

	gix = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_gi, BiasWidth_gi, Weight_t_gi, WeightWidth_gi, DotProductResult_t_gi, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_gii, biases_gih, weights_gi_i, weights_gi_h, image, h_prev, currentHiddenUnit, PE_count);
	gfx = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_gf, BiasWidth_gf, Weight_t_gf, WeightWidth_gf, DotProductResult_t_gf, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_gfi, biases_gfh, weights_gf_i, weights_gf_h, image, h_prev, currentHiddenUnit, PE_count);
	gox = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_go, BiasWidth_go, Weight_t_go, WeightWidth_go, DotProductResult_t_go, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_goi, biases_goh, weights_go_i, weights_go_h, image, h_prev, currentHiddenUnit, PE_count);
	cix = DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_ci, BiasWidth_ci, Weight_t_ci, WeightWidth_ci, DotProductResult_t_ci, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(biases_cii, biases_cih, weights_ci_i, weights_ci_h, image, h_prev, currentHiddenUnit, PE_count);

	tmp_c_prev = c_prev;

	if(currentColumn > 0)
	{
		ap_uint<WeightWidth_wip> wip_temp = peepholes_ip[PE_count][currentHiddenUnit];
		Weight_wip_t wip_internal = *reinterpret_cast<Weight_wip_t *>(&wip_temp);
		wip_state_mul = wip_internal * tmp_c_prev;
		gix_sum = gix + wip_state_mul;
	}
	else{
		gix_sum = gix;
	}
	if(currentColumn > 0) {

		ap_uint<WeightWidth_wfp> wfp_temp = peepholes_fp[PE_count][currentHiddenUnit];
		Weight_wfp_t wfp_internal = *reinterpret_cast<Weight_wfp_t *>(&wfp_temp);
		wfp_state_mul = wfp_internal * tmp_c_prev;
		gfx_sum = gfx + wfp_state_mul;
	}
	else{
		gfx_sum = gfx;
	}

	gi = sigmoid_lut<Lut_Entries_Sigmoid,gix_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gix_sum, lut_sigmoid_1);
	gf = sigmoid_lut<Lut_Entries_Sigmoid,gfx_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gfx_sum, lut_sigmoid_1);

	ci = tanh_lut<Lut_Entries_Tanh,DotProductResult_t_ci,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(cix,lut_tanh_1);
	ci_gi_mul = ci * gi;

	if(currentColumn > 0)
	{

		gf_state_mul = gf * tmp_c_prev;
		tmp_c_next = ci_gi_mul + gf_state_mul;

		ap_uint<WeightWidth_wop> wop_temp = peepholes_op[PE_count][currentHiddenUnit];
		Weight_wop_t wop_internal = *reinterpret_cast<Weight_wop_t *>(&wop_temp);

		wop_state_mul = wop_internal * tmp_c_next;
		gox_sum = gox + wop_state_mul;

	}
	else
	{
		tmp_c_next = ci_gi_mul;
		gox_sum = gox;
	}

	go = sigmoid_lut<Lut_Entries_Sigmoid,gox_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gox_sum, lut_sigmoid_1);

	tanhf_o = tanh_lut<Lut_Entries_Tanh,State_t,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(tmp_c_next,lut_tanh_1);
	h_next = tanhf_o * go;

	c_next = tmp_c_next;
}
//===================================================================================================================================================================================
//NEURON WITHOUT PEEPHOLES
//===================================================================================================================================================================================
template
<
unsigned int PE,					// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Bias_t_gi,					// Type of the bias for gate i
unsigned int BiasWidth_gi,			// number of bits of each bias (gate i)
typename Weight_t_gi,				// Type of the weights for gate i
unsigned int WeightWidth_gi,		// number of bits of each weight (gate i)
typename DotProductResult_t_gi, 	// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename Bias_t_gf,					// Type of the bias for gate f
unsigned int BiasWidth_gf,			// number of bits of each bias (gate f)
typename Weight_t_gf,				// Type of the weights for gate f
unsigned int WeightWidth_gf,		// number of bits of each weight (gate f)
typename DotProductResult_t_gf, 	// type of the result for MAC with weight of gate f
typename gfx_accumulator_t,
typename Bias_t_go,					// Type of the bias for gate o
unsigned int BiasWidth_go,			// number of bits of each bias (gate o)
typename Weight_t_go,				// Type of the weights for gate o
unsigned int WeightWidth_go,		// number of bits of each weight (gate o)
typename DotProductResult_t_go, 	// type of the result for MAC with weight of gate o
typename gox_accumulator_t,
typename Bias_t_ci,					// Type of the bias for gate ci
unsigned int BiasWidth_ci,			// number of bits of each bias (gate ci)
typename Weight_t_ci,				// Type of the weights for gate ci
unsigned int WeightWidth_ci,		// number of bits of each weight (gate ci)
typename DotProductResult_t_ci, 	// type of the result for MAC with weight of gate ci
typename gi_ci_accumulator_t,
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
typename State_t,
typename Sigmoid_out_t,
unsigned int Lut_Entries_Sigmoid,
typename Sigmoid_limit_t,
typename Sigmoid_step_t,
typename Tanh_out_t,
unsigned int Lut_Entries_Tanh,
typename Tanh_limit_t,
typename Tanh_step_t
>
void LSTMCell_noPH(uint16_t currentColumn,
		NumberHiddenUnits_t currentHiddenUnit,
					NumberHiddenUnits_t PE_count,
					ap_uint<ColumnHeight  * PixelWidth> image,
					ap_uint<OutputActivationWidth * NumberHiddenUnits> h_prev,
					State_t c_prev,
					//State_t & c_next,
					//OutputActivation_t & h_next,
					     const Bias_t_gi biases_gi[PE][NumberHiddenUnits/PE], //
const Weight_t_gi weights_gi_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][NumberHiddenUnits/PE],//
								  const Weight_t_gi weights_gi_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],//
								  const Bias_t_gf biases_gf[PE][NumberHiddenUnits/PE],

								  const Weight_t_gf weights_gf_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][NumberHiddenUnits/PE],
								  const Weight_t_gf weights_gf_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],
								  const Bias_t_go biases_go[PE][NumberHiddenUnits/PE],

								  const Weight_t_go weights_go_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][NumberHiddenUnits/PE],
								  const Weight_t_go weights_go_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],
								  const Bias_t_ci biases_ci[PE][NumberHiddenUnits/PE],

								  const Weight_t_ci weights_ci_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][NumberHiddenUnits/PE],
								  const Weight_t_ci weights_ci_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],
					Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid],
					Tanh_out_t lut_tanh_1[Lut_Entries_Tanh],
										  DotProductResult_t_gi &gix_i1, DotProductResult_t_gi &gix_h1,
										  	DotProductResult_t_gf &gfx_i1,DotProductResult_t_gf &gfx_h1,
										  	DotProductResult_t_go &gox_i1,DotProductResult_t_go &gox_h1,
										  	DotProductResult_t_ci &cix_i1,DotProductResult_t_ci &cix_h1

)

{

	gi_ci_accumulator_t ci_gi_mul;
	gix_accumulator_t gix_sum;
	gfx_accumulator_t gfx_sum;
	gox_accumulator_t gox_sum;

	Sigmoid_out_t gi;
	Sigmoid_out_t gf;
	Sigmoid_out_t go;
	Tanh_out_t ci;
	Tanh_out_t tanhf_o;

	State_t tmp_c_next;
	State_t gf_state_mul;

	DotProductResult_t_gi gix_i,gix_h;
	DotProductResult_t_gf gfx_i,gfx_h;
	DotProductResult_t_go gox_i,gox_h;
	DotProductResult_t_ci cix_i,cix_h;



//change
	for(int i=0;i<4;i++){}
	 DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_gi, BiasWidth_gi, Weight_t_gi, WeightWidth_gi, DotProductResult_t_gi, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(weights_gi_i, weights_gi_h, image, h_prev, currentHiddenUnit, PE_count,gix_i,gix_h);
	 DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_gf, BiasWidth_gf, Weight_t_gf, WeightWidth_gf, DotProductResult_t_gf, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(weights_gf_i, weights_gf_h, image, h_prev, currentHiddenUnit, PE_count,gfx_i,gfx_h);
	 DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_go, BiasWidth_go, Weight_t_go, WeightWidth_go, DotProductResult_t_go, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(weights_go_i, weights_go_h, image, h_prev, currentHiddenUnit, PE_count,gox_i,gox_h);
	 DotVectorToMatrix<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth, OutputActivation_t, OutputActivationWidth, Bias_t_ci, BiasWidth_ci, Weight_t_ci, WeightWidth_ci, DotProductResult_t_ci, ColumnHeight_t, ColumnHeight, NumberHiddenUnits_t, NumberHiddenUnits>(weights_ci_i, weights_ci_h, image, h_prev, currentHiddenUnit, PE_count,cix_i,cix_h);

	 gix_i1=gix_i;gix_h1=gix_h;
	 gfx_i1=gfx_i;gfx_h1=gfx_h;
	 gox_i1=gox_i;gox_h1=gox_h;
	 cix_i1=cix_i;cix_h1=cix_h;


}
//===================================================================================================================================================================================
// HIDDEN LAYER
//===================================================================================================================================================================================
template<
unsigned int PE,				// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 			// Number of parallel MAC performed in the gates on recurrent path
typename Pixel_t,     				// Type of the input pixel
unsigned int PixelWidth, 			// number of bits of the input pixel
typename Bias_t_gi,				// Type of the bias for gate i
unsigned int BiasWidth_gi,			// number of bits of each bias (gate i)
typename Weight_t_gi,				// Type of the weights for gate i
unsigned int WeightWidth_gi,			// number of bits of each weight (gate i)
typename DotProductResult_t_gi, 		// type of the result for MAC with weight of gate i
typename gix_accumulator_t,
typename Bias_t_gf,				// Type of the bias for gate f
unsigned int BiasWidth_gf,			// number of bits of each bias (gate f)
typename Weight_t_gf,				// Type of the weights for gate f
unsigned int WeightWidth_gf,			// number of bits of each weight (gate f)
typename DotProductResult_t_gf, 		// type of the result for MAC with weight of gate f
typename gfx_accumulator_t,
typename Bias_t_go,				// Type of the bias for gate o
unsigned int BiasWidth_go,			// number of bits of each bias (gate o)
typename Weight_t_go,				// Type of the weights for gate o
unsigned int WeightWidth_go,			// number of bits of each weight (gate o)
typename DotProductResult_t_go, 		// type of the result for MAC with weight of gate o
typename gox_accumulator_t,
typename Bias_t_ci,				// Type of the bias for gate ci
unsigned int BiasWidth_ci,			// number of bits of each bias (gate ci)
typename Weight_t_ci,				// Type of the weights for gate ci
unsigned int WeightWidth_ci,			// number of bits of each weight (gate ci)
typename DotProductResult_t_ci, 		// type of the result for MAC with weight of gate ci
typename gi_ci_accumulator_t,
typename Weight_wip_t,
unsigned int WeightWidth_wip,
typename Wip_mul_t,
typename Weight_wfp_t,
unsigned int WeightWidth_wfp,
typename Wfp_mul_t,
typename Weight_wop_t,
unsigned int WeightWidth_wop,
typename Wop_mul_t,
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename ColumnHeight_t,
unsigned int ColumnHeight,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
unsigned int MaxNumberColumns,
typename State_t,
typename Sigmoid_out_t,
unsigned int Lut_Entries_Sigmoid,
typename Sigmoid_limit_t,
typename Sigmoid_step_t,
typename Tanh_out_t,
unsigned int Lut_Entries_Tanh,
typename Tanh_limit_t,
typename Tanh_step_t
>
void HiddenLayer(uint32_t numberOfColumns,
			  hls::stream<ap_uint<ColumnHeight * PixelWidth> > &image_stream,
			  hls::stream<ap_uint<OutputActivationWidth> > &result_stream,
			  const ap_uint<BiasWidth_gi> biases_gii[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_gi> biases_gih[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gi> weights_gi_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gi> weights_gi_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_gf> biases_gfi[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_gf> biases_gfh[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gf> weights_gf_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_gf> weights_gf_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_go> biases_goi[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_go> biases_goh[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_go> weights_go_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_go> weights_go_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_ci> biases_cii[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<BiasWidth_ci> biases_cih[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_ci> weights_ci_i[SIMD_INPUT][ColumnHeight/SIMD_INPUT][PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_ci> weights_ci_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][2 * NumberHiddenUnits/PE],
			  Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid],
			  Tanh_out_t lut_tanh_1[Lut_Entries_Tanh],
			  const ap_uint<WeightWidth_wip> peepholes_ip[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_wfp> peepholes_fp[PE][2 * NumberHiddenUnits/PE],
			  const ap_uint<WeightWidth_wop> peepholes_op[PE][2 * NumberHiddenUnits/PE]
			)
{
	if(NumberHiddenUnits % PE != 0) {
		std::cout << "Error: NumberHiddenUnits has to be a multiple of PE" << std::endl;
	}
	if(ColumnHeight % SIMD_INPUT != 0) {
		std::cout << "Error: ColumnHeight size has to be multiple of SIMD_INPUT" << std::endl;
	}
	if(NumberHiddenUnits % SIMD_RECURRENT != 0) {
		std::cout << "Error: NumberHiddenUnits size has to be multiple of SIMD_RECURRENT" << std::endl;
	}

	State_t c_prev, c_next, c_reg[2 * NumberHiddenUnits/PE][PE];
#pragma HLS ARRAY_PARTITION variable=c_reg complete dim=2
	OutputActivation_t output;

	ap_uint<ColumnHeight * PixelWidth> local_image;
	ap_uint<OutputActivationWidth * NumberHiddenUnits> local_input;

	hls::stream<ap_uint<OutputActivationWidth * NumberHiddenUnits> > recurrent_stream;
#pragma HLS STREAM variable=recurrent_stream depth=4

	ap_uint<OutputActivationWidth * NumberHiddenUnits> output_reg = 0;
	for(NumberHiddenUnits_t path = 0; path < 2; path++)
	{
#pragma HLS PIPELINE II=1 rewind
		recurrent_stream.write(output_reg);
	}

	for(uint16_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		for (ap_uint<2> count = 0; count <2; count ++)
		{
			image_stream.read(local_image);
			recurrent_stream.read(local_input);
			for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits/PE; currentHiddenUnit++)
			{
			const unsigned int FoldingInput = ColumnHeight / SIMD_INPUT;
			#pragma HLS PIPELINE II=FoldingInput rewind
			ap_uint<OutputActivationWidth*PE> temp_output_packed;
				for (NumberHiddenUnits_t PE_count = 0; PE_count < PE; PE_count++)
				{
				#pragma HLS UNROLL
					NumberHiddenUnits_t actual_hidden_unit_address = count*NumberHiddenUnits/PE + currentHiddenUnit;
					c_prev = c_reg[actual_hidden_unit_address][PE_count];

					LSTMCell
					<PE, SIMD_INPUT, SIMD_RECURRENT, Pixel_t, PixelWidth,
					Bias_t_gi, BiasWidth_gi, Weight_t_gi, WeightWidth_gi, DotProductResult_t_gi, gix_accumulator_t,
					Bias_t_gf, BiasWidth_gf, Weight_t_gf, WeightWidth_gf, DotProductResult_t_gf, gfx_accumulator_t,
					Bias_t_go, BiasWidth_go, Weight_t_go, WeightWidth_go, DotProductResult_t_go, gox_accumulator_t,
					Bias_t_ci, BiasWidth_ci, Weight_t_ci, WeightWidth_ci, DotProductResult_t_ci, gi_ci_accumulator_t,
					Weight_wip_t, WeightWidth_wip, Wip_mul_t,
					Weight_wfp_t, WeightWidth_wfp, Wfp_mul_t,
					Weight_wop_t, WeightWidth_wop, Wop_mul_t,
					OutputActivation_t, OutputActivationWidth,
					ColumnHeight_t, ColumnHeight,
					NumberHiddenUnits_t, NumberHiddenUnits,
					State_t,
					Sigmoid_out_t, Lut_Entries_Sigmoid, Sigmoid_limit_t, Sigmoid_step_t,
					Tanh_out_t, Lut_Entries_Tanh, Tanh_limit_t, Tanh_step_t>
					(currentColumn,
					actual_hidden_unit_address, PE_count,
					local_image,
					local_input,
					c_prev,
					c_next,
					output,
					biases_gii, biases_gih, weights_gi_i, weights_gi_h,
					biases_gfi, biases_gfh, weights_gf_i, weights_gf_h,
					biases_goi, biases_goh, weights_go_i, weights_go_h,
					biases_cii, biases_cih, weights_ci_i, weights_ci_h,
					lut_sigmoid_1,lut_tanh_1,
					peepholes_ip,peepholes_fp,peepholes_op);

					c_reg[actual_hidden_unit_address][PE_count] = c_next;

					temp_output_packed = temp_output_packed >> OutputActivationWidth;
					ap_uint<OutputActivationWidth> temp_output = *reinterpret_cast<ap_uint<OutputActivationWidth> *>(&output);
					temp_output_packed(OutputActivationWidth*PE-1,OutputActivationWidth*(PE-1)) = temp_output;
					output_reg(((currentHiddenUnit*PE + PE_count) + 1) * OutputActivationWidth - 1, (currentHiddenUnit*PE + PE_count) * OutputActivationWidth) = temp_output;
				} // pe
				result_stream.write(temp_output_packed);
			}//neurons
			if(currentColumn < numberOfColumns - 1)
				recurrent_stream.write(output_reg);
		}// backward/forward
	}//column
}


//===================================================================================================================================================================================
// HIDDEN LAYER
//===================================================================================================================================================================================
template<
unsigned int PE,					// Number of neurons to be executed in parallel
unsigned int SIMD_INPUT, 			// Number of parallel MAC performed in the gates on input pixels
unsigned int SIMD_RECURRENT, 		// Number of parallel MAC performed in the gates on recurrent path
typename Vector_t,     				// 向量的类�?
unsigned int VectorWidth, 			// 向量的位�?
typename Bias_t_gi,					// Type of the bias for//     gate i的bias的数据类�?
unsigned int BiasWidth_gi,			// number of bits of each //  bias (gate i) 数据类型对应的bit�?
typename Weight_t_gi,				// Type of the weights for//   gate i的weight的数据类�?
unsigned int WeightWidth_gi,		// number of bits of each //  weight (gate i) 数据类型对应的bit�?
typename DotProductResult_t_gi, 	// type of the result for MAC with weight of gate i //
typename gix_accumulator_t,                                    //
typename Bias_t_gf,					// Type of the bias for gate f
unsigned int BiasWidth_gf,			// number of bits of each bias (gate f)
typename Weight_t_gf,				// Type of the weights for gate f
unsigned int WeightWidth_gf,		// number of bits of each weight (gate f)
typename DotProductResult_t_gf, 	// type of the result for MAC with weight of gate f
typename gfx_accumulator_t,
typename Bias_t_go,					// Type of the bias for gate o
unsigned int BiasWidth_go,			// number of bits of each bias (gate o)
typename Weight_t_go,				// Type of the weights for gate o
unsigned int WeightWidth_go,		// number of bits of each weight (gate o)
typename DotProductResult_t_go, 	// type of the result for MAC with weight of gate o
typename gox_accumulator_t,
typename Bias_t_ci,					// Type of the bias for gate ci
unsigned int BiasWidth_ci,			// number of bits of each bias (gate ci)
typename Weight_t_ci,				// Type of the weights for gate ci
unsigned int WeightWidth_ci,		// number of bits of each weight (gate ci)
typename DotProductResult_t_ci, 	// type of the result for MAC with weight of gate ci
typename gi_ci_accumulator_t,
typename OutputActivation_t,
unsigned int OutputActivationWidth,
typename VectorSize_t,
unsigned int VectorSize,
typename NumberHiddenUnits_t,
unsigned int NumberHiddenUnits,
unsigned int MaxNumberColumns,
typename State_t,
typename Sigmoid_out_t,              // 查找表内数据的类�? 即查找表的输出类�?
unsigned int Lut_Entries_Sigmoid,   //SIGMOID 查找表的长度
typename Sigmoid_limit_t,          //查找表的limit的数据类�?
typename Sigmoid_step_t,            //查找表的recip_step的数据类�?
typename Tanh_out_t,               //
unsigned int Lut_Entries_Tanh,    //Tanh 查找表的长度
typename Tanh_limit_t,            //
typename Tanh_step_t,
typename t_norm_weights,
typename t_norm_bias//
>
void HiddenLayer_noPH(uint32_t numberOfColumns,
			  hls::stream<ap_uint<VectorSize * VectorWidth> > &vector_stream,
			  hls::stream<ap_uint<OutputActivationWidth*NumberHiddenUnits/PE> > &result_stream,

			  const Bias_t_gi biases_gi[PE][NumberHiddenUnits/PE], //
			  const t_norm_weights norm_w_gi_i[NumberHiddenUnits],
			  const t_norm_bias norm_b_gi_i[NumberHiddenUnits],
			  const Weight_t_gi weights_gi_i[SIMD_INPUT][VectorSize/SIMD_INPUT][PE][NumberHiddenUnits/PE],//
			  const t_norm_weights norm_w_gi_h[NumberHiddenUnits],
			  const t_norm_bias norm_b_gi_h[NumberHiddenUnits],
			  const Weight_t_gi weights_gi_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],//

			  const Bias_t_gf biases_gf[PE][NumberHiddenUnits/PE],
			  const t_norm_weights norm_w_gf_i[NumberHiddenUnits],
			  const t_norm_bias norm_b_gf_i[NumberHiddenUnits],
			  const Weight_t_gf weights_gf_i[SIMD_INPUT][VectorSize/SIMD_INPUT][PE][NumberHiddenUnits/PE],
			  const t_norm_weights norm_w_gf_h[NumberHiddenUnits],
			  const t_norm_bias norm_b_gf_h[NumberHiddenUnits],
			  const Weight_t_gf weights_gf_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],

			  const Bias_t_go biases_go[PE][NumberHiddenUnits/PE],
			  const t_norm_weights norm_w_go_i[NumberHiddenUnits],
			  const t_norm_bias norm_b_go_i[NumberHiddenUnits],
			  const Weight_t_go weights_go_i[SIMD_INPUT][VectorSize/SIMD_INPUT][PE][NumberHiddenUnits/PE],
			  const t_norm_weights norm_w_go_h[NumberHiddenUnits],
			  const t_norm_bias norm_b_go_h[NumberHiddenUnits],
			  const Weight_t_go weights_go_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],

			  const Bias_t_ci biases_ci[PE][NumberHiddenUnits/PE],
			  const t_norm_weights norm_w_ci_i[NumberHiddenUnits],
			  const t_norm_bias norm_b_ci_i[NumberHiddenUnits],
			  const Weight_t_ci weights_ci_i[SIMD_INPUT][VectorSize/SIMD_INPUT][PE][NumberHiddenUnits/PE],
			  const t_norm_weights norm_w_ci_h[NumberHiddenUnits],
			  const t_norm_bias norm_b_ci_h[NumberHiddenUnits],
			  const Weight_t_ci weights_ci_h[SIMD_RECURRENT][NumberHiddenUnits/SIMD_RECURRENT][PE][NumberHiddenUnits/PE],

			  Sigmoid_out_t lut_sigmoid_1[Lut_Entries_Sigmoid],
			  Tanh_out_t lut_tanh_1[Lut_Entries_Tanh]

			)
{
	if(NumberHiddenUnits % PE != 0) {
		std::cout << "Error: NumberHiddenUnits has to be a multiple of PE" << std::endl;
	}
	if(VectorSize % SIMD_INPUT != 0) {
		std::cout << "Error: ColumnHeight size has to be multiple of SIMD_INPUT" << std::endl;
	}
	if(NumberHiddenUnits % SIMD_RECURRENT != 0) {
		std::cout << "Error: NumberHiddenUnits size has to be multiple of SIMD_RECURRENT" << std::endl;
	}


	State_t c_prev, c_next, c_reg[NumberHiddenUnits/PE][PE];
	//初始c�?
	for(int i=0;i<NumberHiddenUnits/PE;i++){for(int j=0;j<PE;j++)
	{
		c_reg[i][j]=0;

	}
	}

#pragma HLS ARRAY_PARTITION variable=c_reg complete dim=2
	OutputActivation_t output;


	ap_uint<VectorSize * VectorWidth> local_vector;
	ap_uint<OutputActivationWidth * NumberHiddenUnits> local_input;

	hls::stream<ap_uint<OutputActivationWidth * NumberHiddenUnits> > recurrent_stream;
#pragma HLS STREAM variable=recurrent_stream depth=4

	ap_uint<OutputActivationWidth * NumberHiddenUnits> output_reg = 0;

	recurrent_stream.write(output_reg);  //初始h �?0
    typedef ap_fixed<32 ,18, AP_RND_ZERO, AP_WRAP> t_avg;
    t_avg gix_i_avg,gix_h_avg,gfx_i_avg,gfx_h_avg,gox_i_avg,gox_h_avg,cix_i_avg,cix_h_avg=0;
    t_avg gix_i_var,gix_h_var,gfx_i_var,gfx_h_var,gox_i_var,gox_h_var,cix_i_var,cix_h_var=0;
    DotProductResult_t_gi gix_i[NumberHiddenUnits/PE][PE],gix_h[NumberHiddenUnits/PE][PE];
	DotProductResult_t_gf gfx_i[NumberHiddenUnits/PE][PE],gfx_h[NumberHiddenUnits/PE][PE];
	DotProductResult_t_go gox_i[NumberHiddenUnits/PE][PE],gox_h[NumberHiddenUnits/PE][PE];
	DotProductResult_t_ci cix_i[NumberHiddenUnits/PE][PE],cix_h[NumberHiddenUnits/PE][PE];
	for(uint16_t currentColumn = 0; currentColumn < numberOfColumns; currentColumn++)
	{
		    vector_stream.read(local_vector);
			recurrent_stream.read(local_input);
			std::cout<<"input"<<currentColumn<<":"<<local_vector<<std::endl;
			for(NumberHiddenUnits_t currentHiddenUnit = 0; currentHiddenUnit < NumberHiddenUnits/PE; currentHiddenUnit++)
			{
			const unsigned int FoldingInput = VectorSize / SIMD_INPUT;
			#pragma HLS PIPELINE II=FoldingInput rewind
				ap_uint<OutputActivationWidth*PE> temp_output_packed;
				for (NumberHiddenUnits_t PE_count = 0; PE_count < PE; PE_count++)
				{
				#pragma HLS UNROLL
					NumberHiddenUnits_t actual_hidden_unit_address = currentHiddenUnit;

					LSTMCell_noPH
					<PE, SIMD_INPUT, SIMD_RECURRENT, Vector_t, VectorWidth,
					Bias_t_gi, BiasWidth_gi, Weight_t_gi, WeightWidth_gi, DotProductResult_t_gi, gix_accumulator_t,
					Bias_t_gf, BiasWidth_gf, Weight_t_gf, WeightWidth_gf, DotProductResult_t_gf, gfx_accumulator_t,
					Bias_t_go, BiasWidth_go, Weight_t_go, WeightWidth_go, DotProductResult_t_go, gox_accumulator_t,
					Bias_t_ci, BiasWidth_ci, Weight_t_ci, WeightWidth_ci, DotProductResult_t_ci, gi_ci_accumulator_t,
					OutputActivation_t, OutputActivationWidth,
					VectorSize_t, VectorSize,
					NumberHiddenUnits_t, NumberHiddenUnits,
					State_t,
					Sigmoid_out_t, Lut_Entries_Sigmoid, Sigmoid_limit_t, Sigmoid_step_t,
					Tanh_out_t, Lut_Entries_Tanh, Tanh_limit_t, Tanh_step_t>
					(currentColumn,
					actual_hidden_unit_address, PE_count,
					local_vector,
					local_input,
					c_prev,
					//c_next,
					//output,
					biases_gi, weights_gi_i, weights_gi_h,
					biases_gf, weights_gf_i, weights_gf_h,
					biases_go, weights_go_i, weights_go_h,
					biases_ci, weights_ci_i, weights_ci_h,
					lut_sigmoid_1,lut_tanh_1,
					gix_i[currentHiddenUnit][PE_count],
					gix_h[currentHiddenUnit][PE_count],
					gfx_i[currentHiddenUnit][PE_count],
					gfx_h[currentHiddenUnit][PE_count],
					gox_i[currentHiddenUnit][PE_count],
					gox_h[currentHiddenUnit][PE_count],
					cix_i[currentHiddenUnit][PE_count],
					cix_h[currentHiddenUnit][PE_count]);
       		} // pe
				//result_stream.write(temp_output_packed);
			}//neurons





			//normalization

			for(NumberHiddenUnits_t a=0;a<NumberHiddenUnits/PE;a++){
				for(NumberHiddenUnits_t b=0;b<PE;b++){
					gix_i_avg+=(t_avg)gix_i[a][b];
					gix_h_avg+=(t_avg)gix_h[a][b];
					gfx_i_avg+=(t_avg)gfx_i[a][b];
					gfx_h_avg+=(t_avg)gfx_h[a][b];
					gox_i_avg+=(t_avg)gox_i[a][b];
					gox_h_avg+=(t_avg)gox_h[a][b];
					cix_i_avg+=(t_avg)cix_i[a][b];
					cix_h_avg+=(t_avg)cix_h[a][b];
					}
				}
			gix_i_avg=gix_i_avg/(t_avg)512;
			gix_h_avg=gix_h_avg/(t_avg)512;
			gfx_i_avg=gfx_i_avg/(t_avg)512;
			gfx_h_avg=gfx_h_avg/(t_avg)512;
			gox_i_avg=gox_i_avg/(t_avg)512;
			gox_h_avg=gox_h_avg/(t_avg)512;
			cix_i_avg=cix_i_avg/(t_avg)512;
			cix_h_avg=cix_h_avg/(t_avg)512;
			//if(currentColumn==1){

							//std::cout<<"avg:"<<"gix_i:"<<gix_i_avg<<","<<"gix_h:"<<gix_h_avg<<","<<"gfx_i:"<<gfx_i_avg<<","<<"gfx_h:"<<gfx_h_avg<<","<<"gox_i:"<<gox_i_avg<<","<<"gox_h:"<<gox_h_avg<<","<<"cix_i:"<<cix_i_avg<<","<<"cix_h:"<<cix_h_avg<<std::endl;

																						//}



			for(NumberHiddenUnits_t a=0;a<NumberHiddenUnits/PE;a++){
							for(NumberHiddenUnits_t b=0;b<PE;b++){
								gix_i_var+=((t_avg)gix_i[a][b]-gix_i_avg)*((t_avg)gix_i[a][b]-gix_i_avg);	//gix_i_var+=(t_avg)pow((t_avg)gix_i[a][b]-gix_i_avg,2);
								gix_h_var+=((t_avg)gix_h[a][b]-gix_h_avg)*((t_avg)gix_h[a][b]-gix_h_avg);//gix_h_var+=(t_avg)pow((t_avg)gix_h[a][b]-gix_h_avg,2);
								gfx_i_var+=((t_avg)gfx_i[a][b]-gfx_i_avg)*((t_avg)gfx_i[a][b]-gfx_i_avg);//gfx_i_var+=(t_avg)pow((t_avg)gfx_i[a][b]-gfx_i_avg,2);
								gfx_h_var+=((t_avg)gfx_h[a][b]-gfx_h_avg)*((t_avg)gfx_h[a][b]-gfx_h_avg);//gfx_h_var+=(t_avg)pow((t_avg)gfx_h[a][b]-gfx_h_avg,2);
								gox_i_var+=((t_avg)gox_i[a][b]-gox_i_avg)*((t_avg)gox_i[a][b]-gox_i_avg);//gox_i_var+=(t_avg)pow((t_avg)gox_i[a][b]-gox_i_avg,2);
								gox_h_var+=((t_avg)gox_h[a][b]-gox_h_avg)*((t_avg)gox_h[a][b]-gox_h_avg);//gox_h_var+=(t_avg)pow((t_avg)gox_h[a][b]-gox_h_avg,2);
								cix_i_var+=((t_avg)cix_i[a][b]-cix_i_avg)*((t_avg)cix_i[a][b]-cix_i_avg);//cix_i_var+=(t_avg)pow((t_avg)cix_i[a][b]-cix_i_avg,2);
								cix_h_var+=((t_avg)cix_h[a][b]-cix_h_avg)*((t_avg)cix_h[a][b]-cix_h_avg);//cix_h_var+=(t_avg)pow((t_avg)cix_h[a][b]-cix_h_avg,2);
								}
							}
			//std::cout<<"gix_i:"<<gix_i_var<<std::endl;

			 gix_i_var=sqrt((ap_fixed<32,18>)(gix_i_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;//gix_i_var/(t_avg)512+(t_avg)1e-2;   //gix_i_var=sqrt(gix_i_var/(t_avg)512);
			 gix_h_var=sqrt((ap_fixed<32,18>)(gix_h_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;		//	gix_h_var=sqrt(gix_h_var/(t_avg)512);
			 gfx_i_var=sqrt((ap_fixed<32,18>)(gfx_i_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;		//	gfx_i_var=sqrt(gfx_i_var/(t_avg)512);
			 gfx_h_var=sqrt((ap_fixed<32,18>)(gfx_h_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;	//	gfx_h_var=sqrt(gfx_h_var/(t_avg)512);
			 gox_i_var=sqrt((ap_fixed<32,18>)(gox_i_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;			//gox_i_var=sqrt(gox_i_var/(t_avg)512);
			 gox_h_var=sqrt((ap_fixed<32,18>)(gox_h_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;			//gox_h_var=sqrt(gox_h_var/(t_avg)512);
			 cix_i_var=sqrt((ap_fixed<32,18>)(cix_i_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;		//cix_i_var=sqrt(cix_i_var/(t_avg)512);
			 cix_h_var=sqrt((ap_fixed<32,18>)(cix_h_var/(t_avg)512));//+(t_avg)1e-5));//+(t_avg)1e-2;  //			cix_h_var=sqrt(cix_h_var/(t_avg)512);
			//
			 //if(currentColumn==1){

			 						 							//std::cout<<"std:"<<"gix_i:"<<gix_i_var<<","<<"gix_h:"<<gix_h_var<<","<<"gfx_i:"<<gfx_i_var<<","<<"gfx_h:"<<gfx_h_var<<","<<"gox_i:"<<gox_i_var<<","<<"gox_h:"<<gox_h_var<<","<<"cix_i:"<<cix_i_var<<","<<"cix_h:"<<cix_h_var<<std::endl;

			 						 																						//}


			// std::cout<<"gix:"<<gix_i[0][0]<<"gix_i_var:"<<gix_i_var<<"gix_i_avg:"<<gix_i_avg<<"w_norm:"<<norm_w_gi_i[0]<<"b_norm:"<<norm_b_gi_i[0]<<std::endl;
						for(NumberHiddenUnits_t a=0;a<NumberHiddenUnits/PE;a++){
						for(NumberHiddenUnits_t b=0;b<PE;b++){
							gix_i[a][b]=(gix_i[a][b]-gix_i_avg)*norm_w_gi_i[a+b*NumberHiddenUnits/PE]/(gix_i_var+(t_avg)1e-4)+norm_b_gi_i[a+b*NumberHiddenUnits/PE];
							gix_h[a][b]=(gix_h[a][b]-gix_h_avg)*norm_w_gi_h[a+b*NumberHiddenUnits/PE]/(gix_h_var+(t_avg)1e-4)+norm_b_gi_h[a+b*NumberHiddenUnits/PE];
						    gfx_i[a][b]=(gfx_i[a][b]-gfx_i_avg)*norm_w_gf_i[a+b*NumberHiddenUnits/PE]/(gfx_i_var+(t_avg)1e-4)+norm_b_gf_i[a+b*NumberHiddenUnits/PE];
							gfx_h[a][b]=(gfx_h[a][b]-gfx_h_avg)*norm_w_gf_h[a+b*NumberHiddenUnits/PE]/(gfx_h_var+(t_avg)1e-4)+norm_b_gf_h[a+b*NumberHiddenUnits/PE];
							gox_i[a][b]=(gox_i[a][b]-gox_i_avg)*norm_w_go_i[a+b*NumberHiddenUnits/PE]/(gox_i_var+(t_avg)1e-4)+norm_b_go_i[a+b*NumberHiddenUnits/PE];
							gox_h[a][b]=(gox_h[a][b]-gox_h_avg)*norm_w_go_h[a+b*NumberHiddenUnits/PE]/(gox_h_var+(t_avg)1e-4)+norm_b_go_h[a+b*NumberHiddenUnits/PE];
							cix_i[a][b]=(cix_i[a][b]-cix_i_avg)*norm_w_ci_i[a+b*NumberHiddenUnits/PE]/(cix_i_var+(t_avg)1e-4)+norm_b_ci_i[a+b*NumberHiddenUnits/PE];
							cix_h[a][b]=(cix_h[a][b]-cix_h_avg)*norm_w_ci_h[a+b*NumberHiddenUnits/PE]/(cix_h_var+(t_avg)1e-4)+norm_b_ci_h[a+b*NumberHiddenUnits/PE];
				}
			}
						/*if(currentColumn==1){
															for(int j=0;j<PE;j++){for(int i=0;i<NumberHiddenUnits/PE;i++)
																			{
																				std::cout<<"gix_i:"<<gix_i[i][j]<<","<<"gix_h:"<<gix_h[i][j]<<","<<"gfx_i:"<<gfx_i[i][j]<<","<<"gfx_h:"<<gfx_h[i][j]<<","<<"gox_i:"<<gox_i[i][j]<<","<<"gox_h:"<<gox_h[i][j]<<","<<"cix_i:"<<cix_i[i][j]<<","<<"cix_h:"<<cix_h[i][j]<<std::endl;

																			}
																			}}*/

						/*if(currentColumn==1){  std::cout<<"gix_h_lnorm:";
																								//for(int j=0;j<PE;j++){for(int i=0;i<NumberHiddenUnits/PE;i++)
																												{
								//std::cout<<gix_h[i][j]<<",";//<<"gix_i:"<<gix_i[i][j]<<","<<"gix_h:"<<gix_h[i][j]<<","<<"gfx_i:"<<gfx_i[i][j]<<","<<"gfx_h:"<<gfx_h[i][j]<<","<<"gox_i:"<<gox_i[i][j]<<","<<"gox_h:"<<gox_h[i][j]<<","<<"cix_i:"<<cix_i[i][j]<<","<<"cix_h:"<<cix_h[i][j]<<std::endl;

																												}
																												}*/
									//std::cout<<std::endl;}
						    gi_ci_accumulator_t ci_gi_mul;
							gix_accumulator_t gix_sum;
							gfx_accumulator_t gfx_sum;
							gox_accumulator_t gox_sum;

							Sigmoid_out_t gi;
							Sigmoid_out_t gf;
							Sigmoid_out_t go;
							Tanh_out_t ci;
							Tanh_out_t tanhf_o;

							State_t tmp_c_next;
							State_t gf_state_mul;
							DotProductResult_t_ci cix;
							for(NumberHiddenUnits_t PE_count=0;PE_count<PE;PE_count++)
							{
								ap_uint<OutputActivationWidth*NumberHiddenUnits/PE> temp_output_packed;
								for(NumberHiddenUnits_t currentHiddenUnit=0;currentHiddenUnit<NumberHiddenUnits/PE;currentHiddenUnit++){
								c_prev = c_reg[currentHiddenUnit][PE_count];
								gix_sum = gix_i[currentHiddenUnit][PE_count]+gix_h[currentHiddenUnit][PE_count]+(DotProductResult_t_gi)biases_gi[PE_count][currentHiddenUnit];
								gfx_sum = gfx_i[currentHiddenUnit][PE_count]+gfx_h[currentHiddenUnit][PE_count]+(DotProductResult_t_gf)biases_gf[PE_count][currentHiddenUnit];
								gox_sum = gox_i[currentHiddenUnit][PE_count]+gox_h[currentHiddenUnit][PE_count]+(DotProductResult_t_go)biases_go[PE_count][currentHiddenUnit];
								gi = sigmoid_lut<Lut_Entries_Sigmoid,gix_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gix_sum, lut_sigmoid_1);
								gf = sigmoid_lut<Lut_Entries_Sigmoid,gfx_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gfx_sum, lut_sigmoid_1);
								go = sigmoid_lut<Lut_Entries_Sigmoid,gox_accumulator_t,Sigmoid_limit_t,Sigmoid_step_t,Sigmoid_out_t>(gox_sum, lut_sigmoid_1);
								cix=cix_i[currentHiddenUnit][PE_count]+cix_h[currentHiddenUnit][PE_count]+(DotProductResult_t_ci)biases_ci[PE_count][currentHiddenUnit];
								ci = tanh_lut<Lut_Entries_Tanh,DotProductResult_t_ci,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(cix,lut_tanh_1);
								ci_gi_mul = ci * gi;
								if(currentColumn > 0)
								{
									gf_state_mul = gf * c_prev;
									tmp_c_next = ci_gi_mul + gf_state_mul;
								}
								else
								{
									tmp_c_next = ci_gi_mul;
								}
								tanhf_o = tanh_lut<Lut_Entries_Tanh,State_t,Tanh_limit_t,Tanh_step_t,Tanh_out_t>(tmp_c_next,lut_tanh_1);

								output = tanhf_o * go;
								c_next = tmp_c_next;
								//std::cout<<"h_"<<currentHiddenUnit + PE_count*NumberHiddenUnits/PE<<":"<<output;//<<std::endl;
								//std::cout<<output<<",";
								c_reg[currentHiddenUnit][PE_count] = c_next;
								temp_output_packed = temp_output_packed >> OutputActivationWidth;
								ap_uint<OutputActivationWidth> temp_output = *reinterpret_cast<ap_uint<OutputActivationWidth> *>(&output);
								temp_output_packed(OutputActivationWidth*NumberHiddenUnits/PE-1,OutputActivationWidth*(NumberHiddenUnits/PE-1)) = temp_output;
								output_reg(((currentHiddenUnit + PE_count*NumberHiddenUnits/PE) + 1) * OutputActivationWidth - 1, (currentHiddenUnit + PE_count*NumberHiddenUnits/PE) * OutputActivationWidth) = temp_output;
							}
								result_stream.write(temp_output_packed);
							}
							std::cout<<std::endl;
			if(currentColumn < numberOfColumns - 1)
				recurrent_stream.write(output_reg);

	}//column
}
#endif
