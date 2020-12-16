template
<
unsigned int NUMBER_OF_LUT_ENTRIES,
typename Input_t,
typename Limit_t,
typename RecipStep_t,
typename Output_t
>
Output_t sigmoid_lut(Input_t & input, Output_t lut_sigmoid[NUMBER_OF_LUT_ENTRIES])
{
	Limit_t lower_limit = -5.0;
	Limit_t upper_limit = 5.0;
	RecipStep_t recip_step = 25.5;

	Input_t input_temp = input;
	Output_t output;

	// If we are outside of LUT range
	if (input_temp <= lower_limit)
	{
		output = lut_sigmoid[0];
	}
	else if (input_temp >= upper_limit)
	{
		output = lut_sigmoid[NUMBER_OF_LUT_ENTRIES-1];
	}
	else
	{
		// Scale from [lower, upper] to [0, N]
		Input_t t = input_temp - lower_limit;
		uint16_t index = t * recip_step;

		output = lut_sigmoid[index];
	}

	return output;
}

template
<
unsigned int NUMBER_OF_LUT_ENTRIES,
typename Input_t,
typename Limit_t,
typename RecipStep_t,
typename Output_t
>
Output_t tanh_lut(Input_t & input, Output_t lut_tanh[NUMBER_OF_LUT_ENTRIES])
{
	Limit_t lower_limit = -3.0;
	Limit_t upper_limit = 3.0;
	RecipStep_t recip_step = 42.5;

	Input_t input_temp = input;
	Output_t output;

	// If we are outside of LUT range
	if (input_temp <= lower_limit)
	{
		output = lut_tanh[0];
	}
	else if (input_temp >= upper_limit)
	{
		output = lut_tanh[NUMBER_OF_LUT_ENTRIES-1];
	}
	else
	{
		// Scale from [lower, upper] to [0, N]
		Input_t t = input_temp - lower_limit;
		uint16_t index = t * recip_step;

		output = lut_tanh[index];
	}

	return output;
}
