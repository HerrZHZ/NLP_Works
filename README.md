# NLP_Works
verilog.rar文件中是代码生成的更底层的仿真文件。
针对量化后的神经网络模型依然很大的问题，本次机器翻译大作业从硬件的角度，通过Vivado的HLS高层次综合工具在zynq-7000平台上实现了一个基于LSTM的语言模型的加速器。
报告结合具体平台给出了各个模块的具体实现细节，个人主要工作在报告中已详细给出。
报告最后利用ptb数据集训练了一个量化的语言模型，将模型部署在硬件实现上后，利用Vivado的综合和布局布线工具以及仿真工具验证设计结果的正确性，并对比两者之间的误差。发现对比CPU平台误差完全在可接受的范围内，证明了Normalization应用在硬件实现的可行性，而且功耗较低。
实验结果的精度对比见报告附录1。
加速器得到的结果（只截取了一部分）对应的布局布线后仿真波形图见报告附录2。
在报告中的最后部分简要总结了上课讲的部分机器翻译知识点内容和对本次机器翻译课程的建议。
