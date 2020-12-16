############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project ptblm411
set_top topLevel_BLSTM_CTC
add_files ptblm411/solution1/activations.hpp -cflags "--std=c++14" -csimflags "--std=c++14"
add_files ptblm411/solution1/hw_config.h -cflags "--std=c++14" -csimflags "--std=c++14"
add_files ptblm411/solution1/hw_lstm.hpp -cflags "--std=c++14" -csimflags "--std=c++14"
add_files ptblm411/solution1/r_model_fw_bw.hpp -cflags "--std=c++14" -csimflags "--std=c++14"
add_files ptblm411/solution1/top.cpp -cflags "--std=c++14" -csimflags "--std=c++14"
add_files -tb ptblm411/solution1/main.cpp -csimflags "--std=c++14"
open_solution "solution1"
set_part {xc7z100iffv900-2L}
create_clock -period 50 -name default
config_export -format ip_catalog -rtl verilog -vivado_optimization_level 2 -vivado_phys_opt place -vivado_report_level 0
config_sdx -optimization_level none -target none
source "./ptblm411/solution1/directives.tcl"
csim_design -clean -compiler clang
csynth_design
cosim_design -compiler gcc -tool xsim
export_design -rtl verilog -format ip_catalog
