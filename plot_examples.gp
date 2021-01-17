#! /usr/bin/gnuplot

set terminal png enhanced

# Single layer experiments
array experiments[1] = [ "spurious_1layer" ]
do for [ i = 1:|experiments| ] {
    set output "figures/".experiments[i]."_loss.png"
    set logscale y
    set xlabel "Batch Number"
    set ylabel "L1 Loss"
    set yrange [*:*]
    set key top right
    plot "< grep 'loss is' data/".experiments[i]."_1.txt" u 2:5 w lp t "Trial 1", \
         "< grep 'loss is' data/".experiments[i]."_2.txt" u 2:5 w lp t "Trial 2", \
         "< grep 'loss is' data/".experiments[i]."_3.txt" u 2:5 w lp t "Trial 3"

    set output "figures/".experiments[i]."_input_1_weights.png"
    unset logscale y
    set ylabel "Parameter Weight"
    set key bottom right
    set yrange [*:1.1]
    capture_str='s:At batch \([0-9]*\) layer 0 has weights \[\([^]]*\)\].*:\1 \2:'
    plot "< grep 'At batch' data/".experiments[i]."_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, first input", \
         "< grep 'At batch' data/".experiments[i]."_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, first input", \
         "< grep 'At batch' data/".experiments[i]."_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, first input"

    set output "figures/".experiments[i]."_input_2_weights.png"
    plot "< grep 'At batch' data/".experiments[i]."_1.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 1, second input", \
         "< grep 'At batch' data/".experiments[i]."_2.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 2, second input", \
         "< grep 'At batch' data/".experiments[i]."_3.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 3, second input"
}

array experiments[5] = [ "spurious_3layer", "spurious_3layer_dropout", "spurious_3layer_negative", \
                         "spurious_3layer_big_batch", "spurious_3layer_big_batch_negative" ]
do for [ i = 1:|experiments| ] {
    set output "figures/".experiments[i]."_loss.png"
    set logscale y
    set xlabel "Batch Number"
    set ylabel "L1 Loss"
    set yrange [*:*]
    set key top right
    plot "< grep 'loss is' data/".experiments[i]."_1.txt" u 2:5 w lp t "Trial 1", \
         "< grep 'loss is' data/".experiments[i]."_2.txt" u 2:5 w lp t "Trial 2", \
         "< grep 'loss is' data/".experiments[i]."_3.txt" u 2:5 w lp t "Trial 3"

    set output "figures/".experiments[i]."_input_1_correlation.png"
    unset logscale y
    set ylabel "Parameter Correlation"
    set key bottom right
    set yrange [*:1.1]
    capture_str='s:At batch \([0-9]*\) input \[1.0, 0.0\] encodes to \(.*\):\1 \2:'
    plot "< grep '1.0, 0.0' data/".experiments[i]."_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, first input", \
         "< grep '1.0, 0.0' data/".experiments[i]."_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, first input", \
         "< grep '1.0, 0.0' data/".experiments[i]."_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, first input"

    capture_str='s:At batch \([0-9]*\) input \[0.0, 1.0\] encodes to \(.*\):\1 \2:'
    set output "figures/".experiments[i]."_input_2_correlation.png"
    plot "< grep '0.0, 1.0' data/".experiments[i]."_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, second input", \
         "< grep '0.0, 1.0' data/".experiments[i]."_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, second input", \
         "< grep '0.0, 1.0' data/".experiments[i]."_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, second input"
}




#set output figures/"example_overfit_1.png"
#set logscale y
#set xlabel "Batch Number"
#set ylabel "L1 Loss"
#set yrange [*:*]
#set key top right
#plot "< grep 'loss is' example_overfit_1_1.txt" u 2:5 w lp t "Trial 1", \
#     "< grep 'loss is' example_overfit_1_2.txt" u 2:5 w lp t "Trial 2", \
#     "< grep 'loss is' example_overfit_1_3.txt" u 2:5 w lp t "Trial 3"
#
#set output figures/"example_overfit_1_input_1_weights.png"
#unset logscale y
#set ylabel "Parameter Weight"
#set key bottom right
#set yrange [*:1.1]
#capture_str='s:At batch \([0-9]*\) layer 0 has weights \[\([^]]*\)\].*:\1 \2:'
#plot "< grep 'At batch' example_overfit_1_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, first input weight", \
#     "< grep 'At batch' example_overfit_1_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, first input weight", \
#     "< grep 'At batch' example_overfit_1_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, first input weight"
#
#set output figures/"example_overfit_1_input_2_weights.png"
#plot "< grep 'At batch' example_overfit_1_1.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 1, second input weight", \
#     "< grep 'At batch' example_overfit_1_2.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 2, second input weight", \
#     "< grep 'At batch' example_overfit_1_3.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 3, second input weight"
#
#
#set output figures/"example_overfit_2.png"
#set xlabel "Batch Number"
#set ylabel "L1 Loss"
#set logscale y
#set yrange [*:*]
#set key top right
#plot "< grep 'loss is' example_overfit_2_1.txt" u 2:5 w lp t "Trial 1", \
#     "< grep 'loss is' example_overfit_2_2.txt" u 2:5 w lp t "Trial 2", \
#     "< grep 'loss is' example_overfit_2_3.txt" u 2:5 w lp t "Trial 3"
#
#set output figures/"example_overfit_2_input_1_weights.png"
#set ylabel "Parameter Weight"
#unset logscale y
#set key bottom right
#set yrange [*:1.1]
#capture_str='s:At batch \([0-9]*\) layer 0 has weights \[\([^]]*\)\].*:\1 \2:'
#plot "< grep 'At batch' example_overfit_2_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, first input weight", \
#     "< grep 'At batch' example_overfit_2_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, first input weight", \
#     "< grep 'At batch' example_overfit_2_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, first input weight"
#
#set output figures/"example_overfit_2_input_2_weights.png"
#plot "< grep 'At batch' example_overfit_2_1.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 1, second input weight", \
#     "< grep 'At batch' example_overfit_2_2.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 2, second input weight", \
#     "< grep 'At batch' example_overfit_2_3.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 3, second input weight"
#
#
#set output figures/"example_overfit_3.png"
#set xlabel "Batch Number"
#set ylabel "L1 Loss"
#set logscale y
#set yrange [*:*]
#set key top right
#plot "< grep 'loss is' example_overfit_3_1.txt" u 2:5 w lp t "Trial 1", \
#     "< grep 'loss is' example_overfit_3_2.txt" u 2:5 w lp t "Trial 2", \
#     "< grep 'loss is' example_overfit_3_3.txt" u 2:5 w lp t "Trial 3"
#
#set output figures/"example_overfit_3_input_1_weights.png"
#set ylabel "Parameter Weight"
#unset logscale y
#set key bottom right
#set yrange [*:1.1]
#capture_str='s:At batch \([0-9]*\) layer 0 has weights \[\([^]]*\)\].*:\1 \2:'
#plot "< grep 'At batch' example_overfit_3_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, first input weight", \
#     "< grep 'At batch' example_overfit_3_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, first input weight", \
#     "< grep 'At batch' example_overfit_3_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, first input weight"
#
#set output figures/"example_overfit_3_input_2_weights.png"
#plot "< grep 'At batch' example_overfit_3_1.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 1, second input weight", \
#     "< grep 'At batch' example_overfit_3_2.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 2, second input weight", \
#     "< grep 'At batch' example_overfit_3_3.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 3, second input weight"
#
#
#set output figures/"two_variable_overfit_1.png"
#set xlabel "Batch Number"
#set ylabel "L1 Loss"
#set logscale y
#set yrange [*:*]
#set key top right
#plot "< grep 'loss is' two_variable_overfit_1_1.txt" u 2:5 w lp t "Trial 1", \
#     "< grep 'loss is' two_variable_overfit_1_2.txt" u 2:5 w lp t "Trial 2", \
#     "< grep 'loss is' two_variable_overfit_1_3.txt" u 2:5 w lp t "Trial 3"
#
#set output figures/"two_variable_overfit_input_1_weights.png"
#set ylabel "Parameter Weight"
#unset logscale y
#set key bottom right
#set yrange [*:1.1]
#capture_str='s:At batch \([0-9]*\) layer 0 has weights \[\([^]]*\)\].*:\1 \2:'
#plot "< grep 'At batch' two_variable_overfit_1_1.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 1, first input weight", \
#     "< grep 'At batch' two_variable_overfit_1_2.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 2, first input weight", \
#     "< grep 'At batch' two_variable_overfit_1_3.txt | sed '".capture_str."'" u 1:2 w lp t "Trial 3, first input weight"
#
#set output figures/"two_variable_overfit_input_2_weights.png"
#plot "< grep 'At batch' two_variable_overfit_1_1.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 1, second input weight", \
#     "< grep 'At batch' two_variable_overfit_1_2.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 2, second input weight", \
#     "< grep 'At batch' two_variable_overfit_1_3.txt | sed '".capture_str."'" u 1:3 w lp t "Trial 3, second input weight"
