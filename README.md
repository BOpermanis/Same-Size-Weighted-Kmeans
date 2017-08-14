# Same-Size-Weighted-Kmeans

This code was helpful used in batching for training encoder-decoder recurrent neural networks.

When batching data for this type of models it is necessary to add padding to input and output sentences. But padding consumes memory.

For efficient batching it is relevant to cluster data points with similar input and output sequence lenghts.

This implemented clustering algo takes unique (input, output) lengths as data and number of instances #(input, output) as weights.
