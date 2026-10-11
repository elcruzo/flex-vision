# Experimental envelope for a standard model-input tensor.
# Header is copied from the source Image, not replaced with processing time.
std_msgs/Header header
tensor_msgs/ExperimentalTensor tensor
string layout
uint32 source_width
uint32 source_height
string source_encoding
# Actual rounded resize dimensions and padding for source-coordinate mapping.
uint32 resized_width
uint32 resized_height
uint32 padding_left
uint32 padding_top
