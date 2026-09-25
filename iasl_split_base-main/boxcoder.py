import torch
import torch.nn as nn

import random
from bitstring import BitStream, BitArray

class boxcoder(nn.Module):
    def __init__(self, bottleneck_channel=12):
        super(boxcoder, self).__init__()

        self.preprocess_conv = nn.Conv2d(bottleneck_channel, bottleneck_channel, kernel_size=5, stride=1, padding=2)
        self.postprocess_conv = nn.Conv2d(bottleneck_channel, bottleneck_channel, kernel_size=5, stride=1, padding=2)
        self.bpp = None
        self.byte_size = 0
        self.compressed_size = 0

        self.is_training = True

        #add one logically since we have 0
        self.bit_depth = 255

        self.values = [-0.0000001]
        increment = 1/self.bit_depth
        value = 0
        for i in range(self.bit_depth):
            value += increment
            self.values.append(increment)

        self.values[-1] = 1.0 #in case there are some floating point rounding errors

        #tensor shape from the encoder may not always be the same so we will set this for each sample
        self.decode_shape = None

    def forward_old(self, x):
        x = self.preprocess_conv(x)

        x_min = x.min(1, keepdim=True)[0]
        x -= x_min
        x_max = x.max(1, keepdim=True)[0]
        x /= x_max

        ones = torch.ones(x.shape).cuda()
        zeros = torch.zeros(x.shape).cuda()

        ones = torch.where(x>=0.5, ones, x).detach().cpu().cuda()
        ones = ones - torch.where(x<0.5, zeros, x)

        zeros = torch.where(x<0.5, x, zeros).detach().cpu().cuda()
        zeros = zeros * -1
        
        ones = ones + zeros
        x = x + ones

        x = x + (x.round() - (x.detach()))
        # print(x)
        self.size_value(x)

        # x = self.postprocess_conv(x)

        return x
    
    #forward for training qunatization process (no bitstream construction)
    def forward(self, x):
        # x = self.preprocess_conv(x)

        x, x_min, x_max = self.forward_quant(x)

        if self.is_training:
            self.size_value(x)

        x = self.forward_dequant(x, x_min, x_max)

        return x
    
    def forward_quant(self, x):
        self.byte_size = 0

        self.decode_shape = x.shape
        x = x.view(x.shape[0], -1)

        x_min = x.min(1, keepdim=True)[0]
        x -= x_min
        x_max = x.max(1, keepdim=True)[0]
        x /= x_max

        # print(x)
        x = x * self.bit_depth
        x = x + (x.round() - (x.detach()))

        x = x.view(self.decode_shape)

        return x, x_min, x_max

    def forward_dequant(self, x, x_min, x_max):
        input_shape = x.shape
        x = x.view(x.shape[0], -1)

        x /= self.bit_depth
        x *= x_max
        x += x_min

        x = x.view(input_shape)

        self.self_target = x
        
        return x
        
    def size_value(self, x):
        x = x.reshape(x.shape[0],-1)
        self.bpp = 0

        for b in range(x.shape[0]):
            sym = -1
            collected = 0
            bytes = 0
            for i in range(x.shape[1]):
                if x[b][i].item() != sym or collected==255:
                    bytes += 2
                    sym = x[b][i].item()
                    collected = 1
                else:
                    collected += 1
            
            # value_counts = torch.bincount(x[b], minlength=256).Float()
            # probabilities = value_counts / torch.sum(value_counts)
            # probabilities = torch.nn.functional.softmax(x[b].float(), dim=0)

            # Calculate the entropy
            self.bpp += (torch.sum(torch.diff(x[b]) ** 2))/(x.shape[1])
            # print(self.bpp)

            # x[b] /= 255
            # x[b] += torch.tensor(bytes).cuda()

        # self.bpp = (x.mean()**2)

        self.byte_size = bytes

    # #given quantized tensor fill with some dummy values of what we are missing
    # def dummy_fill(self, x):
    #     shape = x.shape
    #     x = x.reshape(x.shape[0],-1)

    #     for b in range(x.shape[0]):
    #         sym = -1
    #         collected = 0
    #         drop_bytes = True

    #         for i in range(x.shape[1]):
    #             if (x[b][i].item() != sym or collected==255):
    #                 #set drop_bytes here with gaussian (second if)
    #                 ...
                        
    #                 sym = x[b][i].item()
    #                 if drop_bytes:
    #                     x[b][i] = 3.0
    #                 collected = 1
    #             elif drop_bytes:
    #                 x[b][i] = 3.0
    #                 collected += 1
    #             else:
    #                 collected += 1

    #     x = x.reshape(shape)

    def encode_to_bitstream(self, x, x_min, x_max, packet_length):
        x = x.reshape(x.shape[0],-1)
        compressed_batch = []
        data_sizes = []

        #data payload size should be at least as small as the UPD header information
        assert (packet_length >= 64) and ((packet_length % 8) == 0) and (packet_length != 80)

        for b in range(x.shape[0]):
            bit_array = BitArray()
            sym = -1
            collected = 0

            #minimum packet size will be 80 for first packet 
            bit_array.append(f"uint16=0")
            bit_array.append(f"float32={x_min[b].item()}")
            bit_array.append(f"float32={x_max[b].item()}")
            
            
            first_packet = True
            payload_length = 0
            self.compressed_size = 0
            self.byte_size = 10
            position_value = 0
            for i in range(x.shape[1]):
                if x[b][i].item() != sym or collected==255:
                    if sym != -1:
                        #update index position and store it if we excede the packet length
                        if payload_length >= packet_length or first_packet:
                            bit_array.append(f"uint32={position_value}")
                            payload_length = 32
                            self.byte_size += 2
                            first_packet = False
                            
                        bit_array.append(f"uint8={sym}")
                        bit_array.append(f"uint8={collected}")
                        payload_length += 16
                        self.compressed_size += 2
                        self.byte_size += 2

                    sym = int(x[b][i].item())
                    position_value = i
                    collected = 1
                else:
                    collected += 1


            # data_sizes.append((position_value + payload_length))
            compressed_batch.append(bit_array)

        # self.byte_size = sum(data_sizes)/len(data_sizes)

        return compressed_batch

    #this process is not efficent but gets the job done
    def add_error(self, x, err, rec=0.0):
        new_x = []
        for bit_array in x:
            bit_string = bit_array.bin
            for i in range(len(bit_string)):
                flip_bit = (random.random() <= err) 
                # if flip_bit:
                #     flip_bit = (random.randint(1,100) >= rec) 

                if flip_bit and (bit_string[i] == '0'):
                    bit_string = bit_string[:i] + "1" + bit_string[i+1:]
                elif flip_bit:
                    bit_string = bit_string[:i] + "0" + bit_string[i+1:]

            new_x.append(BitArray("bin="+bit_string))

        x = new_x

        return x
    
    def decode_to_tensor(self, x, packet_length, use_HARQ_varient=None):
        decoded_x = torch.zeros(self.decode_shape).cuda()
        decoded_x = decoded_x.reshape(self.decode_shape[0], -1)
        x_min = torch.zeros(self.decode_shape[0], 1).cuda()
        x_max = torch.zeros(self.decode_shape[0], 1).cuda()

        for b in range(len(x)):
            for i in range(len(x[b])):
                if i == 0:
                    x_min[b][0] = x[b][16:48].float32
                    x_max[b][0] = x[b][48:80].float32

                if (i+1) >= 80 and (((i+1)-80)%packet_length == 0) and (i+1 != len(x[b])):
                    if (len(x[b]) - i+1) >= packet_length:
                        num_values = int((packet_length-32)/16)
                    else:
                        num_values = int(((len(x[b]) - i+1)-32)/16)
                    start_index = x[b][i+1:i+33].uint32

                    # print(num_values, start_index, len(x[b]))
                    dummy_value = 0
                    drop_packet = False
                    # if use_HARQ_varient is not None:
                    #     drop_packet = (random.random() > use_HARQ_varient)

                    for value_pairs in range(num_values):
                        offset = value_pairs*16
                        # value = int((x[b][i+(33+offset):i+(41+offset)].uint8)*0.05) if drop_packet else x[b][i+(33+offset):i+(41+offset)].uint8
                        value = dummy_value if drop_packet else x[b][i+(33+offset):i+(41+offset)].uint8
                        copies = x[b][i+(41+offset):i+(49+offset)].uint8

                        for _ in range(copies):
                            if start_index < decoded_x.shape[1]:
                                decoded_x[b][start_index] = value
                            start_index += 1

        decoded_x = decoded_x.reshape(self.decode_shape)

        return decoded_x, x_min, x_max


# tensor = torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8)
# channel_score = 11
# channel_mean = 9
# channel_std = 4
# mean_scale = 2
# std_scale = 0.5

# def connectionSimulator(x, channel_score, channel_mean=9, channel_std=4, mean_scale=2, std_scale=0.5):
#    ''' This function takes in a string, a channel quality index score, channel quality mean, channel quality std, and scaling
#        factors for the mean and std. It then uses the z score to create a Guassian distribution where the mean increases
#        as the z score increases and the std decreases as the absolute value of the z-score increases. This is then used
#        to stimulate poor connection by randomly replacing each byte with a 0 or deleting it. It is currently set to modify
#        30% of the data with the worst possible connectivity (a channel score of 1). It divides the tensor into packets and
#        prints which packets were modified.'''

#    x = x.reshape(-1)  # Flatten the tensor
#    packets = []
#    packet_length = 16  # Number of elements in each packet

#    for i in range(0, x.numel(), packet_length):
#        # Slice the current chunk of up to `packet_length` elements
#        chunk = x[i:i + packet_length]
#        bit_array = BitArray()

#        # Convert each value of the chunk into its bit representation
#        for value in chunk:
#            bit_array.append(f"uint8={value.item()}")

#        # Convert the BitArray to binary and append to packets. Packets is a list of the bit representation of each packet
#        packets.append(bit_array.bin)


#    # Calculate z score
#    z_score = (channel_score - channel_mean) / channel_std
#    print(f"Z-Score: {z_score}")


#    # Calculate a new mean and new std based on the z score

#    # A better connection and higher z score causes the new mean to increase
#    new_mean = channel_mean + z_score * mean_scale

#    # As the z score grows farther from 0, the new std decreases and the curve gets more vertical
#    new_std = channel_std / (1 + abs(z_score) * std_scale)
#    print(f"New Mean: {new_mean}, New Std: {new_std}")

#    count = 0  # Counter for modifications to calculate percent modified

#    # Create a new list for the packets that are modified
#    modified_packets = []

#    # Iterate over each packet in the list
#    for string in packets:
#         new_string = string
#         i = 0

#         # Iterate over each character in the packet
#         while i < len(string):
#             # Generate Guassian Distribution
#             index = random.gauss(new_mean, new_std)
#             if index < 4:
#                 count += 1
#                 if random.choice([True, False]):  # Randomly choose to delete or replace
#                     new_string = new_string[:i] + new_string[i+1:]  # Delete character
#                     continue  # Skip incrementing index since list has shifted
#                 else:
#                     new_string = new_string[:i] + "0" + new_string[i+1:]  # Replace character with '0'
#             i += 1
            
#         # Add each modified packet to the list
#         modified_packets.append(new_string)

#    original_string = ''.join(packets)
#    modified_string = ''.join(modified_packets)
#    percentage = count / len(original_string)
   
#    # Literally just print things idk why im commenting here
#    print(f"Percentage Modified: {percentage}")
#    print(f"Original String: {original_string}")
#    print(f"Modified String: {modified_string}")
#    print()
   
#    # Test whether each packet has changed or not
#    for i in range(len(packets)-1):
#        if packets[i] == modified_packets[i]:
#            print("Packet " + str(i + 1) + ": Success")
#        else:
#            print("Packet " + str(i + 1) + ": Failure")

# connectionSimulator(tensor, channel_score, channel_mean, channel_std, mean_scale, std_scale)

def process_raw(x, ber):
    byte_size = 0
    decode_shape = x.shape
    x = x.view(x.shape[0], -1)

    x_min = x.min(1, keepdim=True)[0]
    x -= x_min
    x_max = x.max(1, keepdim=True)[0]
    x /= x_max

    x = x * 255

    bit_array = BitArray()

    for i in range(x.shape[1]):
        bit_array.append(f"uint8={int(x[0][i])}")

    
    bit_string = bit_array.bin
    for i in range(len(bit_string)):
        flip_bit = (random.random() <= ber) 
        # if flip_bit:
        #     flip_bit = (random.randint(1,100) >= rec) 

        if flip_bit and (bit_string[i] == '0'):
            bit_string = bit_string[:i] + "1" + bit_string[i+1:]
        elif flip_bit:
            bit_string = bit_string[:i] + "0" + bit_string[i+1:]

    bit_array = BitArray("bin="+bit_string)

    offset = 0
    for i in range(x.shape[1]):
        x[0][i] = float(bit_array[offset:(offset+8)].uint8)
        offset+=8

    x /= 255
    x *= x_max
    x += x_min

    x = x.view(decode_shape)
    
    return x, byte_size

if __name__ == "__main__":
    x = torch.rand(1,2,3,3).cuda()
    bb = boxcoder(5).cuda()

    

    qx, x_min, x_max = bb.forward_quant(x)
    print(qx)
    eqx = bb.encode_to_bitstream(qx, x_min, x_max, 128)
    eqx = bb.add_error(eqx, (10**(-5)))
    dqx, x_min, x_max = bb.decode_to_tensor(eqx, 128)
    print(dqx)
    dx = bb.forward_dequant(dqx, x_min, x_max)

    print(x)
    print(dx)
    print((x-dx).mean())
    print(bb.bpp)

    # out = bb(x)