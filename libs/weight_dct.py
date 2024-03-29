import torch
def weight_dct(args,device):
    if args.block_size==32:
        weight_dct = [
    0,   27,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,
    93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  27,  93,  93,  93,  93,  93,
    93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  94,  94,
    94,  94,  94,  94,  94,  94,  94,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,
    93,  93,  93,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,
    94,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  93,  94,  94,  94,  94,  94,  94,  94,
    94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  93,  93,  93,  93,  93,
    93,  93,  93,  93,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  95,
    95,  95,  95,  95,  95,  95,  95,  96,  93,  93,  93,  93,  93,  93,  93,  94,  94,  94,  94,
    94,  94,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  95,  95,  96,  96,  96,  96,  96,
    96,  97,  93,  93,  93,  93,  93,  93,  94,  94,  94,  94,  94,  94,  94,  94,  94,  94,  95,
    95,  95,  95,  95,  95,  96,  96,  96,  96,  97,  97,  97,  97,  98,  98,  93,  93,  93,  93,
    93,  94,  94,  94,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  95,  96,  96,  96,  96,
    97,  97,  97,  98,  98,  98,  99,  99,  99,  93,  93,  93,  93,  93,  94,  94,  94,  94,  94,
    94,  94,  95,  95,  95,  95,  95,  96,  96,  96,  97,  97,  97,  98,  98,  98,  99,  99,  100,
    100, 101, 101, 93,  93,  93,  93,  94,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  96,
    96,  96,  97,  97,  97,  98,  98,  99,  99,  100, 100, 101, 101, 102, 102, 103, 93,  93,  93,
    93,  94,  94,  94,  94,  94,  94,  95,  95,  95,  95,  96,  96,  96,  97,  97,  98,  98,  99,
    99,  100, 100, 101, 102, 102, 103, 104, 104, 105, 93,  93,  93,  94,  94,  94,  94,  94,  94,
    95,  95,  95,  96,  96,  96,  97,  97,  98,  98,  99,  99,  100, 100, 101, 102, 102, 103, 104,
    105, 106, 107, 107, 93,  93,  93,  94,  94,  94,  94,  94,  95,  95,  95,  96,  96,  96,  97,
    97,  98,  98,  99,  100, 100, 101, 102, 102, 103, 104, 105, 106, 107, 108, 109, 110, 93,  93,
    93,  94,  94,  94,  94,  94,  95,  95,  95,  96,  96,  97,  97,  98,  99,  99,  100, 101, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 93,  93,  93,  94,  94,  94,  94,  95,
    95,  95,  96,  96,  97,  97,  98,  99,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 112, 113, 115, 116, 93,  93,  94,  94,  94,  94,  94,  95,  95,  96,  96,  97,  97,  98,
    99,  99,  100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 112, 113, 115, 116, 118, 120, 93,
    93,  94,  94,  94,  94,  95,  95,  95,  96,  96,  97,  98,  99,  99,  100, 101, 102, 103, 104,
    105, 107, 108, 109, 111, 113, 114, 116, 118, 120, 122, 124, 93,  93,  94,  94,  94,  94,  95,
    95,  96,  96,  97,  98,  98,  99,  100, 101, 102, 103, 104, 106, 107, 108, 110, 112, 113, 115,
    117, 119, 121, 123, 126, 128, 93,  93,  94,  94,  94,  94,  95,  95,  96,  97,  97,  98,  99,
    100, 101, 102, 103, 104, 106, 107, 109, 110, 112, 114, 116, 118, 120, 122, 125, 127, 130, 133,
    93,  93,  94,  94,  94,  95,  95,  96,  96,  97,  98,  99,  100, 101, 102, 103, 104, 106, 107,
    109, 110, 112, 114, 116, 119, 121, 123, 126, 129, 132, 135, 138, 93,  93,  94,  94,  94,  95,
    95,  96,  97,  97,  98,  99,  100, 101, 103, 104, 105, 107, 109, 110, 112, 114, 117, 119, 122,
    124, 127, 130, 133, 136, 140, 144, 93,  93,  94,  94,  94,  95,  95,  96,  97,  98,  99,  100,
    101, 102, 104, 105, 107, 108, 110, 112, 114, 117, 119, 122, 125, 128, 131, 134, 138, 142, 146,
    150, 93,  93,  94,  94,  94,  95,  96,  96,  97,  98,  99,  100, 102, 103, 105, 106, 108, 110,
    112, 114, 117, 119, 122, 125, 128, 131, 135, 139, 143, 147, 152, 157, 93,  94,  94,  94,  95,
    95,  96,  97,  98,  99,  100, 101, 102, 104, 106, 107, 109, 112, 114, 116, 119, 122, 125, 128,
    132, 135, 140, 144, 148, 153, 159, 164, 93,  94,  94,  94,  95,  95,  96,  97,  98,  99,  100,
    102, 103, 105, 107, 109, 111, 113, 116, 119, 122, 125, 128, 132, 136, 140, 144, 149, 154, 160,
    166, 172, 93,  94,  94,  94,  95,  96,  96,  97,  98,  100, 101, 102, 104, 106, 108, 110, 113,
    115, 118, 121, 124, 128, 131, 135, 140, 145, 150, 155, 161, 167, 174, 181, 93,  94,  94,  94,
    95,  96,  97,  98,  99,  100, 102, 103, 105, 107, 109, 112, 114, 117, 120, 123, 127, 131, 135,
    140, 144, 150, 155, 161, 168, 175, 182, 191, 93,  94,  94,  94,  95,  96,  97,  98,  99,  101,
    102, 104, 106, 108, 110, 113, 116, 119, 122, 126, 130, 134, 139, 144, 149, 155, 161, 168, 175,
    183, 192, 201, 93,  94,  94,  95,  95,  96,  97,  98,  100, 101, 103, 105, 107, 109, 112, 115,
    118, 121, 125, 129, 133, 138, 143, 148, 154, 161, 168, 175, 184, 193, 202, 213, 93,  94,  94,
    95,  95,  96,  97,  99,  100, 102, 104, 106, 108, 110, 113, 116, 120, 123, 127, 132, 136, 142,
    147, 153, 160, 167, 175, 183, 193, 203, 214, 225, 93,  94,  94,  95,  95,  96,  98,  99,  101,
    102, 104, 107, 109, 112, 115, 118, 122, 126, 130, 135, 140, 146, 152, 159, 166, 174, 182, 192,
    202, 214, 226, 239, 93,  94,  94,  95,  96,  97,  98,  99,  101, 103, 105, 107, 110, 113, 116,
    120, 124, 128, 133, 138, 144, 150, 157, 164, 172, 181, 191, 201, 213, 225, 239, 255,
]
    

    elif args.block_size==16:
        weight_dct = [
    0,   27,  93,  93,  93,  93,  93,  93,  93,  93,  93,  94,  94,  94,  94,  94,  27,  93,  93,
    93,  93,  94,  94,  94,  94,  94,  94,  94,  94,  94,  95,  95,  93,  93,  93,  94,  94,  94,
    94,  94,  94,  95,  95,  95,  96,  96,  96,  97,  93,  93,  94,  94,  94,  94,  94,  95,  95,
    96,  96,  97,  97,  98,  99,  99,  93,  93,  94,  94,  94,  95,  95,  96,  96,  97,  98,  99,
    100, 101, 102, 103, 93,  94,  94,  94,  95,  95,  96,  97,  98,  99,  100, 101, 102, 104, 106,
    107, 93,  94,  94,  94,  95,  96,  97,  98,  99,  101, 102, 104, 106, 108, 110, 113, 93,  94,
    94,  95,  96,  97,  98,  99,  101, 103, 105, 107, 110, 113, 116, 120, 93,  94,  94,  95,  96,
    98,  99,  101, 103, 106, 108, 112, 115, 119, 123, 128, 93,  94,  95,  96,  97,  99,  101, 103,
    106, 109, 112, 116, 121, 126, 132, 138, 93,  94,  95,  96,  98,  100, 102, 105, 108, 112, 117,
    122, 128, 134, 142, 150, 94,  94,  95,  97,  99,  101, 104, 107, 112, 116, 122, 128, 135, 144,
    153, 164, 94,  94,  96,  97,  100, 102, 106, 110, 115, 121, 128, 135, 145, 155, 167, 181, 94,
    94,  96,  98,  101, 104, 108, 113, 119, 126, 134, 144, 155, 168, 183, 201, 94,  95,  96,  99,
    102, 106, 110, 116, 123, 132, 142, 153, 167, 183, 203, 225, 94,  95,  97,  99,  103, 107, 113,
    120, 128, 138, 150, 164, 181, 201, 225, 255,]
    
    elif args.block_size==8:
        weight_dct = [
    0,  27, 94,  94,  94,  94,  94,  95,  27, 94, 94,  95,  96,  97,  98,  99,
    94, 94, 95,  97,  99,  101, 104, 107, 94, 95, 97,  99,  103, 107, 113, 120,
    94, 96, 99,  103, 109, 116, 126, 138, 94, 97, 101, 107, 116, 128, 144, 164,
    94, 98, 104, 113, 126, 144, 168, 201, 95, 99, 107, 120, 138, 164, 201, 255,]

    else:
        assert(f'block size {block_size} is not supported.')
    weight_dct      = torch.tensor(weight_dct).view(args.block_size, args.block_size)
    weight_dct      = weight_dct.to(device)
    return weight_dct