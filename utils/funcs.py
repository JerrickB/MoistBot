def conv_calc(input_width,kern_size,padding,stri):
    y = (input_width-kern_size+2*padding)/stri+1
    print(y)
    return y

