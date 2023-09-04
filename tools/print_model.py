import torch

def print_model(model):
    blank = ' '
    print('-'*90)
    print('|'+' '*11+'weight name'+' '*10+'|' \
            +' '*15+'weight shape'+' '*15+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-'*90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4
    
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30: 
            key = key + (30-len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40-len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10-len(str_num)) * blank
    
        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-'*90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-'*90)


def print_flops(model, input_size):
    import torch
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
  
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
  
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
  
        list_conv.append(flops)
  
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
  
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
  
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
  
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
  
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
  
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
  
        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width
  
        list_pooling.append(flops)
  
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
  
    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    
    input = torch.rand(2, 3, input_size, input_size)
    
    _ = model(input)
  
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
  
    print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6 / 2))
