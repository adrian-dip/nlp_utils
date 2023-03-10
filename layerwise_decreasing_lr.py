### Implementation of https://arxiv.org/pdf/1905.05583.pdf, section 5.3.4


def layerwise_lr(model, lr):
  
  # get a list of the model's layers
  
  layers = []
  for idx, (name, param) in enumerate(model.named_parameters()):
      layers.append(name)


  # reverse(): The list goes from the bottom layers to the top. 
  
  layers.reverse()

  lr_fn = lr
  ξ = 0.95         # decay factor by which to decrease (lr * epsilon = 5% per layer)

  parameters = []
  prev_group_name = layers[2].split('.')[2]       
                                                 # Check your model's layer names to make sure this line of code works
                                                 # Current behavior: "encoder.layer.11.output.LayerNorm.bias" -> "11"

  for idx, name in enumerate(layers):

      # parameter group name
      cur_group_name = name.split('.')[2]

      # update learning rate
      if cur_group_name != prev_group_name:      # Result: lr = 0.000017, encoder.layer.11.output.LayerNorm.bias,
          lr_fn *= ξ                             # lr = 0.000016, encoder.layer.10.output.LayerNorm.bias,       
      prev_group_name = cur_group_name           # etc. 

      # append layer parameters
      parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                      'lr':     lr}]
