# I2V debug log

Lets start by printing the height and the width of the original image: HeightL 259, Width:194
Patch Size: (1, 2, 2), VAE Stride: (4, 16, 16)
this is dh and dw: 32, 32
This is the best output size: 800, 1088
This is the scale 4.2007722007722
Size of the image before resizing: height=259, width=194
Size of the image after resizing: height=1088, width=815
Size of the image after cropping: height=1088, width=800
OOOOOK, this is the seq len: 26350
The noise tensor has been just generated and this is the shape: torch.Size([48, 31, 68, 50])
Well Well the image latent has been generated, its called z: len=1, z[0].shape=torch.Size([48, 1, 68, 50])
The timestep vector has been just generated, here is the shape: torch.Size([50])
why not have a look at the entire timestep vector: tensor([999, 995, 991, 987, 982, 978, 973, 968, 963, 957, 952, 946, 940, 934,
        927, 920, 913, 906, 898, 890, 882, 873, 863, 854, 843, 833, 821, 809,
        796, 783, 768, 753, 737, 720, 701, 681, 660, 636, 611, 584, 555, 522,
        487, 448, 405, 356, 302, 241, 172,  92], device='cuda:0')
The masks which have been troubling us are here: len(mask1)=1, mask1[0].shape=torch.Size([48, 31, 68, 50]), len(mask2)=1, mask2[0].shape=torch.Size([48, 31, 68, 50])
lets have a little peak into this baddies: tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         ...,

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]]],


        [[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         ...,

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],

         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]]]], device='cuda:0')
Lets see how the latent has changed after it has been multiplied with the mask
latent shape: (48, 31, 68, 50)  (assuming axis 1 = 31 frames)
  frame  0: changed=True  max_abs_diff=5.827572  mean_abs_diff=0.898448  num_changed_elems=163200/163200
  frame  1: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  2: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  3: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  4: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  5: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  6: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  7: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  8: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame  9: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 10: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 11: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 12: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 13: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 14: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 15: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 16: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 17: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 18: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 19: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 20: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 21: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 22: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 23: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 24: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 25: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 26: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 27: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 28: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 29: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
  frame 30: changed=False  max_abs_diff=0.000000  mean_abs_diff=0.000000  num_changed_elems=0/163200
first frame of latent == first frame of image latent z? True
Before the for loop of the timesteps begin, lets examine the mask which is going into the calculations: torch.Size([31, 34, 25])
this is iteration 0
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 1
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 2
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 3
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 4
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 5
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 6
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 7
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 8
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 9
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 10
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 11
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 12
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 13
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 14
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 15
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 16
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 17
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 18
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 19
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 20
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 21
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 22
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 23
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 24
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 25
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 26
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 27
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 28
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 29
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 30
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 31
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 32
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 33
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 34
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 35
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 36
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 37
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 38
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 39
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 40
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 41
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 42
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 43
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 44
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 45
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 46
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 47
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 48
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
this is iteration 49
len(timestep)=1
shape of timestep before mutliplication with mask: torch.Size([1])
shape of timestep after mutliplication with mask: torch.Size([26350])
shape of timestep after concat: torch.Size([26350])
This is the final timestep shape before going into the model: torch.Size([1, 26350])
