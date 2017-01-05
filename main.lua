require 'nn'
require 'nngraph'
local caffegraph = require 'caffegraph'
local npy4th = require 'npy4th'

local dirmodel = 'resnet152'
local pathprototxt = paths.concat(dirmodel, 'ResNet-152-448-deploy.prototxt')
local pathcaffemodel = paths.concat(dirmodel, 'ResNet-152-model.caffemodel')
local pathmeancaffe = paths.concat(dirmodel, 'ResNet_mean.binaryproto')
local pathmeannpy = paths.concat(dirmodel, 'ResNet_mean.npy')
local pathmeant7 = paths.concat(dirmodel, 'ResNet_mean.t7')
local pathnet = paths.concat(dirmodel, 'resnet152caffe.t7')
os.execute('mkdir -p '..dirmodel)

if not paths.filep(pathprototxt) then
  os.execute('wget https://raw.githubusercontent.com/akirafukui/vqa-mcb/master/preprocess/ResNet-152-448-deploy.prototxt -P '..pathprototxt)
end

assert(paths.filep(pathcaffemodel) and paths.filep(pathmeancaffe),
  'Download ResNet-152-model.caffemodel to '..dirmodel..' on https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777')

local net = caffegraph.load(pathprototxt, pathcaffemodel)
net:float()
print('Save net to '..pathnet)
torch.save(pathnet, net)

print('Convert mean to '..pathmeannpy)
os.execute('python convert_protomean.py '..pathmeancaffe..' '..pathmeannpy)

print('Convert mean to '..pathmeant7)
local mean = npy4th.loadnpy(pathmeannpy)
print(mean:size())
print(mean:mean())
torch.save(pathmeant7, mean)

print('Beware this net takes img_color=BGR and intensity=[0, 255] as input instead of RGB[0,1] (to verify)')




