-- require 'torch'
local image = require 'image'
local lfs = require 'lfs'
local ffi = require 'ffi'
local paths = require 'paths'

-- absolute paths
srcDir = '/home/lpcinelli/repos/fb.resnet.torch.perso/gen/cdnet_70-30'
dstDir = '/home/lpcinelli/repos/fb.resnet.torch.perso/gen/imgs/cdnet_70-30'

function attrdir (srcPath, dstPath)
	for file in lfs.dir(srcPath) do
		if file ~= "." and file ~= ".." then
			local f = srcPath .. '/' .. file
			local g = dstPath .. '/' .. file
			print ("\t "..f)
	                local attr = lfs.attributes (f)
			assert (type(attr) == "table")
			if attr.mode == "directory" then
				paths.mkdir(g)
				attrdir (f,g)
			else
				img = torch.load(f)
				if not paths.filep(paths.concat(paths.dirname(g),'bg.jpg')) then
					image.save(paths.concat(paths.dirname(g),'bg.jpg'),img.input[1]:view(1, table.unpack(img.input[1]:size():totable())))
				end

				image.save(g .. '.jpg', img.input[2]:view(1, table.unpack(img.input[2]:size():totable())))
			end
		end
	end
end

attrdir (srcDir, dstDir)
