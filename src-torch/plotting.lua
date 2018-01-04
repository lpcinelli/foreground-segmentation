require 'gnuplot'

local plotting = {}

-- stats should be 2D tensor
function plotting.loss_curve(stats, opt)
  local fn = paths.concat(opt.save,'training_loss.eps')
  gnuplot.epsfigure(fn)
  gnuplot.title('Training loss\nBest Value : ' .. tostring(stats:min()))
  gnuplot.grid('off')
  local xs = torch.range(1, stats:size(1))
  gnuplot.plot(
    { 'train', xs, torch.Tensor(stats), '+' }
  )
  gnuplot.axis({ 1, stats:size(1), 0, stats:max()})
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  gnuplot.plotflush()
end

-- trainingStats and testingStats should be tables
function plotting.curve(trainingStats, testingStats, title, filename, opt, ylabel, ylimit)
  local fn = paths.concat(opt.save, filename .. '.eps')
  gnuplot.epsfigure(fn)
  gnuplot.title(title .. '\nBest Test Value : ' .. tostring(torch.Tensor(testingStats):min()))
  gnuplot.grid('on')
  local xsTrain = torch.range(1, #trainingStats)
  local xsTest = torch.range(1, #testingStats)
  -- local xsTest = torch.range(1, testingStats:size(1))

  gnuplot.plot(
    { 'train', xsTrain, torch.Tensor(trainingStats), '-' },
    { 'test', xsTest, torch.Tensor(testingStats), '-' }
  )
  if ylabel == nil then ylabel = 'error' end
  if ylimit == nil then ylimit = 1 end
  gnuplot.axis({ 1, math.max(#testingStats, #trainingStats), 0, ylimit})
  -- gnuplot.axis({ 0, math.max(trainingStats:size(1),testingStats:size(1)), 0, ylimit})
  gnuplot.xlabel('epoch')
  gnuplot.ylabel(ylabel)
  gnuplot.plotflush()
end

return plotting
