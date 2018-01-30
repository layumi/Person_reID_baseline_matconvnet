classdef Global_Pooling < dagnn.Filter
  properties
    method = 'avg'
    poolSize = [1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(self, inputs, params)
      self.poolSize(1) = size(inputs{1},1);
      self.poolSize(2) = size(inputs{1},2);
      outputs{1} = vl_nnpool(inputs{1}, self.poolSize, ...
                             'pad', self.pad, ...
                             'stride', self.stride, ...
                             'method', self.method, ...
                             self.opts{:}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      self.poolSize(1) = size(inputs{1},1);
      self.poolSize(2) = size(inputs{1},2);
      derInputs{1} = vl_nnpool(inputs{1}, self.poolSize, derOutputs{1}, ...
                               'pad', self.pad, ...
                               'stride', self.stride, ...
                               'method', self.method, ...
                               self.opts{:}) ;
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Global_Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
