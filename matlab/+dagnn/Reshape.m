classdef Reshape < dagnn.ElementWise
  properties
    s = [49,512,1]
    frozen = false
  end

  properties (Transient)
    mask
  end

  methods
    function outputs = forward(obj, inputs, params)
        input_size = size(inputs{1});
        outputs{1} = reshape(inputs{1}, obj.s(1), obj.s(2), obj.s(3), input_size(4));
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        input_size = size(inputs{1});
        derInputs{1} = reshape( derOutputs{1}, input_size(1), input_size(2), input_size(3), input_size(4)) ;
        derParams = {} ;
    end

    % ---------------------------------------------------------------------
    function obj = Reshape(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      obj.mask = [] ;
      obj.frozen = false ;
    end
  end
end
