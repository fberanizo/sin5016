classdef MLP
   % Classe que implementa um MLP
   properties
      hiddenLayerSize
      maxEpochs
      validationSize
   end
   methods
      function obj = MLP(varargin)
         p=inputParser;
         addRequired(p,'hiddenLayerSize',@isnumeric));
         addParameter(p,'maxEpochs',1000,@isnumeric);
         addParameter(p,'validationSize',0.25,@isnumeric);
         parse(p,varargin{:});
         obj.hiddenLayerSize=p.Results.hiddenLayerSize;
         obj.maxEpochs=p.Results.maxEpochs;
         obj.validationSize=p.Results.validationSize;
      end

      function fit(obj,X,y)
        % Ajusta cada feature entre -1 e 1
        X=2*(X-min(X(:)))/(max(X(:))-min(X(:)))-1;
        % Adiciona coluna bias
        X=[X ones(size(X,1),1)];
        % Divide conjunto de dados em treino e validação
        [TrainInd,ValidationInd,TestInd]=dividerand(size(X,1),1.0-obj.validationSize,obj.validationSize,.0);
        obj.X=X(TrainInd,:);
        obj.y=y(TrainInd,:);
        XValidation=X(ValidationInd,:);
        yValidation=y(ValidationInd,:);
        % Obtém número de entradas, saídas e características
        [obj.nSamples,obj.nFeatures]=size(obj.X);
        [obj.nSamples,obj.nOutputs]=size(obj.y);
        % Inicializa pesos da rede
        W1=rand(obj.hiddenLayerSize,obj.nFeatures)
        W2=rand(obj.nOutputs,1+obj.hiddenLayerSize)
      end
  end
end