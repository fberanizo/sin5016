classdef SVM
   % Classe que implementa um SVM
   properties
      kernel
      C
      gamma
      degree
      coef0
      validationSize
      nSamples
      nFeatures
      alpha
      b
      error
   end
   methods
      function obj = SVM(varargin)
         p=inputParser;
         addRequired(p,'kernel',@(x)ismember(x, {'linear','rbf','polynomial'}));
         addParameter(p,'C',1.0,@isnumeric);
         addParameter(p,'gamma',0.0,@isnumeric);
         addParameter(p,'degree',2,@isscalar);
         addParameter(p,'coef0',1.0,@isnumeric);
         addParameter(p,'validationSize',0.25,@isnumeric);
         parse(p,varargin{:});
         obj.kernel=p.Results.kernel
         obj.C=p.Results.C
         obj.gamma=p.Results.gamma
         obj.degree=p.Results.degree
         obj.coef0=p.Results.coef0
         obj.validationSize=p.Results.validationSize
      end

      function obj = fit(obj,X,y)
         % Treina um SVM utilizando o algoritmo SMO

         %self.X, X_validation, self.y, y_validation = train_test_split(self.X, self.y, test_size=self.validation_size)
         [obj.nSamples,obj.nFeatures]=size(X);

         % Calcula valores do kernel para diminuir retrabalho
         obj.K=zeros(obj.nSamples);
         for sample1=1:obj.nSamples
            for sample1=2:obj.nSamples
               obj.K{sample1,sample2}=obj.kernelFunc(X(sample1,:),X(sample2,:));
            end
         end

         % Inicializa multiplicadores de Lagrange, parâmetro b, e cache de erro
         obj.alpha=zeros(obj.nSamples);
         obj.b=0;
         obj.error=zeros(obj.nSamples);

         epoch=1;
         epochsWithoutImprovement=0;
         bestParams=containers.Map({'validationError','alpha','b'},{inf,obj.alpha,obj.b});
         numChanged=0;
         examineAll=true;
         while (numChanged > 0 || examineAll) && (epochsWithoutImprovement < 20)
            numChanged=0;
            if examineAll
               for sample=1:obj.nSamples
                  numChanged=numChanged+obj.examineExample(obj,X,y,sample);
               end
               examineAll=false;
            else
               nonBoundSamples=find(obj.error>obj.tol & obj.error<(obj.C-obj.tol));
               for i=1:length(nonBoundSamples)
                  numChanged=numChanged+obj.examineExample(obj,X,y,nonBoundSamples(i));
               end
               if numChanged == 0
                  examineAll=true;
               end
            end

            trainError=obj.predict(obj,X).*y;
            trainError=double(length(find(trainError<0)))/double(obj.nSamples)*100.0;
            validationError=obj.predict(obj,XValidation).*yValidation;
            validationError=double(length(find(validationError<0)))/double(size(XValidation,1))*100.0;

            if validationError < bestParams('validationError')
               bestParams=containers.Map({'validationError','alpha','b'},{validationError,obj.alpha,obj.b});
               epochsWithoutImprovement=0;
            else
               epochsWithoutImprovement=epochsWithoutImprovement+1;
            end

            disp(epoch);
            disp(trainError);
            disp(validationError);
            epoch=epoch+1;
         end

         obj.alpha = bestParams('alpha')
         obj.b = bestParams('b')
      end

      function numChanged = examineExample(obj,X,y,sample2)
         X2=X(sample2);
         y2=y(sample2);
         alpha2=obj.alpha(sample2);
         if alpha2 < obj.tol || alpha2 > obj.C - obj.tol
            E2=obj.f(sample2)-y2;
         else
            E2=obj.error(sample2);
         end
         r2=E2*y2;

         if (r2 < -obj.tol && alpha2 < obj.C) || (r2 > obj.tol && alpha2 > 0)
            nonBoundSamples=find(obj.error>obj.tol & obj.error<(obj.C-obj.tol));
            if length(nonBoundSamples) > 0 && E2 > 0
               [E1,sample1]=max(obj.error);
               if obj.takeStep(obj,sample1,sample2,E2)
                  numChanged = 1;
                  return
               end
            elseif length(nonBoundSamples) > 0 && E2 < 0
               [E1,sample1]=min(obj.error);
               if obj.takeStep(obj,sample1,sample2,E2)
                  numChanged = 1
                  return
               end
            end

            if length(nonBoundSamples) > 0
               permutation = randperm(n);
               nonBoundSamples = nonBoundSamples(permutation);
               for i=1:length(nonBoundSamples)
                  if obj.takeStep(obj,nonBoundSamples(i),sample2, E2)
                     numChanged = 1
                     return
                  end
               end
            end

            allSamples=randperm(obj.nSamples);
            for i=1:length(allSamples)
               if obj.takeStep(allSamples(i),sample2,E2)
                  numChanged = 1
                  return
               end
            end
         end
      end

      function tookStep = takeStep(obj,sample1,sample2,E2)
         
      end

      function value = f(obj)
         value=sum(obj.y'.*obj.alpha'.*obj.K(sample,:)) - obj.b
      end

      function value = kernelFunc(obj,X1,X2)
         % Calcula saída da função kernel
         if obj.kernel == 'linear'
            value = sum(X1.*X2)
         elseif obj.kernel == 'rbf'
            value = numpy.exp(-self.gamma * numpy.power(numpy.linalg.norm(X1 - X2), 2))
         elseif self.kernel == 'polynomial'
            value = sum((obj.gamma.*X1.*X2+obj.coef0).^obj.degree)
         else
            throw(MException('kernelFunc:BadKernel',sprintf('Kernel %s not implemented', self.kernel)))
         end
      end
   end
end