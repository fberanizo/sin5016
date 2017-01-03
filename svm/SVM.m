classdef SVM
   % Classe que implementa um SVM
   properties
      kernel
      C
      gamma
      degree
      coef0
      validationSize
      X
      y
      nSamples
      nFeatures
      nOutputs
      alpha
      b
      K
      tol
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
         obj.kernel=p.Results.kernel;
         obj.C=p.Results.C;
         obj.gamma=p.Results.gamma;
         obj.degree=p.Results.degree;
         obj.coef0=p.Results.coef0;
         obj.validationSize=p.Results.validationSize;
         obj.tol=1e-3;
      end

      function fit(obj,X,y)
         % Treina um SVM utilizando o algoritmo SMO
         % Aplica normalização aos dados
         obj.X=obj.scale(X)
         obj.y=y
         % TODO: Dividir conjunto de dados em treino e teste
         % TODO: Adicionar código de classificador multiclasses
         [obj.nSamples,obj.nFeatures]=size(X);
         [obj.nSamples,obj.nOutputs]=size(y);
         % Parâmetro gamma é 1/nFeatures por padrão
         if obj.gamma == 0
            obj.gamma=1.0/double(obj.nFeatures);
         end
         % Calcula valores do kernel para diminuir retrabalho
         obj.K=zeros(obj.nSamples,obj.nSamples);
         for sample1=1:obj.nSamples
            for sample2=2:obj.nSamples
               obj.K(sample1,sample2)=obj.kernelFunc(X(sample1,:),X(sample2,:));
            end
         end
         % Inicializa multiplicadores de Lagrange, parâmetro b, e cache de erros
         obj.alpha=zeros(obj.nSamples,1);
         obj.b=0;
         obj.error=zeros(obj.nSamples,1);
         % 
         epoch=1;
         epochsWithoutImprovement=0;
         bestParams=containers.Map({'validationError','alpha','b'},{inf,obj.alpha,obj.b});
         numChanged=0;
         examineAll=true;
         while (numChanged > 0 || examineAll) && (epochsWithoutImprovement < 20)
            numChanged=0;
            if examineAll
               for sample=1:obj.nSamples
                  numChanged=numChanged+obj.examineExample(sample);
               end
               examineAll=false;
            else
               nonBoundSamples=find(obj.error>obj.tol & obj.error<(obj.C-obj.tol));
               for i=1:length(nonBoundSamples)
                  numChanged=numChanged+obj.examineExample(nonBoundSamples(i));
               end
               if numChanged == 0
                  examineAll=true;
               end
            end
            % Calcula erro de treino e de validação
            trainError=obj.predict(obj.X).*y;
            trainError=double(length(find(trainError<0)))/double(obj.nSamples)*100.0;
            validationError=obj.predict(obj,XValidation).*yValidation;
            validationError=double(length(find(validationError<0)))/double(size(XValidation,1))*100.0;
            % Armazena parâmetros que tiveram menor erro de validação
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
         % Utiliza os parâmetros de menor erro de validação
         obj.alpha=bestParams('alpha');
         obj.b=bestParams('b');
      end

      function numChanged = examineExample(obj,sample2)
         X2=obj.X(sample2,:);
         y2=obj.y(sample2);
         alpha2=obj.alpha(sample2);
         if alpha2 < obj.tol || alpha2 > obj.C - obj.tol
            E2=obj.f(sample2)-y2;
         else
            E2=obj.error(sample2);
         end
         r2=E2*y2;
         % Testa se amostra viola as condições de KKT
         if (r2 < -obj.tol && alpha2 < obj.C) || (r2 > obj.tol && alpha2 > 0)
            nonBoundSamples=find(obj.error>obj.tol & obj.error<(obj.C-obj.tol));
            % Tenta heurística 1: amostras com maior erro
            if length(nonBoundSamples) > 0 && E2 > 0
               [E1,sample1]=max(obj.error);
               if obj.takeStep(sample1,sample2,E2)
                  numChanged = 1;
                  return
               end
            elseif length(nonBoundSamples) > 0 && E2 < 0
               [E1,sample1]=min(obj.error);
               if obj.takeStep(sample1,sample2,E2)
                  numChanged = 1;
                  return
               end
            end
            % Tenta heurística 2: amostras non-bound
            if length(nonBoundSamples) > 0
               permutation = randperm(n);
               nonBoundSamples = nonBoundSamples(permutation);
               for i=1:length(nonBoundSamples)
                  if obj.takeStep(nonBoundSamples(i),sample2, E2)
                     numChanged = 1;
                     return
                  end
               end
            end
            % Tenta heurística 3: amostras aleatórias
            allSamples=randperm(obj.nSamples);
            for i=1:length(allSamples)
               if obj.takeStep(allSamples(i),sample2,E2)
                  numChanged = 1;
                  return
               end
            end
         end
      end

      function tookStep = takeStep(obj,sample1,sample2,E2)
         if sample1 == sample2
            tookStep=false;
            return
         end
         X1=obj.X(sample1,:);
         y1=obj.y(sample1);
         alpha1=obj.alpha(sample1);
         X2=obj.X(sample2,:);
         y2=obj.y(sample2);
         alpha2=obj.alpha(sample2);
         if alpha1 < obj.tol || alpha1 > obj.C - obj.tol
            E1=obj.f(sample1)-y1;
         else
            E1=obj.error(sample1);
         end
         s=y1*y2;
         if s > 0
            L=max(0, alpha2+alpha1-obj.C);
            H=min(obj.C, alpha1+alpha2);
         else
            L=max(0, alpha2-alpha1);
            H=min(obj.C, obj.C+alpha2-alpha1);
         end
         if L == H
            tookStep=false;
            return
         end
         k11=obj.K(sample1,sample1);
         k12=obj.K(sample1,sample2);
         k22=obj.K(sample2,sample2);
         eta=2*k12-k11-k22;
         gamma=obj.alpha(sample1)+s*obj.alpha(sample2);
         if eta < 0
            a2=alpha2-y2*(E1-E2)/eta;
            if a2 < L
                a2=L;
            elseif a2 > H
                a2=H;
            end
         else
            Lobj=-s*L+L-0.5*k11*(gamma-s*L)^2-0.5*k22*L^2-s*k12*(gamma-s*L)*L-y1*(gamma-s*L)*(obj.f(sample1)+obj.b-y1*alpha1*k11-y2*alpha2*obj.K(sample2,sample1))-y2*L*(obj.f(sample2)+obj.b-y1*alpha1*k12-y2*alpha2*k22);
            Hobj=-s*H+H-0.5*k11*(gamma-s*H)^2-0.5*k22*H^2-s*k12*(gamma-s*H)*H-y1*(gamma-s*H)*(obj.f(sample1)+obj.b-y1*alpha1*k11-y2*alpha2*obj.K(sample2,sample1))-y2*H*(obj.f(sample2)+obj.b-y1*alpha1*k12-y2*alpha2*obj.K(sample2,sample1));
            if Lobj < Hobj - 1e-3
               a2=L;
            elseif Lobj > Hobj+1e-3
               a2=H;
            else
               a2=alpha2;
            end
         end
         % Arredonda multiplicadores às extremidades (0 e C)
         if a2 < 1e-8
            a2=0;
         elseif a2 > obj.C-1e-8
            a2=obj.C;
         end
         % Se a atualização do multiplicador for mínima, retorna que não houve ajuste
         if abs(a2-alpha2) < 1e-3 * (a2+alpha2+1e-3)
            tookStep=false;
            return
         end
         % Calcula novo valor do multiplicador alpha1
         a1=alpha1+s*(alpha2-a2);
         % Atualiza parâmetro b
         b1=E1+y1*(a1-alpha1)*k11+y2*(a2-alpha2)*k12+obj.b;
         b2=E2+y1*(a1-alpha1)*k12+y2*(a2-alpha2)*k22+obj.b;
         b_old=obj.b;
         obj.b=(b1+b2)/2.0;
         % Atualiza cache de erros
         obj.error=obj.error+y1*(a1-alpha1).*obj.K(sample1,:)'+y2*(a2-alpha2).*obj.K(sample2,:)'+b_old-obj.b;
         obj.error(sample1)=0;
         obj.error(sample2)=0;
         % Atualiza multiplicadores de Lagrange
         obj.alpha(sample1)=a1;
         obj.alpha(sample2)=a2;
         tookStep=true;
      end

      function value = f(obj,sample)
         value=sum(obj.y'.*obj.alpha'.*obj.K(sample,:))-obj.b;
      end

      function value = kernelFunc(obj,X1,X2)
         % Calcula saída da função kernel
         if obj.kernel == 'linear'
            value=sum(X1.*X2);
         elseif obj.kernel == 'rbf'
            value=exp(-obj.gamma*norm(X1-X2).^2);
         elseif self.kernel == 'polynomial'
            value=sum((obj.gamma.*X1.*X2+obj.coef0).^obj.degree);
         else
            throw(MException('kernelFunc:BadKernel',sprintf('Kernel %s not implemented', self.kernel)));
         end
      end

      function Y = predict(obj,X)
         % Rotula amostras utilizando o SVM previamente treinado

         % TODO: Aplicar scaler
         % TODO: Adicionar código multiclasses
         [nSamples,nFeatures]=size(X);
         Y=zeros(nSamples,obj.nOutputs);
         for sample1=1:nSamples
            K=zeros(nSamples,obj.nSamples);
            for sample2=1:obj.nSamples
               K(sample1,sample2)=obj.kernelFunc(X(sample1,:),obj.X(sample2,:));
            end
            value=sum(obj.y'.*obj.alpha'.*K(sample1,:))-obj.b
            if value >= 0 
               Y(sample1)=1
            else
               Y(sample1)=-1
            end
         end
         % TODO: "Traduzir classes"
      end

      function XScaled = scale(obj,X)
         % Normaliza cada feature entre -1 e 1
         XScaled=2*(X-min(X(:)))/(max(X(:))-min(X(:)))-1
   end
end